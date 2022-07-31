import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention_RoI_Module(nn.Module):
    def __init__(self,img_feat_channel=256,global_dim=256,hidden_dim=256,atten_method="dot",use_channel_atten=False,use_pixel_atten=True,global_detach=True,roi_align=True):
        super(Attention_RoI_Module,self).__init__()
        self.use_roi_align=roi_align
        self.use_pixel_atten=use_pixel_atten
        if self.use_pixel_atten:
            self.feat_conv = nn.Sequential(
                nn.Conv2d(in_channels=img_feat_channel, out_channels=hidden_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
            )
            self.glo_mlp = nn.Sequential(
                nn.Linear(in_features=global_dim, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            )
        self.use_channel_atten = use_channel_atten
        self.global_detach = global_detach
        if use_channel_atten:
            self.channel_mlp = nn.Sequential(
                nn.Linear(in_features=global_dim, out_features=img_feat_channel),
                nn.ReLU(),
                nn.Linear(in_features=img_feat_channel, out_features=img_feat_channel),
                nn.ReLU(),
                nn.Linear(in_features=img_feat_channel, out_features=img_feat_channel),
                nn.Sigmoid(),
        )
        self.post_conv=nn.Sequential(
            nn.Conv2d(in_channels=img_feat_channel, out_channels=img_feat_channel, kernel_size=5,padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=img_feat_channel, out_channels=img_feat_channel, kernel_size=5,padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=img_feat_channel, out_channels=img_feat_channel, kernel_size=5,padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=img_feat_channel, out_channels=img_feat_channel, kernel_size=5,padding=2),
        )
        self.pre_conv=nn.Sequential(
            nn.Conv2d(in_channels=img_feat_channel, out_channels=img_feat_channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=img_feat_channel, out_channels=img_feat_channel, kernel_size=1),
        )


    def forward(self,img_feat,global_feat,bdb_grid):
        if self.use_roi_align:
            roi_feat=F.grid_sample(img_feat,bdb_grid,align_corners=True,mode='bilinear')
            roi_feat = self.pre_conv(roi_feat)
        else:
            roi_feat=self.pre_conv(img_feat)
        if self.global_detach:
            global_feat=global_feat.detach()
        if self.use_channel_atten:
            channel_wise_weight=self.channel_mlp(global_feat)
        if self.use_pixel_atten:
            query_feat = self.feat_conv(roi_feat)
            global_tran_feat = self.glo_mlp(global_feat)
            atten_weight = torch.sum(global_tran_feat[:, :, None, None] * query_feat, dim=1) / torch.norm(
                global_tran_feat, dim=1)[:, None, None] / torch.norm(query_feat, dim=1)
            atten_weight = torch.abs(atten_weight)
        if self.use_channel_atten and self.use_pixel_atten:
            out_feat=atten_weight.unsqueeze(1)*channel_wise_weight.unsqueeze(2).unsqueeze(3)*roi_feat
        elif self.use_channel_atten:
            out_feat=channel_wise_weight.unsqueeze(2).unsqueeze(3)*roi_feat
        elif self.use_pixel_atten:
            out_feat=atten_weight.unsqueeze(1)*roi_feat
        #print(out_feat.shape,roi_feat.shape)
        out_feat=self.post_conv(out_feat)+roi_feat
        ret_dict={
            "roi_feat":out_feat,
        }
        if self.use_pixel_atten:
            ret_dict["pixel_atten_weight"]=atten_weight
        if self.use_channel_atten:
            ret_dict['channel_atten_weight']=channel_wise_weight
        return ret_dict
