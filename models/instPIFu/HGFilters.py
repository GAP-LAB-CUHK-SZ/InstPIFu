import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.net_util import *


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, norm='batch'):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.norm = norm

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        '''
        can add attention module in here
        '''

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        # NOTE: for newer PyTorch (1.3~), it seems that training results are degraded due to implementation diff in F.grid_sample
        # if the pretrained model behaves weirdly, switch with the commented line.
        # NOTE: I also found that "bicubic" works better.
        up2 = F.interpolate(low3, size=(up1.shape[2],up1.shape[3]), mode='bicubic', align_corners=True)
        # up2 = F.interpolate(low3, scale_factor=2, mode='nearest)

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)

class PixelAtten(nn.Module):
    def __init__(self,in_channels,glo_channels,atten_channels,use_channel_atten,global_detach):
        super(PixelAtten,self).__init__()
        self.feat_conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channels+2,out_channels=atten_channels,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=atten_channels,out_channels=atten_channels,kernel_size=1)
        )
        self.glo_mlp=nn.Sequential(
            nn.Linear(in_features=glo_channels,out_features=atten_channels),
            nn.ReLU(),
            nn.Linear(in_features=atten_channels,out_features=atten_channels),
        )
        self.use_channel_atten=use_channel_atten
        self.global_detach=global_detach
        if use_channel_atten:
            self.channel_mlp=nn.Sequential(
                nn.Linear(in_features=glo_channels,out_features=in_channels),
                nn.ReLU(),
                nn.Linear(in_features=in_channels, out_features=in_channels),
                nn.ReLU(),
                nn.Linear(in_features=in_channels, out_features=in_channels),
                nn.Sigmoid(),
            )

    def forward(self,im_feat,global_feat,rel_coord):
        #print(im_feat.shape,rel_coord.shape)
        if self.global_detach:
            global_feat=global_feat.detach()
        resize_coord=F.interpolate(rel_coord,size=(im_feat.shape[2],im_feat.shape[3]),align_corners=True,mode='bilinear')
        cat_feat=torch.cat([im_feat,resize_coord],axis=1)
        query_feat=self.feat_conv(cat_feat)
        global_tran_feat=self.glo_mlp(global_feat)
        atten_weight=torch.sum(global_tran_feat[:,:,None,None]*query_feat,dim=1)/torch.norm(global_tran_feat,dim=1)[:,None,None]/torch.norm(query_feat,dim=1)
        atten_weight=torch.abs(atten_weight)
        pixel_atten_feat=atten_weight.unsqueeze(1)*im_feat
        if self.use_channel_atten:
            channel_wise_weight=self.channel_mlp(global_feat)
            #channel_atten_feat=channel_wise_weight[:,:,None,None]*atten_weight.unsqueeze(1)*im_feat
            out_feat=channel_wise_weight[:,:,None,None]*atten_weight.unsqueeze(1)*im_feat
        else:
            out_feat=pixel_atten_feat
        return out_feat,atten_weight

class HourGlass_PixelAtten(nn.Module):
    def __init__(self, num_modules, depth, num_features, norm='batch',use_channel_atten=False,global_detach=False):
        super(HourGlass_PixelAtten, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.norm = norm
        self.use_channel_atten=use_channel_atten
        self.global_detach=global_detach

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('atten_'+str(level),PixelAtten(self.features,self.features+9,self.features,use_channel_atten=self.use_channel_atten,global_detach=self.global_detach))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, level, inp,global_feat,rel_coord):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2,_ = self._forward(level - 1, low1,global_feat,rel_coord)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        '''
        can add attention module in here
        '''
        low2,atten_weight=self._modules['atten_'+str(level)](low2,global_feat,rel_coord)
        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        # NOTE: for newer PyTorch (1.3~), it seems that training results are degraded due to implementation diff in F.grid_sample
        # if the pretrained model behaves weirdly, switch with the commented line.
        # NOTE: I also found that "bicubic" works better.
        up2 = F.interpolate(low3, size=(up1.shape[2], up1.shape[3]), mode='bicubic', align_corners=True)
        # up2 = F.interpolate(low3, scale_factor=2, mode='nearest)

        return up1 + up2,atten_weight

    def forward(self, x,global_feat,rel_coord):
        return self._forward(self.depth, x,global_feat,rel_coord)

class HGFilter(nn.Module):
    def __init__(self, opt):
        super(HGFilter, self).__init__()
        self.num_modules = opt["model"]["num_stack"]

        self.opt = opt

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        if self.opt["model"]["norm"] == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.opt["model"]["norm"] == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        if self.opt["model"]["hg_down"] == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.opt["model"]["norm"])
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.opt["model"]["hg_down"] == 'conv128':
            self.conv2 = ConvBlock(64, 128, self.opt["model"]["norm"])
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.opt["model"]["hg_down"] == 'ave_pool':
            self.conv2 = ConvBlock(64, 128, self.opt["model"]["norm"])
        else:
            raise NameError('Unknown Fan Filter setting!')

        self.conv3 = ConvBlock(128, 128, self.opt["model"]["norm"])
        self.conv4 = ConvBlock(128, 256, self.opt["model"]["norm"])

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module),
                            HourGlass(1, opt["model"]["num_hourglass"], 256, self.opt["model"]["norm"]))
            self.add_module('m' + str(hg_module), HourGlass(1, opt["model"]["num_hourglass"], 256, self.opt["model"]["norm"]))

            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, self.opt["model"]["norm"]))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.opt["model"]["norm"] == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.opt["model"]["norm"] == 'group':
                self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))

            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            opt["model"]["hourglass_dim"], kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(opt["model"]["hourglass_dim"],
                                                                 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        tmpx = x
        if self.opt["model"]["hg_down"] == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.opt["model"]["hg_down"] in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')

        normx = x

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)
            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_
        return outputs, tmpx.detach(), normx

class HGFilter_pixatten(nn.Module):
    def __init__(self, opt):
        super(HGFilter_pixatten, self).__init__()
        self.num_modules = opt["model"]["num_stack"]

        self.opt = opt

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        if self.opt["model"]["norm"] == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.opt["model"]["norm"] == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        if self.opt["model"]["hg_down"] == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.opt["model"]["norm"])
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.opt["model"]["hg_down"] == 'conv128':
            self.conv2 = ConvBlock(64, 128, self.opt["model"]["norm"])
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.opt["model"]["hg_down"] == 'ave_pool':
            self.conv2 = ConvBlock(64, 128, self.opt["model"]["norm"])
        else:
            raise NameError('Unknown Fan Filter setting!')

        self.conv3 = ConvBlock(128, 128, self.opt["model"]["norm"])
        self.conv4 = ConvBlock(128, 256, self.opt["model"]["norm"])

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module),
                            HourGlass_PixelAtten(1, opt["model"]["num_hourglass"], 256, self.opt["model"]["norm"],
                                                 use_channel_atten=opt['model']['channelwise_attention'],global_detach=opt['model']['global_detach']))
            #self.add_module('m' + str(hg_module), HourGlass_(1, opt["model"]["num_hourglass"], 256, self.opt["model"]["norm"]))

            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, self.opt["model"]["norm"]))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.opt["model"]["norm"] == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.opt["model"]["norm"] == 'group':
                self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))

            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            opt["model"]["hourglass_dim"], kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(opt["model"]["hourglass_dim"],
                                                                 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x,global_feat,rel_coord):
        x = F.relu(self.bn1(self.conv1(x)), True)
        tmpx = x
        if self.opt["model"]["hg_down"] == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.opt["model"]["hg_down"] in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')

        normx = x

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        atten_weight_list=[]
        for i in range(self.num_modules):
            hg,last_atten_weight = self._modules['m' + str(i)](previous,global_feat,rel_coord)
            atten_weight_list.append(last_atten_weight)
            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_
        return outputs, tmpx.detach(), normx,atten_weight_list
