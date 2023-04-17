from models.instPIFu.BasePIFuNet import BasePIFuNet
from models.instPIFu.SurfaceClassifier import SurfaceClassifier
from models.instPIFu.HGFilters import *
from net_utils.init_net import init_net
from skimage import measure
import trimesh
from models.instPIFu.PositionEmbedder import get_embedder
import pickle as p
from models.instPIFu.Attention_module import Attention_RoI_Module
import numpy as np
from models.modules.resnet import resnet18_full,resnet18_small_stride

def positionalEncoder(cam_points, embedder, output_dim):
    cam_points = cam_points.permute(0, 2, 1)#[B,N,3]
    inputs_flat = torch.reshape(cam_points, [-1, cam_points.shape[-1]])
    embedded = embedder(inputs_flat)
    output = torch.reshape(embedded, [cam_points.shape[0], cam_points.shape[1], output_dim])
    return output.permute(0, 2, 1)


class InstPIFu(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 error_term=nn.L1Loss(),
                 ):
        super(InstPIFu, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'hgpifu'
        self.config=opt

        self.opt = opt
        self.num_views = 1
        self.image_filter = HGFilter(opt)

        if self.config['data']['use_positional_embedding']:
            self.origin_embedder,self.embedder_outDim=get_embedder(self.config['model']['multires'],log_sampling=True)
            self.embedder=positionalEncoder

        self.surface_classifier = SurfaceClassifier(
            filter_channels=self.opt["model"]["mlp_dim"],
            num_views=1,
            no_residual=self.opt["model"]["no_residual"],
            last_op=None)

        if self.config['model']['global_recon']:
            self.global_surface_classifier=SurfaceClassifier(
                filter_channels=self.opt["model"]["global_mlp_dim"],
                num_views=1,
                no_residual=self.opt["model"]["no_residual"],
                last_op=None
            )

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []

        if not self.config["resume"]:
            init_net(self)
        if self.config['data']['use_instance_mask']:
            self.mask_decoder=nn.Sequential(
                nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
                nn.Sigmoid()
            )
        if self.config['model']['use_atten']:
            self.post_op_module=Attention_RoI_Module(img_feat_channel=256,
                                                   global_dim=256+9)
        # self.global_encoder=nn.Sequential(
        #     nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2,2), #32
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2), #16
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2), # 8
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),  # 4
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(256,256),
        #     nn.ReLU(),
        #     nn.Linear(256,256)
        # )
        self.global_encoder = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=64,kernel_size=3,padding=1),
            resnet18_small_stride(pretrained=False,input_channels=64),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def filter(self, images,patch):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        #if not self.training:
        #    self.im_feat_list = [self.im_feat_list[-1]]

    def query(self, points, z_feat,img_coor, bdb_grid,cls_codes,transforms=None, labels=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, N, 3] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param cls_codes: [B,9]
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        if labels is not None:
            self.labels = labels

        x_coor = img_coor[:, :,0]
        y_coor = img_coor[:, :,1]
        '''
        p_project is B,7,NUM_SAM,2
        '''
        xy=torch.cat([x_coor[:,:,None],y_coor[:,:,None]],dim=2) #B,NUM_SAM,2
        self.z_feat=z_feat
        '''extract global feature from feature map from hourglass network'''
        '''try it from the first layer'''

        if self.opt["model"]["skip_hourglass"]:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []

        '''reconstruction using only the global feature'''
        if self.config['data']['use_positional_embedding']:
            position_feat = self.embedder(points.transpose(1,2), self.origin_embedder, self.embedder_outDim)
            point_global_feat_list = [position_feat, z_feat.transpose(1, 2),
                                     cls_codes.unsqueeze(2).repeat(1, 1, points.shape[1])]
        else:
            point_global_feat_list = [points.transpose(1,2),z_feat.transpose(1,2),cls_codes.unsqueeze(2).repeat(1,1,points.shape[1])]
            #print(point_local_feat_list[0].shape,points.shape)
        point_global_feat_list.append(self.global_feat.unsqueeze(2).repeat(1, 1, points.shape[1]))
        global_point_feat=torch.cat(point_global_feat_list,dim=1)
        global_pred=self.global_surface_classifier(global_point_feat)

        self.intermediate_preds_list.append(global_pred)
        self.mask_list=[]
        self.channel_atten_list=[]
        # if self.training==False:
        #     self.im_feat_list=[self.im_feat_list[-1]]
        if self.training:
            input_im_feat=self.im_feat_list[0:-1]
        else:
            input_im_feat=self.im_feat_list[-2:-1]
        for im_feat in input_im_feat:
            if self.config['model']['use_atten']:
                ret_dict=self.post_op_module(im_feat,torch.cat([self.global_feat,cls_codes],dim=1),bdb_grid)
                roi_feat=ret_dict["roi_feat"]
                self.channel_atten_list.append(ret_dict['channel_atten_weight'])
            else:
                roi_feat = F.grid_sample(im_feat, bdb_grid, align_corners=True, mode='bilinear')
            if self.config['data']['use_instance_mask']:
                pred_mask=self.mask_decoder(roi_feat)
                self.mask_list.append(pred_mask)
            if self.config['data']['use_positional_embedding']:
                position_feat = self.embedder(points.transpose(1,2), self.origin_embedder, self.embedder_outDim)
                point_local_feat_list = [self.index(roi_feat, xy),position_feat, z_feat.transpose(1, 2),
                                         cls_codes.unsqueeze(2).repeat(1, 1, points.shape[1])]
            else:
                point_local_feat_list = [self.index(roi_feat, xy), points.transpose(1,2),z_feat.transpose(1,2),cls_codes.unsqueeze(2).repeat(1,1,points.shape[1])]

            if self.opt["model"]["skip_hourglass"]:
                point_local_feat_list.append(tmpx_local_feature)
            point_local_feat_list.append(self.global_feat.unsqueeze(2).repeat(1,1,points.shape[1]))
            point_local_feat = torch.cat(point_local_feat_list, 1)

            pred=self.surface_classifier(point_local_feat)
            self.intermediate_preds_list.append(pred)
        #print(len(self.intermediate_preds_list))
        self.preds = self.intermediate_preds_list[-1].squeeze(1)


    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def get_error(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        if self.config['data']['dataset']=="pix3d_recon":
            nss_error = 0
            uni_error = 0
            for preds in self.intermediate_preds_list:
                #print(preds.shape,self.labels.shape)
                nss_error += self.error_term(preds[:,0,0:2048],self.labels[:,0:2048])
                uni_error += self.error_term(preds[:,0,2048:],self.labels[:,2048:])
            nss_error /= len(self.intermediate_preds_list)
            uni_error /= len(self.intermediate_preds_list)
            self.nss_recon_loss=nss_error
            self.uni_recon_loss=uni_error
            mask_error=0
            for pred_mask in self.mask_list:
                mask_error+=nn.MSELoss()(pred_mask,F.interpolate(self.mask_label,size=(pred_mask.shape[2],pred_mask.shape[3]),mode="nearest"))
            self.mask_loss=mask_error/len(self.mask_list)
            return nss_error*0.1+uni_error*1 + self.mask_loss
        elif self.config['data']['dataset']=="front3d_recon":
            error = 0
            for preds in self.intermediate_preds_list:
                # print(preds.shape,self.labels.shape)
                error += self.error_term(preds, self.labels)
            error /= len(self.intermediate_preds_list)
            self.recon_loss = error
            mask_error = 0
            for pred_mask in self.mask_list:
                mask_error += nn.MSELoss()(pred_mask,
                                           F.interpolate(self.mask_label, size=(pred_mask.shape[2], pred_mask.shape[3]),
                                                         mode="nearest"))
            self.mask_loss = mask_error / len(self.mask_list)
            return error + self.mask_loss

    def forward(self, data_dict):
        # Get image feature
        images, points, labels,img_coor,cls_codes = data_dict["whole_image"],data_dict["samples"],data_dict["inside_class"],data_dict["img_coor"],data_dict["cls_codes"]
        patch=data_dict["patch"]
        z_feat=data_dict["z_feat"]
        bdb_grid=data_dict['bdb_grid']
        self.mask_label=data_dict["mask"]

        transforms = None
        self.filter(images,patch)
        last_roi_feat = F.grid_sample(self.im_feat_list[0], bdb_grid, align_corners=True, mode='bilinear')
        self.global_feat = self.global_encoder(last_roi_feat)
        # Phase 2: point query
        self.query(points=points, z_feat=z_feat,bdb_grid=bdb_grid,transforms=transforms,cls_codes=cls_codes, labels=labels,img_coor=img_coor)#,depth=depth)

        # get the prediction
        res = self.get_preds()
        #print(res)

        # get the error
        error = self.get_error()

        pred_occ=torch.zeros(res.shape).to(res.device)
        pred_occ[res>0.5]=1
        pred_occ[res<0.5]=0
        if self.config['data']['dataset'] == "pix3d_recon":
            nss_pred_acc=torch.mean(1-torch.abs(pred_occ[:,0:2048]-self.labels[:,0:2048]))
            uni_pred_acc=torch.mean(1-torch.abs(pred_occ[:,2048:]-self.labels[:,2048:]))
            loss_info = {
                "loss": error * 10,
                "nss_pred_acc": nss_pred_acc,
                "uni_pred_acc": uni_pred_acc,
                "nss_recon_loss": self.nss_recon_loss,
                "uni_recon_loss": self.uni_recon_loss
            }
        elif self.config['data']['dataset'] == "front3d_recon":
            pred_acc=torch.mean(1-torch.abs(pred_occ-self.labels))
            loss_info = {
                "loss": error,
                "pred_acc": pred_acc,
                "recon_loss": self.recon_loss,
            }
            loss_info['mask_loss'] = self.mask_loss
            atten_weight = self.channel_atten_list[-1]
            max_atten = torch.max(atten_weight)
            min_atten = torch.min(atten_weight)
            loss_info["max_atten"] = max_atten
            loss_info["min_atten"] = min_atten
        ret_dict={
            "pred_class":res,
        }
        ret_dict["pred_mask"]=self.mask_list[-1]
        return ret_dict,loss_info
    def extract_mesh(self,data_dict,marching_cube_resolution=64):
        whole_image,image, cls_codes =data_dict["whole_image"],data_dict["image"],data_dict["cls_codes"]
        patch = data_dict["patch"]
        if self.config['data']['dataset']=='pix3d_recon':
            K=data_dict['org_K']
        else:
            K=data_dict["K"]
        rot_matrix=data_dict["rot_matrix"]
        bbox_size=data_dict["bbox_size"]
        obj_cam_center=data_dict["obj_cam_center"]
        bdb_grid=data_dict['bdb_grid']
        transforms = None
        self.filter(whole_image, patch)
        last_roi_feat = F.grid_sample(self.im_feat_list[0], bdb_grid, align_corners=True, mode='bilinear')
        self.global_feat = self.global_encoder(last_roi_feat)

        x_coor = torch.linspace(-1.2, 1.2, steps=marching_cube_resolution).float().to(image.device)
        y_coor = torch.linspace(-1.2, 1.2, steps=marching_cube_resolution).float().to(image.device)
        z_coor = torch.linspace(-1.2, 1.2, steps=marching_cube_resolution).float().to(image.device)
        X, Y, Z = torch.meshgrid(x_coor, y_coor, z_coor)

        samples_incan = torch.cat([X[:, :, :, None], Y[:, :, :, None], Z[:, :, :, None]], dim=3).unsqueeze(0)
        samples_incan=samples_incan.view(samples_incan.shape[0],marching_cube_resolution**3,3)

        '''adds some test sample to debug'''
        if self.config['debug']:
            test_samples_incan = data_dict["samples"]
            samples_incan=torch.cat([samples_incan,test_samples_incan],dim=1)


        samples_inrecan = torch.einsum('ijk,ikq->ijq',samples_incan, rot_matrix.transpose(1,2))
        z_feat=samples_inrecan[:,:,2:3]

        samples_incam=samples_incan*bbox_size.unsqueeze(1)/2
        samples_incam=torch.einsum('ijk,ikq->ijq',samples_incam,rot_matrix.transpose(1,2))
        samples_incam[:,:,0:2]=-samples_incam[:,:,0:2] #y down coordinate
        samples_incam[:,:,0:3]=samples_incam[:,:,0:3]+obj_cam_center

        img_samples=torch.einsum('ijk,ikq->ijq',samples_incam[:,:,0:3],K.transpose(1,2))
        width=K[:,0,2]*2
        height=K[:,1,2]*2
        x_coor=img_samples[:,:,0]/img_samples[:,:,2] #these are image coordinate
        y_coor=img_samples[:,:,1]/img_samples[:,:,2] #these are image coordinate
        if self.config['data']['use_crop']:
            bdb2D=data_dict['bdb2D_pos']
            x_coor=x_coor-(bdb2D[0,0]+bdb2D[0,2])/2
            y_coor=y_coor-(bdb2D[0,1]+bdb2D[0,3])/2
            x_coor=x_coor/(bdb2D[0,2]-bdb2D[0,0])*2
            y_coor=y_coor/(bdb2D[0,3]-bdb2D[0,1])*2
        else:
            x_coor=((x_coor-width/2)/width)*2
            y_coor=((y_coor-height/2)/height)*2
        #print(x_coor)
        img_coor=torch.cat([x_coor[:,:,None],y_coor[:,:,None]],dim=2)

        '''conduct test on prepared sampled'''
        if self.config['debug']:
            #print(samples_incam[:,-4096:].shape,img_coor[:,0:-4096:].shape,data_dict["inside_class"].shape)
            self.query(points=samples_incan[:,-4096:],z_feat=z_feat[:,-4096:,],transforms=transforms,cls_codes=cls_codes,labels=data_dict["inside_class"],
                       img_coor=img_coor[:,-4096:],bdb_grid=bdb_grid)
            res=self.get_preds()
            pred_occ = torch.zeros(res.shape).to(res.device)
            pred_occ[res > 0.5] = 1
            pred_occ[res < 0.5] = 0
            pred_acc = torch.mean(1 - torch.abs(pred_occ - self.labels))
            print("debuging test accuracy is %f"%(pred_acc))
        # Phase 2: point query
        if self.config['debug']:
            sample_list = torch.split(samples_incan[:,0:-4096], 200000, dim=1)
            img_coor_list = torch.split(img_coor[:,0:-4096], 200000, dim=1)
            z_feat_list = torch.split(z_feat[:,0:-4096], 200000, dim=1)
        else:
            sample_list = torch.split(samples_incan, 200000, dim=1)
            img_coor_list = torch.split(img_coor, 200000, dim=1)
            z_feat_list = torch.split(z_feat, 200000, dim=1)
        pred_list = []
        for i in range(len(sample_list)):
            # Phase 2: point query
            self.query(points=sample_list[i], z_feat=z_feat_list[i], transforms=transforms, cls_codes=cls_codes,
                       img_coor=img_coor_list[i],bdb_grid=bdb_grid)

            res = self.get_preds()
            pred_list.append(res)
        pred = torch.cat(pred_list, dim=1)
        # get the prediction
        pred=pred.view(-1,marching_cube_resolution,marching_cube_resolution,marching_cube_resolution).squeeze(0).detach().cpu().numpy()
        mesh=self.marching_cubes(pred,mcubes_extent=(1.2,1.2,1.2))[1]
        clean_mesh=self.delete_disconnected_component(mesh)
        return clean_mesh
    def delete_disconnected_component(self,mesh):

        split_mesh = mesh.split(only_watertight=False)
        max_vertice = 0
        max_ind = -1
        for idx, mesh in enumerate(split_mesh):
            # print(mesh.vertices.shape[0])
            if mesh.vertices.shape[0] > max_vertice:
                max_vertice = mesh.vertices.shape[0]
                max_ind = idx
        # print(max_ind)
        # print(max_vertice)
        return split_mesh[max_ind]

    def marching_cubes(self,volume, mcubes_extent):
        """Maps from a voxel grid of implicit surface samples to a Trimesh mesh."""
        volume = np.squeeze(volume)
        length, height, width = volume.shape
        resolution = length
        # This function doesn't support non-cube volumes:
        assert resolution == height and resolution == width
        thresh = 0.5
        try:
            vertices, faces, normals, _ = measure.marching_cubes(volume, thresh)
            del normals
            x, y, z = [np.array(x) for x in zip(*vertices)]
            xyzw = np.stack([x, y, z, np.ones_like(x)], axis=1)
            # Center the volume around the origin:
            xyzw += np.array(
                [[-resolution / 2.0, -resolution / 2.0, -resolution / 2.0, 0.]])
            # This assumes the world is right handed with y up; matplotlib's renderer
            # has z up and is left handed:
            # Reflect across z, rotate about x, and rescale to [-0.5, 0.5].
            xyzw *= np.array([[(2.0 * mcubes_extent[0]) / resolution,
                               (2.0 * mcubes_extent[1]) / resolution,
                               (2.0 * mcubes_extent[2]) / resolution, 1]])
            # y_up_to_z_up = np.array([[0., 0., -1., 0.], [0., 1., 0., 0.],
            #                         [1., 0., 0., 0.], [0., 0., 0., 1.]])
            # xyzw = np.matmul(y_up_to_z_up, xyzw.T).T
            faces = np.stack([faces[..., 0], faces[..., 2], faces[..., 1]], axis=-1)
            world_space_xyz = np.copy(xyzw[:, :3])
            mesh = trimesh.Trimesh(vertices=world_space_xyz, faces=faces)
            return True, mesh
        except (ValueError, RuntimeError) as e:
            print(
                'Failed to extract mesh with error %s. Setting to unit sphere.' %
                repr(e))
            return False, trimesh.primitives.Sphere(radius=0.5)