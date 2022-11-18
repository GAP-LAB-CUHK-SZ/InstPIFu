import torch
import torch.nn as nn
import torch.nn.functional as F
from models.instPIFu.BasePIFuNet import BasePIFuNet
from models.instPIFu.SurfaceClassifier import SurfaceClassifier
from models.instPIFu.HGFilters import *
from net_utils.init_net import init_net
from models.modules.resnet import resnet18_full,resnet18
from skimage import measure
import trimesh
from models.instPIFu.PositionEmbedder import get_embedder
from models.modules.resnet import model_urls
import torch.utils.model_zoo as model_zoo

def positionalEncoder(cam_points, embedder, output_dim):
    cam_points = cam_points.permute(0, 2, 1)#[B,N,3]
    inputs_flat = torch.reshape(cam_points, [-1, cam_points.shape[-1]])
    embedded = embedder(inputs_flat)
    output = torch.reshape(embedded, [cam_points.shape[0], cam_points.shape[1], output_dim])
    return output.permute(0, 2, 1)


class HGPIFuNet(BasePIFuNet):
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
        super(HGPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'hgpifu'
        self.config=opt

        self.opt = opt
        self.num_views = 1

        self.image_filter = HGFilter(opt)

        self.surface_classifier = SurfaceClassifier(
            filter_channels=self.opt["model"]["mlp_dim"],
            num_views=1,
            no_residual=self.opt["model"]["no_residual"],
            last_op=None)

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []

        if not self.config["resume"]:
            init_net(self)

        self.global_encoder=resnet18_full(pretrained=False)
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = self.global_encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)

        if self.config['data']['use_positional_embedding']:
            self.origin_embedder,self.embedder_outDim=get_embedder(self.config['data']['multires'],log_sampling=False,input_dim=3)
            self.embedder=positionalEncoder

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        self.global_feat=self.global_encoder(images)
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def query(self, points, intrinsic, height,width, rot_matrix=None, M=None ,transforms=None, labels=None):
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
        img_coor = torch.einsum("ijk,ikq->ijq", points, intrinsic[:,0:3,0:3].transpose(1, 2))
        #img_coor = torch.einsum("ijk,ikl->ijl", points, intrinsic[:,0:3,0:3].transpose(1, 2))
        x_coor = img_coor[:, :,0]/img_coor[:,:,2]
        y_coor = img_coor[:, :,1]/img_coor[:,:,2]

        if rot_matrix is not None:
            coor = torch.cat([x_coor[:, :, None], y_coor[:, :, None], torch.ones([x_coor.shape[0],
                                                                                        x_coor.shape[1],
                                                                                        1]).to(points.device)], dim=-1)
            coor = torch.einsum("ijk,ikq->ijq", coor, rot_matrix.transpose(1, 2))
            x_coor = coor[:, :, 0]
            y_coor = coor[:, :, 1]
        p_project = torch.cat([(x_coor[:, :, None] / (width - 1) - 0.5) * 2,
                               (y_coor[:, :, None] / (height - 1) - 0.5) * 2],
                              dim=-1).float() #B,NUM_SAM,2
        '''
        p_project is B,7,NUM_SAM,2
        '''
        xy=p_project #B,NUM_SAM,2
        self.in_img = (xy[:, :, 0] >= -1.0) & (xy[:, :, 0] <= 1.0) & (xy[:, :, 1] >= -1.0) & (xy[:, :, 1] <= 1.0)
        z_feat=img_coor[:,:,2:3]
        self.z_feat=z_feat

        if self.opt["model"]["skip_hourglass"]:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []

        for im_feat in self.im_feat_list:
            if M is not None:
                points_feat= torch.einsum('ijk,ikq->ijq',points,M.transpose(1,2))# augmentation
            else:
                points_feat=points
            if self.config['data']['use_positional_embedding']:
                position_feat=self.embedder(points_feat.transpose(1,2),self.origin_embedder,self.embedder_outDim)
                point_local_feat_list = [self.index(im_feat, xy), position_feat]
            else:
                point_local_feat_list = [self.index(im_feat, xy), points_feat.transpose(1,2)]

            if self.opt["model"]["skip_hourglass"]:
                point_local_feat_list.append(tmpx_local_feature)
            point_local_feat_list.append(self.global_feat.unsqueeze(2).repeat(1,1,points.shape[1]))
            point_local_feat = torch.cat(point_local_feat_list, 1)

            pred=self.surface_classifier(point_local_feat)#*self.in_img[:,None].float()
            self.intermediate_preds_list.append(pred)
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
        error = 0
        for preds in self.intermediate_preds_list:
            error += self.error_term(preds.squeeze(1),self.labels)
        error /= len(self.intermediate_preds_list)

        return error

    def delete_disconnected_component(self,mesh):

        split_mesh = mesh.split(only_watertight=False)
        max_vertice = 0
        max_ind = -1
        for idx, mesh in enumerate(split_mesh):
            # print(mesh.vertices.shape[0])
            if mesh.vertices.shape[0] > max_vertice:
                max_vertice = mesh.vertices.shape[0]
                max_ind = idx
        return split_mesh[max_ind]

    def forward(self, data_dict):
        # Get image feature
        #rot matrix and M is for data augmentation
        images,points,intrinsic,rot_matrix,M = data_dict["image"],data_dict["sample_points"],data_dict["intrinsic"],data_dict["rot_matrix"],data_dict["M"]
        label=data_dict["label"]
        self.labels=label
        transforms = None
        self.filter(images)
        height,width=images.shape[2:4]

        # Phase 2: point query
        self.query(points=points,intrinsic=intrinsic,rot_matrix=rot_matrix,M=M,transforms=transforms,height=height,width=width)#,depth=depth)
        # get the prediction
        res = self.get_preds()
        #print(res)

        # get the error
        error = self.get_error()

        pred_occ=torch.zeros(res.shape).to(res.device)
        pred_occ[res>0.5]=1
        pred_occ[res<0.5]=0
        pred_acc=torch.mean(1-torch.abs(pred_occ-self.labels))

        ret_dict={
            "pred_class":res
        }
        loss_info={
            "loss":error,
            "pred_acc":pred_acc
        }

        return ret_dict,loss_info
    def extract_mesh(self,data_dict,marching_cube_resolution=64):
        image = data_dict["image"]
        height, width = image.shape[2:4]
        K=data_dict["intrinsic"]
        self.filter(image)
        volumn = torch.ones((marching_cube_resolution, marching_cube_resolution, marching_cube_resolution)).float().to(
            image.device)
        x_coor = torch.linspace(-3, 3, steps=marching_cube_resolution).float()
        y_coor = torch.linspace(-2, 2, steps=marching_cube_resolution).float()
        z_coor = torch.linspace(1, 10, steps=marching_cube_resolution).float()
        X, Y, Z = torch.meshgrid(x_coor, y_coor, z_coor)
        samples_incam = torch.cat([X[:, :, :, None], Y[:, :, :, None], Z[:, :, :, None]], dim=3).unsqueeze(0)
        samples_incam=samples_incam.reshape(samples_incam.shape[0],-1,3)

        project_sample = torch.einsum("ijk,ikq->ijq", samples_incam, data_dict["intrinsic"][:, 0:3, 0:3].transpose(1, 2).cpu())
        project_x = project_sample[:, :, 0] / project_sample[:, :, 2]
        project_y = project_sample[:, :, 1] / project_sample[:, :, 2]
        visible_ind = torch.where(
            (project_x <= width-1) & (project_x > 0) & (project_y > 0) & (project_y <= height-1) & (project_sample[:, :, 2] > 0))
        visible_sample=samples_incam[visible_ind[0],visible_ind[1],:].unsqueeze(0)
        sample_list=torch.split(visible_sample,200000,dim=1)
        #print(visible_sample.shape,K.shape)
        # Phase 2: point query
        for i,sample in enumerate(sample_list):
            #print(sample.shape)
            self.query(points=sample.to(image.device),intrinsic=K,height=height,width=width)

            res = self.get_preds()
            volumn = volumn.view(-1, 1)
            if i < len(sample_list)-1:
                volumn[visible_ind[1][i*200000:(i+1)*200000], :] = res.squeeze(0).unsqueeze(1)
            else:
                volumn[visible_ind[1][i*200000:], :] = res.squeeze(0).unsqueeze(1)
        volumn = volumn.view(marching_cube_resolution, marching_cube_resolution,
                             marching_cube_resolution).detach().cpu().numpy()
        volumn=1-volumn
        mesh=self.marching_cubes(volumn,mcubes_extent=(3,2,4.5))[1]

        #vertices=mesh.vertices
        mesh=self.delete_invisible_vert(mesh,K,height,width)
        #mesh=self.delete_disconnected_component(mesh)
        return mesh

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
                [[-(resolution-1) / 2.0, -(resolution-1) / 2.0, 0, 0.]])
            xyzw *= np.array([[(2.0 * mcubes_extent[0]) / (resolution-1),
                               (2.0 * mcubes_extent[1]) / (resolution-1),
                               (2.0 * mcubes_extent[2]) / (resolution-1), 1]])
            xyzw[:,2]+=1
            faces = np.stack([faces[..., 0], faces[..., 1], faces[..., 2]], axis=-1)
            world_space_xyz = np.copy(xyzw[:, :3])
            mesh = trimesh.Trimesh(vertices=world_space_xyz, faces=faces)
            return True, mesh
        except (ValueError, RuntimeError) as e:
            print(
                'Failed to extract mesh with error %s. Setting to unit sphere.' %
                repr(e))
            return False, trimesh.primitives.Sphere(radius=0.5)
    def delete_invisible_vert(self,mesh,intrinsic,height,width):
        vertices=mesh.vertices
        faces=mesh.faces
        intrinsic=intrinsic.squeeze(0).cpu().numpy()
        img_coor=np.dot(vertices,intrinsic[0:3,0:3].T)
        x_coor=img_coor[:,0]/img_coor[:,2]
        y_coor=img_coor[:,1]/img_coor[:,2]
        #print(np.min(img_coor[:,2]))
        select_vert=(x_coor<=width-3) & (x_coor>=2) & (y_coor<=height-3) & (y_coor>=2)
        select_vertices_ind=np.where(select_vert)[0]
        select_face=np.in1d(faces.reshape(-1),select_vertices_ind)
        select_face = select_face.reshape(-1, 3)
        select_face=select_face[:,0]&select_face[:,1]&select_face[:,2]
        select_face_mask=(select_face==1)[:,np.newaxis].all(axis=1)

        select_vert_mask=(select_vert==1)
        mesh.update_vertices(select_vert_mask)
        mesh.update_faces(select_face_mask)
        return mesh
