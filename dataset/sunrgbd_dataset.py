import os
from torch.utils.data import Dataset
import json
import pickle
from PIL import Image
import numpy as np
import torch
from .front3d_recon_dataset import R_from_yaw_pitch_roll,get_centroid_from_proj
from torchvision import transforms
from net_utils.bins import *
from scipy import io

sunrgbd_front_label_mapping={
    3:2,
    4:6,
    5:4,
    6:1,
    7:0,
    10:5,
    14:7,
    17:8,
    32:3
}

HEIGHT_PATCH = 256
WIDTH_PATCH = 256

data_transforms_crop = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_image=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

sunrgbd_layout_path="./data/sunrgbd/preprocessed/layout_avg_file.pkl"
with open(sunrgbd_layout_path,'rb') as f:
    sunrgbd_avg_layout=p.load(f)
sunrgbd_avgsize_path="./data/sunrgbd/preprocessed/size_avg_category.pkl"
with open(sunrgbd_avgsize_path,'rb') as f:
    sunrgbd_avgsize=p.load(f)


class SUNRGBD_Recon_Dataset(Dataset):
    def __init__(self, config, mode,testid=None):
        super(SUNRGBD_Recon_Dataset, self).__init__()

        self.config = config
        self.mode = mode
        split_file = config['data']['split']
        with open(split_file) as file:
            self.split = json.load(file)
        if testid is not None:
            new_split=[]
            for item in self.split:
                id = item[0].split("/")[-1].split(".")[0]
                if id==testid:
                    new_split.append(item)
            self.split=new_split
        # select_id_list=[1180,1202,1206,1207,140,2040,2041,2043,330,571,182,179,181,784,928
        # ,930,941,1185,1176,1174,1195,1281,1426,1451,1567,1747,1753,1900,2207,2398,2472,2586
        # ,2768,3385,4082,4086,4302,4650,5000,5003]
        # new_split=[]
        # for item in self.split:
        #     id = int(item[0].split("/")[-1].split(".")[0])
        #     if id in select_id_list:
        #         new_split.append(item)
        # self.split=new_split
    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        if isinstance(self.split[index],list):
            file_path,object_id = self.split[index]
            #print(file_path)
            with open(file_path, 'rb') as f:
                sequence = pickle.load(f)
            image = Image.fromarray(sequence['rgb_img'])
            width,height=image.size
            depth = Image.fromarray(sequence['depth_map'])
            camera = sequence['camera']
            boxes = sequence['boxes']

            bdb=boxes['bdb2D_pos'][object_id]
            cls_codes = torch.zeros([9])
            cls_codes[sunrgbd_front_label_mapping[boxes['size_cls'][object_id]]] = 1

            patch=image.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
            patch=data_transforms_crop(patch)

            pitch_cls,pitch_reg=camera['pitch_cls'],camera['pitch_reg']
            roll_cls,roll_reg=camera['roll_cls'],camera['roll_reg']
            ori_cls,ori_reg=boxes['ori_cls'][object_id],boxes['ori_reg'][object_id]

            pitch=np.mean(bin['pitch_bin'][pitch_cls])+pitch_reg*PITCH_WIDTH
            roll=np.mean(bin['roll_bin'][roll_cls])+roll_reg*ROLL_WIDTH
            yaw=np.mean(bin['ori_bin'][ori_cls])+ori_reg*ORI_BIN_WIDTH-np.pi/2

            yaw=yaw
            rot_matrix=R_from_yaw_pitch_roll(yaw,pitch,-roll)

            if self.config['data']['use_pred_pose']:
                pred_pose_path=os.path.join(self.config['data']['pred_pose_path'],str(sequence['sequence_id']),"bdb_3d.mat")
                layout_path=os.path.join(self.config['data']['pred_pose_path'],str(sequence['sequence_id']),"layout.mat")
                camera_path=os.path.join(self.config['data']['pred_pose_path'],str(sequence['sequence_id']),"r_ex.mat")
                camera_content = io.loadmat(camera_path)['cam_R']
                camR = np.array(camera_content)
                bdb_3d_content=io.loadmat(pred_pose_path)['bdb']
                layout_content=io.loadmat(layout_path)
                bdb3d=bdb_3d_content[0][int(object_id)][0][0]
                yaw_rot=bdb3d[0]
                half_rot=R_from_yaw_pitch_roll(-np.pi/2,0,0)
                yaw_rot=np.dot(np.linalg.inv(yaw_rot),half_rot)
                bbox_size=bdb3d[1][0]*2
                #print(bbox_size)
                center=bdb3d[2][0][[2,1,0]]
                center[1]=-center[1]
                pred_pitch=layout_content['pitch'][0][0]
                pred_roll=layout_content['roll'][0][0]
                pred_rot_matrix=np.dot(R_from_yaw_pitch_roll(0,pred_pitch,-pred_roll),yaw_rot)
                obj_cam_center=np.dot(center,np.linalg.inv(camR).T)
                #print(obj_cam_center)
                rot_matrix=pred_rot_matrix

            debug_points = np.array([-1, -1, -1])
            debug_zfeat = np.dot(debug_points, rot_matrix.T)
            delta2d = boxes['delta_2D'][object_id]
            bdb2D = boxes['bdb2D_pos'][object_id]
            project_center = np.zeros([2])
            project_center[0] = (bdb2D[0] + bdb2D[2]) / 2 - delta2d[0] * (bdb2D[2] - bdb2D[0])
            project_center[1] = (bdb2D[1] + bdb2D[3]) / 2 - delta2d[1] * (bdb2D[3] - bdb2D[1])

            bdb_x = np.linspace(bdb[0], bdb[2], 64)
            bdb_y = np.linspace(bdb[1], bdb[3], 64)
            bdb_X, bdb_Y = np.meshgrid(bdb_x, bdb_y)
            bdb_X = (bdb_X - width/2) / width*2
            bdb_Y = (bdb_Y - height/2) / height*2
            bdb_grid = np.concatenate([bdb_X[:, :, np.newaxis], bdb_Y[:, :, np.newaxis]], axis=-1)

            centroid_cls, centroid_reg = boxes['centroid_cls'][object_id], boxes['centroid_reg'][object_id]
            centroid_depth = np.mean(bin['centroid_bin'][centroid_cls]) + centroid_reg * DEPTH_WIDTH


            size_reg=boxes['size_reg'][object_id]
            avg_size=sunrgbd_avgsize[boxes['size_cls'][object_id]]
            if not self.config['data']['use_pred_pose']:
                bbox_size=(1+size_reg)*avg_size*2

            #print(bbox_size)

            orientation=[pitch,roll,yaw]

            bdb_center = np.array([(bdb2D[0] + bdb2D[2]) / 2 / width, (bdb2D[1] + bdb2D[3]) / 2 / height])
            if self.config['data']['use_crop']:
                bdb_center=np.array([0.5,0.5])
            img_x = np.linspace(0, 1, width // 4)
            img_y = np.linspace(0, 1, height // 4)
            img_X, img_Y = np.meshgrid(img_x, img_y)
            rel_coord = np.concatenate([img_X[:, :, np.newaxis], img_Y[:, :, np.newaxis]], axis=2)
            rel_coord = rel_coord - bdb_center[np.newaxis, np.newaxis, :]
            rel_coord = rel_coord.transpose([2, 0, 1])
            box_info={
                "project_center":project_center,
                "centroid_depth":centroid_depth,
                "size":bbox_size,
                "orientation":orientation
            }
            intrinsic = camera['K']
            #print(intrinsic)
            org_intrinsic=intrinsic.copy()
            '''scale the image, so that the focal length is the same as 3d front dataset'''
            target_f = 584
            current_f = intrinsic[0, 0]
            scale_factor=current_f/target_f
            intrinsic[0] = intrinsic[0] / scale_factor
            intrinsic[1] = intrinsic[1] / scale_factor
            camera['K'] = intrinsic
            obj_cam_center=get_centroid_from_proj(centroid_depth, project_center, org_intrinsic)
            objrecon_image=image.resize(size=(int(width/scale_factor),int(height/scale_factor)))
            objrecon_image = data_transforms_image(objrecon_image)
            depth=np.asarray(depth)
            bdb=bdb/scale_factor

            '''construct input for bg PIFu'''
            bg_width=268
            bg_height=200
            bg_intrinsic=org_intrinsic.copy()
            scale_width = width / bg_width
            scale_height = height / bg_height
            bg_intrinsic[0] = bg_intrinsic[0] / scale_width
            bg_intrinsic[1] = bg_intrinsic[1] / scale_height
            #print(image.size)
            bg_image=image.resize(size=(bg_width,bg_height))
            bg_image = data_transforms_image(bg_image)

            #for debug usage
            #print(obj_cam_center)
            data_batch={"whole_image":objrecon_image.float(),'bg_image':bg_image.float(),'image':patch.float(),'bg_intrinsic':bg_intrinsic,'depth':depth, "K":camera['K'],"intrinsic":camera["K"],'bbox_size':bbox_size,'bdb2D_pos':bdb,
                        'org_intrinsic':org_intrinsic,'taskid':str(sequence['sequence_id']),'jid':"sunrgbd",'obj_id':str(object_id),
                        'patch':patch,'rot_matrix':rot_matrix,'obj_cam_center':obj_cam_center,
                        'cls_codes':cls_codes,'rel_coord':rel_coord,'bdb_grid':bdb_grid}

        return data_batch