import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--instpifu_dir",type=str,default="/apdcephfs/private_haolinliu/InstPIFu")
parser.add_argument("--taskid",type=str,default="2003")
args= parser.parse_args()
import os,sys
#sys.path.append("..")
sys.path.append(args.instpifu_dir)
os.chdir(args.instpifu_dir)
import torch
import numpy as np
import datetime
import time
import cv2
from scipy import io
import json
from configs.config_utils import CONFIG
from models.instPIFu.InstPIFu_net import InstPIFu
from dataset.front3d_recon_dataset import R_from_yaw_pitch_roll,get_centroid_from_proj
from dataset.pix3d_recon_dataset import data_transforms_crop,data_transforms
from net_utils.bins import *
from PIL import Image
import copy
import tqdm
import pickle as p

'''use pix3d label mapping, since the model is trained on pix3d dataset'''
category_label_mapping = {"table":0,
                          "sofa": 1,
                          "misc": 2,
                          "tool":3,
                          "chair":4,
                          "bookcase":5,
                          "bed":6,
                          "desk":7,
                          "wardrobe":8
                          }
def prepare_data(data_folder):
    img_path=os.path.join(data_folder,"img.jpg")
    detection_img_path=os.path.join(data_folder,"2d_detection.jpg") #only for visualization
    detection_path=os.path.join(data_folder,"detections.json")
    layout_pred_path=os.path.join(data_folder,"layout.mat")
    camera_path=os.path.join(data_folder,"r_ex.mat")
    pred_pose_path = os.path.join(data_folder,"bdb_3d.mat")
    K_path=os.path.join(data_folder,"cam_K.txt")

    image=Image.open(img_path)
    detection_img=cv2.imread(detection_img_path) #only for visualization
    width,height=image.size
    max_length=max(width,height) #use to conduct padding, pad a square image with length=max_length

    camera_content=io.loadmat(camera_path)
    #print(camera_content)
    pred_pitch=np.array(camera_content["pitch"])
    pred_roll=np.array(camera_content["roll"])
    camR=np.array(camera_content['cam_R'])
    bdb_3d_content=io.loadmat(pred_pose_path)['bdb'][0]
    with open(detection_path,'r') as f:
        detection_2d=json.load(f)
    K=np.loadtxt(K_path)

    assert len(bdb_3d_content)==len(detection_2d)

    patches=[]
    cls_codes=[]
    rot_matrices=[]
    bbox_sizes=[]
    obj_cam_centers=[]
    bdb_grids=[]
    bdb2D_pos=[]

    for i in range(len(bdb_3d_content)):
        bdb3d=bdb_3d_content[i][0][0]
        bdb2D=detection_2d[i]['bbox']

        '''add padding to the images to make it square, need to adjust 2d bounding box'''
        bdb2D_pad=copy.deepcopy(bdb2D)
        bdb2D_pad[0]+=max(0,(height-width)/2)
        bdb2D_pad[1] += max(0, (width - height) / 2)
        bdb2D_pad[2] += max(0, (height - width) / 2)
        bdb2D_pad[3] += max(0, (width - height) / 2)
        bdb2D_pos.append(torch.tensor(bdb2D_pad))

        classname=detection_2d[i]['class']
        cls_code=torch.zeros([9])
        cls_code[category_label_mapping[classname]]=1
        cls_codes.append(cls_code)

        patch=image.crop((bdb2D[0],bdb2D[1],bdb2D[2],bdb2D[3]))
        patch=data_transforms_crop(patch).float()
        patches.append(patch)

        yaw_rot = bdb3d['basis']
        half_rot = R_from_yaw_pitch_roll(-np.pi / 2, 0, 0)
        yaw_rot=np.dot(np.linalg.inv(yaw_rot),half_rot) #rotate -90 degree, and invert the rotation
        bbox_size=bdb3d[1][0]*2
        bbox_sizes.append(torch.from_numpy(bbox_size))

        center_world = bdb3d['centroid'][0]
        center_cam = np.dot(center_world, np.linalg.inv(camR).T)
        center_cam = center_cam[[2, 1, 0]]
        center_cam[1] = -1 * center_cam[1]
        #print(center_cam)

        obj_cam_centers.append(torch.from_numpy(center_cam))
        #center[1] = -center[1]

        pred_rot_matrix = np.dot(R_from_yaw_pitch_roll(0, pred_pitch,
                                -pred_roll), yaw_rot) #compute rotation matrix of the detection results

        rot_matrix = pred_rot_matrix
        rot_matrices.append(torch.from_numpy(rot_matrix))
        #obj_img_center = np.dot(K,obj_cam_center)
        #project_center=np.array([obj_img_center[0]/obj_img_center[2],obj_img_center[1]/obj_img_center[2]])
        #centroid_depth=obj_img_center[2]

        '''generate uniform grids inside the 2d bbox, used for afterwards grid_sample function'''
        bdb_x = np.linspace(bdb2D_pad[0], bdb2D_pad[2], 64)
        bdb_y = np.linspace(bdb2D_pad[1], bdb2D_pad[3], 64)
        bdb_X, bdb_Y = np.meshgrid(bdb_x, bdb_y)

        bdb_X = (bdb_X - max_length / 2) / max_length * 2 #scale to -1 ~ 1
        bdb_Y = (bdb_Y - max_length / 2) / max_length * 2
        bdb_grid = np.concatenate([bdb_X[:, :, np.newaxis], bdb_Y[:, :, np.newaxis]], axis=-1)
        bdb_grids.append(torch.from_numpy(bdb_grid))
    patches=torch.stack(patches)
    cls_codes=torch.stack(cls_codes)
    rot_matrices=torch.stack(rot_matrices)
    obj_cam_centers=torch.stack(obj_cam_centers)
    bdb_grids=torch.stack(bdb_grids)
    bdb2D_pos=torch.stack(bdb2D_pos)
    bbox_sizes=torch.stack(bbox_sizes)

    '''add padding to the image, to make it square, as training on pix3d does'''
    org_intrinsic=copy.deepcopy(K)
    f=org_intrinsic[0,0]
    max_length=max(width,height)
    intrinsic=np.array([[f,0,max_length/2],
                        [0,f,max_length/2],
                        [0,0,1]])

    pad_img = np.zeros([max_length, max_length, 3])
    image=np.asarray(image)/255.0
    if width >= height:
        margin = int((width - height) / 2)
        # print(image.shape,pad_img.shape)
        pad_img[margin:margin + height, :] = image
    else:
        margin = int((height - width) / 2)
        pad_img[:, margin:margin + width] = image

    pad_img = data_transforms(pad_img).float()

    data_batch={
        "whole_image":pad_img.float()[None],
        "image":patches.float(),
        "cls_codes":cls_codes.float(),
        "patch":patches.float(),
        "K":torch.from_numpy(intrinsic).float()[None],
        "rot_matrix":rot_matrices.float(),
        "bbox_size":bbox_sizes.float(),
        "obj_cam_center":obj_cam_centers.float(),
        "bdb_grid":bdb_grids.float(),
        "bdb2D_pos":bdb2D_pos.float(),
    }
    return data_batch


data_folder="./real_demo/%s"%(args.taskid)
data_batch=prepare_data(data_folder)
# with open("./real_demo/2003/debug_data.pkl",'wb') as f:
#     p.dump(data_batch,f)

# for key in data_batch:
#     print(key,data_batch[key].shape)

device=torch.device("cuda:0")
#print(torch.cuda.is_available())
instPIFu_config_path="./configs/test_instPIFu.yaml"
instPIFu_config=CONFIG(instPIFu_config_path).config
instPIFu_config['debug']=False
instPIFu_model=InstPIFu(instPIFu_config).to(device)

instPIFu_ckpt_path="./checkpoints/instPIFu/model_best_pix3d.pth" # use pretrained weight trained on pix3d
instPIFu_ckpt=torch.load(instPIFu_ckpt_path)
instPIFu_net_weight=instPIFu_ckpt['net']

instPIFu_new_net_weight={}
for key in instPIFu_net_weight:
    if key.startswith("module."):
        k_ = key[7:]
        instPIFu_new_net_weight[k_] = instPIFu_net_weight[key]
instPIFu_model.load_state_dict(instPIFu_new_net_weight)
instPIFu_model.eval()

num_objs=data_batch['patch'].shape[0]
for obj_id in tqdm.tqdm(range(num_objs)):
    input_dict={}
    for key in data_batch:
        if key!="whole_image" and key!="K":
            input_dict[key]=data_batch[key][obj_id:obj_id+1].cuda().float()
        else:
            input_dict[key]=data_batch[key].cuda().float()
    with torch.no_grad():
        mesh=instPIFu_model.extract_mesh(input_dict,64)

    rot_mat=data_batch['rot_matrix'][obj_id].cpu().numpy()
    obj_cam_center=data_batch['obj_cam_center'][obj_id].cpu().numpy()
    bbox_size=data_batch['bbox_size'][obj_id].cpu().numpy()

    obj_vert = np.asarray(mesh.vertices)
    obj_vert = obj_vert / 2 * bbox_size
    obj_vert = np.dot(obj_vert, rot_mat.T)
    obj_vert[:, 0:2] = -obj_vert[:, 0:2]
    obj_vert += obj_cam_center
    mesh.vertices = np.asarray(copy.deepcopy(obj_vert))

    mesh_save_path=os.path.join(data_folder,"%d.ply"%(obj_id))
    mesh.export(mesh_save_path)

