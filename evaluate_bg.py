import numpy as np
import os
import glob
import cv2
import torch
import trimesh
from external.pyTorchChamferDistance.chamfer_distance import ChamferDistance
dist_chamfer=ChamferDistance()
import argparse

def reconstruct_pcd(depth,mask,intrinsic):
    valid_Y, valid_X = np.where(mask>0)
    random_ind=np.random.choice(valid_Y.shape[0],10000)
    valid_Y=valid_Y[random_ind]
    valid_X=valid_X[random_ind]
    unprojected_Y = valid_Y * depth[valid_Y, valid_X]
    unprojected_X = valid_X * depth[valid_Y, valid_X]
    unprojected_Z = depth[valid_Y, valid_X]
    point_cloud_xyz = np.concatenate(
        [unprojected_X[:, np.newaxis], unprojected_Y[:, np.newaxis], unprojected_Z[:, np.newaxis]], axis=1)
    intrinsic_inv = np.linalg.inv(intrinsic[0:3,0:3])
    point_cloud_xyz = np.dot(intrinsic_inv, point_cloud_xyz.T).T
    point_cloud=point_cloud_xyz[:,0:3]
    return point_cloud

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('totalindoorrecon evaluation')
    parser.add_argument('--result_dir', type=str,
                        help='folder contains the results of object mesh')
    parser.add_argument('--gt_dir',type=str,default="./data/3dfront/bgdepth",help="folder containing the watertight ground truth mesh")
    return parser.parse_args()

args=parse_args()
cd_loss_list=[]
result_filelist=glob.glob(args.result_dir+"/*.ply")
for result_file in result_filelist:
    taskid = result_file.split("/")[-1].split(".")[0]
    ind = int(taskid[10:])
    if ind < 3000 or ind >= 9000:
        continue

    pred_mesh = trimesh.load(result_file)
    '''depth image is scaled into 268x200'''
    width = 268
    height = 200
    K = np.array([[1168, 0, 1296 / 2],
                  [0, 1168, 968 / 2],
                  [0, 0, 1]])
    '''the intrinsic need to scale as well, the original size is 1296x968'''
    K[0] = K[0] / 1296 * 268
    K[1] = K[1] / 968 * 200

    gt_path = os.path.join(args.gt_dir, taskid, "depth.png")
    gt_depth = cv2.imread(gt_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # print(gt_depth.shape)
    # gt_depth=cv2.resize(gt_depth,dsize=(width,height),interpolation=cv2.INTER_NEAREST)
    gt_depth = (1 - gt_depth / 255.0) * 10

    mask = ((gt_depth > 0) & (gt_depth < 10)).astype(np.float32)

    pred_pcd = pred_mesh.sample(10000)
    gt_pcd = reconstruct_pcd(gt_depth, mask, K)

    pred_sample_gpu = torch.from_numpy(pred_pcd).float().cuda().unsqueeze(0)
    gt_sample_gpu = torch.from_numpy(gt_pcd).float().cuda().unsqueeze(0)
    # print(pred_sample_gpu.shape,gt_sample_gpu.shape)
    # loss,_=chamfer_distance(x=pred_sample_gpu,y=gt_sample_gpu)
    dist1, dist2 = dist_chamfer(gt_sample_gpu, pred_sample_gpu)[:2]
    cd_loss = torch.mean(dist1) + torch.mean(dist2)
    cd_loss_list.append(cd_loss.item())
    print("processing %s, current cd loss is %f, current mean cd loss is %f" % (
    result_file, cd_loss.item(), np.mean(np.array(cd_loss_list))))

print("mean cd loss is %f" % (np.mean(np.array(cd_loss_list))))