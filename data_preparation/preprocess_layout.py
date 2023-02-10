import numpy as np
import os
import json
import cv2
import trimesh
import pickle as p
import argparse
from tqdm import tqdm

def q2rot(Q):
    # Extract the values from Q
    x = Q[0]
    y = Q[1]
    z = Q[2]
    w = Q[3]
    x2=x+x
    y2=y+y
    z2=z+z
    xx=x*x2
    xy=x*y2
    xz=x*z2
    yy=y*y2
    yz=y*z2
    zz=z*z2
    wx=w*x2
    wy=w*y2
    wz=w*z2
    # First row of the rotation matrix
    r00 = (1-(yy+zz))
    r01 = xy+wz
    r02 = xz-wy
    # Second row of the rotation matrix
    r10 = xy-wz
    r11 = (1-(xx+zz))
    r12 = yz+wx
    # Third row of the rotation matrix
    r20 = xz+wy
    r21 = yz-wx
    r22 = 1-(xx+yy)

    # 3x3 rotation matrix
    pos = np.array([[r00, r10, r20],
                    [r01, r11, r21],
                    [r02, r12, r22],
                    ])
    return pos

def reconstruct_pcd(depth,intrinsic,cam2wrd_rot):
    valid_Y, valid_X = np.where((depth > 0) & (depth<10))
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
    #point_cloud_xyz[:,1] = -point_cloud_xyz[:,1].copy()
    point_cloud_incam=point_cloud_xyz[0:,0:3].copy()
    center=np.mean(point_cloud_incam,axis=0)
    point_cloud_xyz = np.dot(point_cloud_xyz,cam2wrd_rot.T)
    point_cloud=point_cloud_xyz
    #point_cloud=-point_cloud_xyz[:,0:3]
    return point_cloud,point_cloud_incam

def get_pitch_from_R(cam_R):
    angle=np.arctan2(cam_R[1,2],cam_R[1,1])
    roll=np.arctan2(-cam_R[1,0],cam_R[0,0])
    return angle

def get_rot_from_pitch(pitch):
    cp=np.cos(pitch)
    sp=np.sin(pitch)
    rot=np.array([[1,0,0],
              [0,cp,-sp],
              [0,sp,cp]])
    return rot


def run(data_root,save_root):
    if os.path.exists(save_root) == False:
        os.makedirs(save_root)
    folder_list = os.listdir(data_root)
    for folder in tqdm(folder_list):
        #print(folder)
        json_path=os.path.join(data_root,folder,"desc.json")
        depth_path=os.path.join(data_root,folder,"depth.png")
        try:
            depth=cv2.imread(depth_path,cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
            depth=(1-depth/255.0)*10
        except:
            continue
        with open(json_path,'rb') as f:
            content=json.load(f)
        bbox_infos=content["bbox_infos"]
        camera_Q = bbox_infos["camera"]["rot"]
        camera_K=np.abs(np.asarray(bbox_infos["camera"]["K"]))
        wrd2cam_rot = q2rot(camera_Q)
        cam2wrd_rot = np.linalg.inv(wrd2cam_rot)
        pitch=get_pitch_from_R(cam2wrd_rot)
        pitch_rot=get_rot_from_pitch(pitch)
        point_cloud,point_cloud_incam=reconstruct_pcd(depth,camera_K,pitch_rot)
        '''generate axis aligned bbox for layout'''
        x_max=np.max(point_cloud[:,0])
        x_min=np.min(point_cloud[:,0])
        y_max=np.max(point_cloud[:,1])
        y_min=np.min(point_cloud[:,1])
        z_max=np.max(point_cloud[:,2])
        z_min=np.min(point_cloud[:,2])
        size=np.array([x_max-x_min,y_max-y_min,z_max-z_min])
        centroid=np.array([(x_max+x_min)/2,(y_max+y_min)/2,(z_max+z_min)/2])
        proj_centroid=np.dot(np.linalg.inv(pitch_rot),centroid)
        layout={
            "proj_centroid":proj_centroid,
            "size":size,
            "pitch":pitch, #TODO:or negative pitch?
            "roll":0
        }
        save_path=os.path.join(save_root,folder+".pkl")
        with open(save_path,'wb') as f:
            p.dump(layout,f)


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('sample occupancy of 3D-FUTURE dataset')
    parser.add_argument('--data_root', type=str,
                        help='root path of 3dfront 2d data (depth should be contained)')
    parser.add_argument('--save_root', type=str, default='train', help='root path where to save the layout')
    return parser.parse_args()

if __name__=="__main__":
    args=parse_args()
    data_root=args.data_root
    save_root=args.save_root
    run(data_root,save_root)