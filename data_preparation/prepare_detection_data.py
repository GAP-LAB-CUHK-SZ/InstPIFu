import numpy as np
import os,sys
sys.path.append("..")
import json
import pickle as p
import cv2
from .tools import bin_cls_reg,obj_size_avg_residual,list_of_dict_to_dict_of_list
from .tools import rotation_matrix,quaternion_rotation_matrix,Q2rot,get_bbox_corners,project_points2img,camera_cls_reg,get_layout_corner,bbox_corner_from_pred
from net_utils.bins import *
import multiprocessing as mp
import math
import argparse
category_label_mapping = {"table":0,
                          "sofa": 1,
                          "cabinet": 2,
                          "night_stand":3,
                          "chair":4,
                          "bookshelf":5,
                          "bed":6,
                          "desk":7,
                          "dresser":8
                          }
mean_size_path='../data/3dfront/3dfuture-mean_size.pkl'
with open(mean_size_path,'rb') as f:
    content=p.load(f)
    avg_size={}
    #print(content)
    for key in content:
        label_id=category_label_mapping[key]
        avg_size[label_id]=content[key]
#print(avg_size)
bin['avg_size'] = np.vstack([avg_size[key] for key in range(len(avg_size))])
#print(bin['avg_size'])
mean_layout_path="../data/3dfront/avg_layout.pkl"
with open(mean_layout_path,'rb') as f:
    avg_layout=p.load(f)

def save_gt_sample(data_folder,save_path):
    id=data_folder.split("/")[-1]
    print("processing %s"%(id))
    json_path=os.path.join(data_folder,"desc.json")
    with open(json_path,'r') as f:
        json_content=json.load(f)
    scene_id=json_content["scene_id"]

    scene_json_path=os.path.join(scene_json_dir,scene_id+".json")
    with open(scene_json_path,'r') as f:
        scene_json=json.load(f)
    model_uid=[]
    model_jid=[]
    model_bbox=[]
    for ff in scene_json["furniture"]:
        if ("valid" in ff and ff["valid"]):
            model_uid.append(ff['uid'])
            model_jid.append(ff['jid'])
            model_bbox.append(ff['bbox'])
    scene=scene_json["scene"]
    room=scene["room"]
    ref_list=[]
    idx_list=[]
    instance_id_list=[]
    R_list=[]
    pos_list=[]
    scale_list=[]
    for r in room:
        children = r["children"]
        for c in children:
            ref=c["ref"]
            instance_id=c["instanceid"]
            try:
                idx = model_uid.index(ref)
                pos=c['pos']
                rot=c['rot']
                scale=c['scale']
                ref_vec = [0, 0, 1]
                axis = np.cross(ref_vec, rot[1:])
                theta = np.arccos(np.dot(ref_vec, rot[1:])) * 2
                #print(axis)
                #print(theta)
                if np.sum(axis) != 0 and not math.isnan(theta):
                    R = rotation_matrix(axis, theta)
                    R_list.append(R)
                else:
                    R=np.array([[1,0,0],
                                [0,1,0],
                                [0,0,1]])
                    R_list.append(R)
                pos_list.append(pos)
                scale_list.append(scale)
                ref_list.append(ref)
                idx_list.append(idx)
                instance_id_list.append(instance_id)

            except:
                continue
    #print(instance_id_list)
    bbox_infos=json_content["bbox_infos"]
    object_infos=bbox_infos["object_infos"]
    camera_K=np.abs(np.array(bbox_infos["camera"]["K"]))
    camera_K[0]=camera_K[0]
    camera_K[1]=camera_K[1]
    camera_Q=bbox_infos["camera"]["rot"]
    camera_trans=bbox_infos["camera"]["pos"]
    #print(camera_trans)
    wrd2cam_matrix=quaternion_rotation_matrix(camera_Q,camera_trans)
    wrd2cam_matrix2=Q2rot(camera_Q,camera_trans)
    #yaw,pitch,roll=yaw_pitch_roll_from_R(wrd2cam_matrix[0:3,0:3])
    camera = {"K": camera_K,
              "scale_factor":2,
              "wrd2cam_matrix":wrd2cam_matrix,
              'wrd2cam_matrix2':wrd2cam_matrix2}
    layout_path=os.path.join(layour_dir,id+".pkl")
    with open(layout_path,'rb') as f:
        layout_content=p.load(f)
    camera['pitch_cls'],camera['pitch_reg'],camera['roll_cls'],camera['roll_reg']=camera_cls_reg(layout_content['pitch'],layout_content['roll'])
    layout={}
    l_centroid=layout_content['proj_centroid']
    l_size=layout_content["size"]
    layout['centroid_reg']=l_centroid-avg_layout['avg_centroid']
    layout['coeffs_reg']=l_size-avg_layout['avg_size']
    layout['bdb3D']=get_layout_corner(l_centroid,l_size,layout_content["pitch"])
    layout['pitch']=layout_content['pitch']
    layout['roll']=0
    height,width=968,1296 #same image height and width with scannet dataset
    bboxes_out=[]
    if len(object_infos)==0:
        print(data_folder,"do not have objects")
        return
    for object_info in object_infos:
        box_set={}
        instance_id=object_info["id"]
        try:
            search_id=instance_id_list.index(instance_id)
        except:
            return
        pos=pos_list[search_id]
        scale=scale_list[search_id]
        corr_id=idx_list[search_id]
        jid=model_jid[corr_id]
        box_set["jid"]=jid
        box_set["pos"]=pos
        box_set["scale"]=scale

        class_label=object_info["label"]
        class_id=category_label_mapping[class_label]
        if jid not in model_category_dict:
            model_category_dict[jid]=class_id
        #print(model_category_dict)
        box_set["size_cls"]=class_id
        bbox=object_info["bbox"]
        bbox_center=np.array(bbox["center"])
        #print(bbox_center)
        bbox_size=bbox["size"]
        if (bbox_size[0]==0) or (bbox_size[1]==0) or (bbox_size[2]==0):
            continue
        Q=object_info['6dpose']["rot"]
        obj_matrix=quaternion_rotation_matrix(Q,bbox_center)
        box_set["tran_matrix"]=obj_matrix
        box_set["R"]=R_list[search_id]
        bbox_corner=get_bbox_corners(bbox_center,bbox_size,obj_matrix,wrd2cam_matrix,layout_content['pitch'])
        bbox_center_cam=np.mean(bbox_corner,axis=0)

        rot_matrix=np.dot(wrd2cam_matrix[0:3,0:3],obj_matrix[0:3,0:3])
        vec_rot=np.dot(np.array([1,0,0]),rot_matrix.T)
        angle=np.arctan2(vec_rot[2],vec_rot[0])
        ori_cls, ori_reg = bin_cls_reg(bin['ori_bin'], angle)
        box_set['yaw']=angle
        box_set["ori_cls"],box_set["ori_reg"]=ori_cls,ori_reg
        #box_set["bdb3D"]=bbox_corner
        project_centroid=project_points2img(bbox_center_cam[np.newaxis],camera_K[0:3,0:3])[0]
        if project_centroid[0]<0 or project_centroid[0]>1296 or project_centroid[1]<0 or project_centroid[1]>968:
            continue
        bdb2D_from_3D=project_points2img(bbox_corner,camera_K[0:3,0:3])
        bdb2D_from_3D={
            "x1":max(bdb2D_from_3D[:,0].min(),0),
            "y1":max(bdb2D_from_3D[:,1].min(),0),
            "x2":min(bdb2D_from_3D[:,0].max(),width-1),
            "y2":min(bdb2D_from_3D[:,1].max(),height-1),
        }
        if ((bdb2D_from_3D['x2']-bdb2D_from_3D['x1'])<10) or ((bdb2D_from_3D['y2']-bdb2D_from_3D['y1'])<10):
            continue
        box_set['bdb2D_pos']=[bdb2D_from_3D["x1"],bdb2D_from_3D["y1"],bdb2D_from_3D["x2"],bdb2D_from_3D["y2"]]
        #if bbox_center_cam[2]>5: #filter far away object
        #    continue
        box_set['centroid_cls'], box_set['centroid_reg'] = bin_cls_reg(bin['centroid_bin'],
                                                                       np.linalg.norm(bbox_center_cam))
        box_set['size_reg']=obj_size_avg_residual(bbox_size, avg_size,class_id)
        yaw=np.mean(bin['ori_bin'][ori_cls])+ori_reg*ORI_BIN_WIDTH
        aa_bbox_corner=bbox_corner_from_pred(yaw, layout_content["pitch"], bbox_center_cam, bbox_size)
        box_set["bdb3D"] = aa_bbox_corner
        box_set["cam_center"]=bbox_center_cam
        delta_2D=[]
        delta_2D.append(((bdb2D_from_3D['x1']+bdb2D_from_3D['x2'])/2.-project_centroid[0])/(bdb2D_from_3D['x2']-bdb2D_from_3D['x1']))

        delta_2D.append(((bdb2D_from_3D['y1'] + bdb2D_from_3D['y2']) / 2. - project_centroid[1]) / (
                    bdb2D_from_3D['y2'] - bdb2D_from_3D['y1']))
        #print(delta_2D)
        box_set["delta_2D"]=delta_2D
        bboxes_out.append(box_set)
    if len(bboxes_out)<1:
        print(data_folder, "do not have objects")
        return
    data_dict={
        'layout':layout,
        "camera":camera,
        "boxes":list_of_dict_to_dict_of_list(bboxes_out),
        "sequence_id":id,
        "scene_id":scene_id
    }
    with open(save_path,'wb') as f:
        p.dump(data_dict,f)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('sample occupancy of 3D-FUTURE dataset')
    parser.add_argument('--data_root', type=str,
                        help='root path of 3dfront 2d data (without image and depth, only camera infomation and objects information)')
    parser.add_argument('--save_root', type=str, default='train', help='root path where to save the occupancy')
    parser.add_argument('--FRONT3D_root',type=str,help="root path of 3dfront data")
    parser.add_argument('--layout_root',type=str,help="root path of layout data")
    return parser.parse_args()

if __name__=="__main__":
    args=parse_args()
    data_dir=args.data_root
    save_dir=args.save_root
    layour_dir=args.layout_root
    scene_json_dir=args.FRONT3D_root

    model_category_dict={}
    if os.path.exists(save_dir)==False:
        os.makedirs(save_dir)
    folder_list=os.listdir(data_dir)
    folder_list.sort(key=lambda x:int(x[10:]))
    pool = mp.Pool(10)
    for folder in folder_list[:]:
        folder_path=os.path.join(data_dir,folder)
        save_path=os.path.join(save_dir,folder+".pkl")
        pool.apply_async(save_gt_sample, (folder_path, save_path,))
    pool.close()
    pool.join()




