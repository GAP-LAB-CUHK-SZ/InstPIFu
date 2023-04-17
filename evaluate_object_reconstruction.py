import os,sys
#.path.append("/")
import numpy as np
import trimesh
import argparse
import glob
import pickle as p
import torch
import json
import tempfile
import shutil
import subprocess
from net_utils.bins import *
from external.pyTorchChamferDistance.chamfer_distance import ChamferDistance
import scipy

dist_chamfer=ChamferDistance()

category_label_mapping = {0:"table",
                          1:"sofa",
                          2:"cabinet",
                          3:"night_stand",
                          4:"chair",
                          5:"bookshelf",
                          6:"bed",
                          7:"desk",
                          8:"dresser"
                          }

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('totalindoorrecon evaluation')
    parser.add_argument('--result_dir', type=str,
                        help='folder contains the results of object mesh')
    parser.add_argument('--gt_dir',type=str,default="/data3/haolin/data/3D-FUTURE-watertight/",help="folder containing the watertight ground truth mesh")
    return parser.parse_args()

def delete_disconnected_component(mesh):

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

OCCNET_FSCORE_EPS = 1e-09

def percent_below(dists, thresh):
  return np.mean((dists**2 <= thresh).astype(np.float32)) * 100.0

def f_score(a_to_b, b_to_a, thresh):
  precision = percent_below(a_to_b, thresh)
  recall = percent_below(b_to_a, thresh)

  return (2 * precision * recall) / (precision + recall + OCCNET_FSCORE_EPS)
def pointcloud_neighbor_distances_indices(source_points, target_points):
  target_kdtree = scipy.spatial.cKDTree(target_points)
  distances, indices = target_kdtree.query(source_points, n_jobs=-1)
  return distances, indices
def fscore(points1,points2,tau=0.002):
  """Computes the F-Score at tau between two meshes."""
  dist12, _ = pointcloud_neighbor_distances_indices(points1, points2)
  dist21, _ = pointcloud_neighbor_distances_indices(points2, points1)
  f_score_tau = f_score(dist12, dist21, tau)
  return f_score_tau

args=parse_args()
result_list=glob.glob(args.result_dir+"/*.ply")
prepare_data_dir="./data/3dfront/prepare_data/test"
gt_dir=args.gt_dir
split_path="./data/3dfront/split/test/all.json"
with open(split_path,'r') as f:
    split=json.load(f)
select_split_list=[]
for idx,(taskid,object_id) in enumerate(split):
    if idx>2000:
        break
    select_split_list.append((taskid,object_id))
def get_rot_from_yaw(yaw):
    cy=np.cos(yaw)
    sy=np.sin(yaw)
    rot=np.array([[cy,0,sy],
                     [0,1,0],
                     [-sy,0,cy]])
    return rot

chamfer_distance_list=[]
cd_loss_dict={}
fscore_list=[]
fst_dict={}
#select_class_list=["bed"]
log_txt=os.path.join(args.result_dir,"evaluate_log.txt")
for (taskid,object_id) in select_split_list:
    #taskid,object_id,depth_error=item
    result_file=os.path.join(args.result_dir,"%s_%s.ply"%(taskid,object_id))
    if os.path.isfile(result_file)==False:
        continue
    #print(result_file)
    file_name=result_file.split("/")[-1].split(".")[0]
    taskid=file_name.split("_")[0]
    object_id=file_name.split("_")[1]
    #pred_mesh.vertices=np.dot(pred_mesh.vertices,yaw_rot.T)

    prepare_data_path=os.path.join(prepare_data_dir,taskid+".pkl")
    with open(prepare_data_path,'rb') as f:
        prepare_data=p.load(f)

    size_cls, size_reg = prepare_data['boxes']['size_cls'][int(object_id)], prepare_data['boxes']['size_reg'][
        int(object_id)]
    size = avg_size[size_cls] * (1 + size_reg)
    classname = category_label_mapping[size_cls]
    #if classname not in select_class_list:
    #    continue
    if classname not in cd_loss_dict:
        cd_loss_dict[classname]=[]
        fst_dict[classname]=[]

    jid=prepare_data['boxes']['jid'][int(object_id)]
    tran_matrix=prepare_data['boxes']['tran_matrix'][int(object_id)]
    cam_center=prepare_data['boxes']['cam_center'][int(object_id)]
    K=prepare_data['camera']['K']
    wrd2cam_matrix = prepare_data['camera']['wrd2cam_matrix']
    rot_matrix = np.dot(wrd2cam_matrix[0:3, 0:3], tran_matrix[0:3, 0:3])
    inv_rot=np.linalg.inv(rot_matrix)
    #print(prepare_data['boxes'].keys())
    try:
        pred_mesh = trimesh.load(result_file)

        gt_mesh_path=os.path.join(gt_dir,jid,"normalized_watertight.obj")
        gt_mesh=trimesh.load(gt_mesh_path)
    except:
        continue

    '''align two mesh firstly'''
    #pxmin,pxmax=np.min(pred_mesh.vertices[:,0]),np.max(pred_mesh.vertices[:,0])
    #pymin, pymax = np.min(pred_mesh.vertices[:, 1]), np.max(pred_mesh.vertices[:, 1])
    #pzmin, pzmax = np.min(pred_mesh.vertices[:, 2]), np.max(pred_mesh.vertices[:, 2])

    gxmin, gxmax = np.min(gt_mesh.vertices[:, 0]), np.max(gt_mesh.vertices[:, 0])
    gymin, gymax = np.min(gt_mesh.vertices[:, 1]), np.max(gt_mesh.vertices[:, 1])
    gzmin, gzmax = np.min(gt_mesh.vertices[:, 2]), np.max(gt_mesh.vertices[:, 2])

    #pred_mesh.vertices=pred_mesh.vertices-np.array([(pxmin+pxmax)/2,(pymin+pymax)/2,(pzmin+pzmax)/2])
    #pred_mesh.vertices=pred_mesh.vertices/np.array([pxmax-pxmin,pymax-pymin,pzmax-pymin])*2

    pred_mesh.vertices=pred_mesh.vertices/2*size/np.max(size)*2
    gt_mesh.vertices=gt_mesh.vertices/2*size/np.max(size)*2

    temp_folder = tempfile.mktemp(dir='/dev/shm')
    os.makedirs(temp_folder)
    shutil.copy('./external/ldif/gaps/bin/x86_64/mshalign', temp_folder)
    output_file = os.path.join(temp_folder, 'output.ply')
    pred_mesh.export(output_file)
    align_file = os.path.join(temp_folder, 'align.ply')
    gt_file = os.path.join(temp_folder, 'gt.ply')
    gt_mesh.export(gt_file)
    cmd = f"{os.path.join(temp_folder, 'mshalign')} {output_file} {gt_file} {align_file}"
    subprocess.check_output(cmd, shell=True)
    align_mesh = trimesh.load(align_file)

    #pred_mesh.export("/data1/haolin/alignmesh.ply")
    #gt_mesh.export("/data1/haolin/gt_mesh.ply")
    cmd="rm -r %s"%(temp_folder)
    os.system(cmd)

    pred_sample_points=align_mesh.sample(10000)
    gt_sample_points=gt_mesh.sample(10000)

    fst=fscore(pred_sample_points,gt_sample_points)
    fst_dict[classname].append(fst)
    fscore_list.append(fst)


    pred_sample_gpu=torch.from_numpy(pred_sample_points).float().cuda().unsqueeze(0)
    gt_sample_gpu=torch.from_numpy(gt_sample_points).float().cuda().unsqueeze(0)
    #print(pred_sample_gpu.shape,gt_sample_gpu.shape)
    #loss,_=chamfer_distance(x=pred_sample_gpu,y=gt_sample_gpu)
    dist1,dist2=dist_chamfer(gt_sample_gpu,pred_sample_gpu)[:2]
    cd_loss=torch.mean(dist1)+torch.mean(dist2)
    cd_loss_dict[classname].append(cd_loss.item())
    chamfer_distance_list.append(cd_loss.item())
    msg="processing %s ,class %s, cd loss: %f,mean cd_loss: %f, fscore: %f, mean fscore: %f" %(
        result_file,classname,cd_loss.item(),np.mean(np.array(chamfer_distance_list)),fst,np.mean(np.array(fscore_list)))
    print(msg)
    with open(log_txt,'a') as f:
        f.write(msg+"\n")

mean_chamfer_distance=np.mean(np.array(chamfer_distance_list))
msg="mean chamfer distance is %f"%(mean_chamfer_distance)
print(msg)
with open(log_txt, 'a') as f:
    f.write(msg + "\n")
for key in cd_loss_dict:
    cd_loss_dict[key]=np.mean(np.array(cd_loss_dict[key]))
for key in fst_dict:
    fst_dict[key]=np.mean(np.array(fst_dict[key]))
for key in cd_loss_dict:
    msg="cd loss of category %s is %f"%(key,cd_loss_dict[key])
    print(msg)
    with open(log_txt, 'a') as f:
        f.write(msg + "\n")
for key in fst_dict:
    msg="fscore of category %s is %f"%(key,fst_dict[key])
    print(msg)
    with open(log_txt,'a') as f:
        f.write(msg+"\n")


