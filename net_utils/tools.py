"""
Created on April, 2019

@author: Yinyu Nie

Toolkit functions used for processing training data.

Cite:
Huang, Siyuan, et al. "Cooperative Holistic Scene Understanding: Unifying 3D Object, Layout,
and Camera Pose Estimation." Advances in Neural Information Processing Systems. 2018.

"""

import numpy as np
from scipy.spatial import ConvexHull
import re
import cv2
import pickle
import json
from copy import deepcopy
from net_utils.bins import *
import torch

def bin_cls_reg(bins, loc):
    '''
    Given bins and value, compute where the value locates and the distance to the center.

    :param bins: list
    The bins, eg. [[-x, 0], [0, x]]
    :param loc: float
    The location
    :return cls: int, bin index.
    indicates which bin is the location for classification.
    :return reg: float, [-0.5, 0.5].
    the distance to the center of the corresponding bin.
    '''
    width_bin = bins[0][1] - bins[0][0]
    # get the distance to the center from each bin.
    dist = ([float(abs(loc - float(bn[0] + bn[1]) / 2)) for bn in bins])
    cls = dist.index(min(dist))
    reg = float(loc - float(bins[cls][0] + bins[cls][1]) / 2) / float(width_bin)
    return cls, reg

def sample_pnts_from_obj(data, n_pnts = 5000, mode = 'uniform'):
    # sample points on each object mesh.

    flags = data.keys()

    all_pnts = data['v'][:,:3]

    area_list = np.array(calculate_face_area(data))
    distribution = area_list/np.sum(area_list)

    # sample points the probability depends on the face area
    new_pnts = []
    if mode == 'random':

        random_face_ids = np.random.choice(len(data['f']), n_pnts, replace=True, p=distribution)
        random_face_ids, sample_counts = np.unique(random_face_ids, return_counts=True)

        for face_id, sample_count in zip(random_face_ids, sample_counts):

            face = data['f'][face_id]

            vid_in_face = [int(item.split('/')[0]) for item in face]

            weights = np.diff(np.sort(np.vstack(
                [np.zeros((1, sample_count)), np.random.uniform(0, 1, size=(len(vid_in_face) - 1, sample_count)),
                 np.ones((1, sample_count))]), axis=0), axis=0)

            new_pnt = all_pnts[np.array(vid_in_face) - 1].T.dot(weights)

            if 'vn' in flags:
                nid_in_face = [int(item.split('/')[2]) for item in face]
                new_normal = data['vn'][np.array(nid_in_face)-1].T.dot(weights)
                new_pnt = np.hstack([new_pnt, new_normal])


            new_pnts.append(new_pnt.T)

        random_pnts = np.vstack(new_pnts)

    else:

        for face_idx, face in enumerate(data['f']):
            vid_in_face = [int(item.split('/')[0]) for item in face]

            n_pnts_on_face = distribution[face_idx] * n_pnts

            if n_pnts_on_face < 1:
                continue

            dim = len(vid_in_face)
            npnts_dim = (np.math.factorial(dim - 1)*n_pnts_on_face)**(1/(dim-1))
            npnts_dim = int(npnts_dim)

            weights = np.stack(np.meshgrid(*[np.linspace(0, 1, npnts_dim) for _ in range(dim - 1)]), 0)
            weights = weights.reshape(dim - 1, -1)
            last_column = 1 - weights.sum(0)
            weights = np.vstack([weights, last_column])
            weights = weights[:, last_column >= 0]

            new_pnt = (all_pnts[np.array(vid_in_face) - 1].T.dot(weights)).T

            if 'vn' in flags:
                nid_in_face = [int(item.split('/')[2]) for item in face]
                new_normal = data['vn'][np.array(nid_in_face) - 1].T.dot(weights)
                new_pnt = np.hstack([new_pnt, new_normal])

            new_pnts.append(new_pnt)

        random_pnts = np.vstack(new_pnts)

    return random_pnts

def normalize_to_unit_square(points):
    centre = (points.max(0) + points.min(0))/2.
    point_shapenet = points - centre

    scale = point_shapenet.max()
    point_shapenet = point_shapenet / scale

    return point_shapenet, centre, scale

def read_obj(model_path, flags = ('v')):
    fid = open(model_path, 'r')

    data = {}

    for head in flags:
        data[head] = []

    for line in fid:
        # line = line.strip().split(' ')
        line = re.split('\s+', line.strip())
        if line[0] in flags:
            data[line[0]].append(line[1:])

    fid.close()

    if 'v' in data.keys():
        data['v'] = np.array(data['v']).astype(np.float)

    if 'vt' in data.keys():
        data['vt'] = np.array(data['vt']).astype(np.float)

    if 'vn' in data.keys():
        data['vn'] = np.array(data['vn']).astype(np.float)

    return data

def write_obj(objfile, data):

    with open(objfile, 'w+') as file:
        for item in data['v']:
            file.write('v' + ' %f' * len(item) % tuple(item) + '\n')

        for item in data['f']:
            file.write('f' + ' %s' * len(item) % tuple(item) + '\n')


def read_pkl(pkl_file):
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    return data

def read_json(json_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)
    return json_data

def write_split(sample_num, split_file, train_ratio = 0.8):
    train_ids = np.random.choice(sample_num, int(sample_num * train_ratio), replace=False)
    test_ids = np.setdiff1d(range(sample_num), train_ids)
    split_data = dict()
    split_data[u'train_ids'] = train_ids.tolist()
    split_data[u'test_ids'] = test_ids.tolist()

    with open(split_file, 'w') as file:
        json.dump(split_data, file)

def proj_pnt_to_img(cam_paras, point, faces, im_size, convex_hull = False):
    '''
    Project points from world system to image plane.
    :param cam_paras: a list of [camera origin (3-d), toward vec (3-d), up vec (3-d), fov_x, fov_y, quality_value]
    :param point: Nx3 points.
    :param faces: faces related to points.
    :param im_size: [width, height]
    :param convex_hull: Only use convex instead of rendering.
    :return: Mask image of object on the image.
    '''
    if point.shape[1] == 4:
        point = point[:,:3]

    ori_pnt = cam_paras[:3]
    toward = cam_paras[3:6] # x-axis
    toward /= np.linalg.norm(toward)
    up = cam_paras[6:9] # y-axis
    up /= np.linalg.norm(up)
    right = np.cross(toward, up) # z-axis
    right /= np.linalg.norm(right)
    width, height = im_size
    foc_w = width / (2. * np.tan(cam_paras[9]))
    foc_h = height/ (2. * np.tan(cam_paras[10]))
    K = np.array([[foc_w, 0., (width-1)/2.], [0, foc_h, (height-1)/2.], [0., 0., 1.]])

    R = np.vstack([toward, up, right]).T # columns respectively corresponds to toward, up, right vectors.
    p_cam = (point - ori_pnt).dot(R)

    # convert to traditional image coordinate system
    T_cam = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
    p_cam = p_cam.dot(T_cam.T)

    # delete those points whose depth value is non-positive.
    invalid_ids = np.where(p_cam[:,2]<=0)[0]

    p_cam[invalid_ids, 2] = 0.0001

    p_cam_h = p_cam/p_cam[:,2][:, None]
    pixels = K.dot(p_cam_h.T)

    pixels = pixels[:2, :].T.astype(np.int)

    new_image = np.zeros([height, width], np.uint8)
    if convex_hull:
        chull = ConvexHull(pixels)
        pixel_polygon =  pixels[chull.vertices, :]
        cv2.fillPoly(new_image, [pixel_polygon], 255)
    else:
        polys = [np.array([pixels[index-1] for index in face]) for face in faces]
        # cv2.fillPoly(new_image, polys, 255)
        for poly in polys:
            cv2.fillConvexPoly(new_image, poly, 255)

    return new_image.astype(np.bool)

def cvt2nyuclass_map(class_map, nyuclass_mapping):
    '''
    convert suncg classes in semantic map to nyu classes
    :param class_map: semantic segmentation map with suncg classes.
    :return nyu_class_map: semantic segmentation map with nyu40 classes.
    '''

    old_classes = np.unique(class_map)

    nyu_class_map = np.zeros_like(class_map)

    for class_id in old_classes:
        nyu_class_map[class_map == class_id] = nyuclass_mapping[nyuclass_mapping[:, 0] == class_id, 1]

    return nyu_class_map

def get_inst_classes(inst_map, cls_map):
    # get the class id for each instance
    instance_ids = np.unique(inst_map)
    class_ids = dict()
    for inst_id in instance_ids:
        classes, counts = np.unique(cls_map[inst_map==inst_id], return_counts=True)
        class_ids[inst_id] = classes[counts.argmax()]

    return class_ids

def yaw_pitch_roll_from_R(cam_R):
    '''
    get the yaw, pitch, roll angle from the camera rotation matrix.
    :param cam_R: Camera orientation. R:=[v1, v2, v3], the three column vectors respectively denote the toward, up,
    right vector relative to the world system.
    Hence, the R = R_y(yaw)*R_z(pitch)*R_x(roll).
    :return: yaw, pitch, roll angles.
    '''
    yaw = np.arctan(-cam_R[2][0]/cam_R[0][0])
    pitch = np.arctan(cam_R[1][0] / np.sqrt(cam_R[0][0] ** 2 + cam_R[2][0] ** 2))
    roll = np.arctan(-cam_R[1][2]/cam_R[1][1])

    return yaw, pitch, roll

def R_from_yaw_pitch_roll(yaw, pitch, roll):
    '''
    Retrieve the camera rotation from yaw, pitch, roll angles.
    Camera orientation. R:=[v1, v2, v3], the three column vectors respectively denote the toward, up,
    right vector relative to the world system.

    Hence, the R = R_y(yaw)*R_z(pitch)*R_x(roll).
    '''
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(yaw) * np.cos(pitch)
    R[0, 1] = np.sin(yaw) * np.sin(roll) - np.cos(yaw) * np.cos(roll) * np.sin(pitch)
    R[0, 2] = np.cos(roll) * np.sin(yaw) + np.cos(yaw) * np.sin(pitch) * np.sin(roll)
    R[1, 0] = np.sin(pitch)
    R[1, 1] = np.cos(pitch) * np.cos(roll)
    R[1, 2] = - np.cos(pitch) * np.sin(roll)
    R[2, 0] = - np.cos(pitch) * np.sin(yaw)
    R[2, 1] = np.cos(yaw) * np.sin(roll) + np.cos(roll) * np.sin(yaw) * np.sin(pitch)
    R[2, 2] = np.cos(yaw) * np.cos(roll) - np.sin(yaw) * np.sin(pitch) * np.sin(roll)
    return R

def normalize_point(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def get_world_R(cam_R):
    '''
    set a world system from camera matrix
    :param cam_R:
    :return:
    '''
    toward_vec = deepcopy(cam_R[:,0])
    toward_vec[1] = 0.
    toward_vec = normalize_point(toward_vec)
    up_vec = np.array([0., 1., 0.])
    right_vec = np.cross(toward_vec, up_vec)

    world_R = np.vstack([toward_vec, up_vec, right_vec]).T
    # yaw, _, _ = yaw_pitch_roll_from_R(cam_R)
    # world_R = R_from_yaw_pitch_roll(yaw, 0., 0.)

    return world_R

def bin_cls_reg(bins, loc):
    '''
    Given bins and value, compute where the value locates and the distance to the center.

    :param bins: list
    The bins, eg. [[-x, 0], [0, x]]
    :param loc: float
    The location
    :return cls: int, bin index.
    indicates which bin is the location for classification.
    :return reg: float, [-0.5, 0.5].
    the distance to the center of the corresponding bin.
    '''
    width_bin = bins[0][1] - bins[0][0]
    # get the distance to the center from each bin.
    dist = ([float(abs(loc - float(bn[0] + bn[1]) / 2)) for bn in bins])
    cls = dist.index(min(dist))
    reg = float(loc - float(bins[cls][0] + bins[cls][1]) / 2) / float(width_bin)
    return cls, reg

def camera_cls_reg_sunrgbd(cam_R, bin):
    '''
    Generate ground truth data for camera parameters (classification with regression manner).
    (yaw, pitch, roll)
    :param cam_R: Camera orientation. R:=[v1, v2, v3], the three column vectors respectively denote the toward, up,
    right vector relative to the world system.
    :param bin: ranges for classification and regression.
    :return: class labels and regression targets.
    '''
    pitch_bin = bin['pitch_bin']
    roll_bin = bin['roll_bin']
    _, pitch, roll = yaw_pitch_roll_from_R(cam_R)

    pitch_cls, pitch_reg = bin_cls_reg(pitch_bin, pitch)
    roll_cls, roll_reg = bin_cls_reg(roll_bin, roll)

    # with open('/home/ynie1/Projects/im2volume/data/sunrgbd/preprocessed/pitch.txt','a') as file:
    #     file.write('%d, %f\n' % (sample_id, pitch))
    # with open('/home/ynie1/Projects/im2volume/data/sunrgbd/preprocessed/roll.txt','a') as file:
    #     file.write('%d, %f\n' % (sample_id, roll))

    return pitch_cls, pitch_reg, roll_cls, roll_reg

def camera_cls_reg(cam_R, bin):
    '''
    Generate ground truth data for camera parameters (classification with regression manner).
    (yaw, pitch, roll)
    :param cam_R: Camera orientation. R:=[v1, v2, v3], the three column vectors respectively denote the toward, up,
    right vector relative to the world system.
    :param bin: ranges for classification and regression.
    :return: class labels and regression targets.
    '''
    pitch_bin = bin['pitch_bin']
    roll_bin = bin['roll_bin']
    _, pitch, roll = yaw_pitch_roll_from_R(cam_R)

    # remove eps for zeros. SUNCG cameras do not have roll angles.
    roll = 0. if abs(roll) < 0.001 else roll

    pitch_cls, pitch_reg = bin_cls_reg(pitch_bin, pitch)
    roll_cls, roll_reg = bin_cls_reg(roll_bin, roll)

    return pitch_cls, pitch_reg, roll_cls, roll_reg

def layout_centroid_depth_avg_residual(centroid_depth, avg_depth):
    """
    get the residual of the centroid depth of layout
    :param centroid depth: layout centroid depth
    :param avg depth: layout centroid average depth
    :return: regression value
    """
    reg = (centroid_depth - avg_depth) / avg_depth

    return reg

def layout_size_avg_residual(coeffs, avg):
    """
    get the residual of the centroid of layout
    :param coeffs: layout coeffs
    :param avg: layout centroid average
    :return: regression value
    """
    reg = (coeffs - avg) / avg
    return reg

def layout_basis_from_ori_sungrbd(ori):
    """
    :param ori: orientation angle
    :return: basis: 3x3 matrix
            the basis in 3D coordinates
    """
    basis = np.zeros((3,3))

    basis[0, 0] = np.sin(ori)
    basis[0, 2] = np.cos(ori)
    basis[1, 1] = 1
    basis[2, 0] = -np.cos(ori)
    basis[2, 2] = np.sin(ori)

    return basis

def ori_cls_reg(orientation, ori_bin):
    '''
    Generating the ground truth for object orientation

    :param orientation: numpy array
    orientation vector of the object.
    :param ori_bin: list
    The bins, eg. [[-x, 0], [0, x]]
    :return cls: int, bin index.
    indicates which bin is the location for classification.
    :return reg: float, [-0.5, 0.5].
    the distance to the center of the corresponding bin.
    '''
    # Note that z-axis (3rd dimension) points toward the frontal direction
    # The orientation angle is along the y-axis (up-toward axis)
    angle = np.arctan2(orientation[0], orientation[2])
    cls, reg = bin_cls_reg(ori_bin, angle)
    return cls, reg

def obj_size_avg_residual(coeffs, avg_size, class_id):
    """
    :param coeffs: object sizes
    :param size_template: dictionary that saves the mean size of each category
    :param class_id: nyu class id.
    :return: size residual ground truth normalized by the average size
    """
    size_residual = (coeffs - avg_size[class_id]) / avg_size[class_id]
    return size_residual

def list_of_dict_to_dict_of_list(dic):
    '''
    From a list of dict to a dict of list
    Each returned value is numpy array
    '''
    new_dic = {}
    keys = dic[0].keys()
    for key in keys:
        new_dic[key] = []
        for di in dic:
            new_dic[key].append(di[key])
        new_dic[key] = np.array(new_dic[key])
    return new_dic


# determinant of matrix a
def det(a):
    return a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] - a[0][2]*a[1][1]*a[2][0] - a[0][1]*a[1][0]*a[2][2] - a[0][0]*a[1][2]*a[2][1]

# unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    if magnitude == 0.:
        return (0., 0., 0.)
    else:
        return (x/magnitude, y/magnitude, z/magnitude)

#dot product of vectors a and b
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

#cross product of vectors a and b
def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return (x, y, z)

#area of polygon poly
def get_area(poly):
    if len(poly) < 3: # not a plane - no area
        return 0

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]

    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)

def calculate_face_area(data):

    face_areas = []

    for face in data['f']:
        vid_in_face = [int(item.split('/')[0]) for item in face]
        face_area = get_area(data['v'][np.array(vid_in_face) - 1,:3].tolist())
        face_areas.append(face_area)

    return face_areas

def write_logfile(text, log_file):
    # print and record loss
    print(text)
    with open(log_file, 'a') as f:  # open and append
        f.write(text + '\n')

def convert_result(data_batch,est_data):
    split=data_batch["split"]
    save_dict_list=[]
    for idx,interval in enumerate(split):
        sequence_id=data_batch['sequence_id'][idx]
        length=interval[1]-interval[0]
        bbox_list=[]
        gt_bbox_list=[]
        for i in range(length):
            # ------------------------------object detection result--------------------------------
            centroid_cls_result=est_data['centroid_cls_result'][interval[0]+i]
            centroid_reg_result=est_data['centroid_reg_result'][interval[0]+i]
            size_reg_result=est_data['size_reg_result'][interval[0]+i].cpu().numpy()
            size_cls=data_batch['size_cls'][interval[0]+i]
            ori_cls_result=est_data['ori_cls_result'][interval[0]+i]
            ori_reg_result=est_data['ori_reg_result'][interval[0]+i]
            offest2D_result=est_data['offset_2D_result'][interval[0]+i].cpu().numpy()
            bdb2D=data_batch['bdb2D'][interval[0]+i].cpu().numpy()
            #print(bdb2D)

            max_centroid_ind=torch.argmax(centroid_cls_result)
            centroid_reg=centroid_reg_result[max_centroid_ind].cpu().numpy()
            centroid=np.mean(bin['centroid_bin'][max_centroid_ind])+centroid_reg*DEPTH_WIDTH
            max_ori_ind=torch.argmax(ori_cls_result)
            ori_reg=ori_reg_result[max_ori_ind].cpu().numpy()
            ori=np.mean(bin['ori_bin'][max_ori_ind])+ori_reg*ORI_BIN_WIDTH
            #print(max_ori_ind, ori_reg,ori)
            project_center=np.zeros([2])
            project_center[0]=(bdb2D[2]+bdb2D[0])/2-offest2D_result[0]*(bdb2D[2]-bdb2D[0])
            project_center[1]=(bdb2D[1]+bdb2D[3])/2-offest2D_result[1]*(bdb2D[3]-bdb2D[1])
            size_class_ind=torch.where(size_cls>0)[0]
            size=torch.tensor(bin['avg_size'][size_class_ind])*(1+size_reg_result)
            bbox_dict={
                "project_center":project_center,
                "centroid_depth":centroid,
                "size":size.cpu().numpy(),
                "yaw":ori,
                "bdb2D":bdb2D,
            }
            bbox_list.append(bbox_dict)

            # TODO: add ground truth data
            #-----------------------------------object detection ground truth-----------------
            centroid_cls = data_batch['centroid_cls'][interval[0] + i].cpu().numpy().astype(np.int32)
            centroid_reg = data_batch['centroid_reg'][interval[0] + i]
            size_reg = data_batch['size_reg'][interval[0] + i].cpu().numpy()
            size_cls = data_batch['size_cls'][interval[0] + i]
            ori_cls = data_batch['ori_cls'][interval[0] + i].cpu().numpy().astype(np.int32)
            ori_reg = data_batch['ori_reg'][interval[0] + i]
            offest2D = data_batch['offset_2D'][interval[0] + i].cpu().numpy()
            bdb2D = data_batch['bdb2D'][interval[0] + i].cpu().numpy()

            gt_centroid_reg = centroid_reg.cpu().numpy()
            gt_centroid = np.mean(bin['centroid_bin'][centroid_cls]) + gt_centroid_reg * DEPTH_WIDTH
            #max_ori_ind = torch.argmax(ori_cls)
            gt_ori_reg = ori_reg.cpu().numpy()
            gt_ori = np.mean(bin['ori_bin'][ori_cls]) + gt_ori_reg * ORI_BIN_WIDTH
            #print(ori_cls,gt_ori_reg,gt_ori)
            project_center = np.zeros([2])
            project_center[0] = (bdb2D[2] + bdb2D[0]) / 2 - offest2D[0] * (bdb2D[2] - bdb2D[0])
            project_center[1] = (bdb2D[1] + bdb2D[3]) / 2 - offest2D[1] * (bdb2D[3] - bdb2D[1])
            size_class_ind = torch.where(size_cls > 0)[0]
            size = torch.tensor(bin['avg_size'][size_class_ind])*(1+ size_reg)
            gt_bbox_dict = {
                "project_center": project_center,
                "centroid_depth": gt_centroid,
                "size": size.cpu().numpy(),
                "yaw": gt_ori,
                "bdb2D": bdb2D,
            }
            gt_bbox_list.append(gt_bbox_dict)

        #-------------------------------layout estinamtion result----------------------------
        pitch_cls_result=est_data['pitch_cls_result'][idx]
        pitch_reg_result=est_data['pitch_reg_result'][idx]
        max_pitch_ind=torch.argmax(pitch_cls_result)
        pitch_reg=pitch_reg_result[max_pitch_ind].cpu().numpy()
        pitch=np.mean(bin['pitch_bin'][max_pitch_ind])+pitch_reg*PITCH_WIDTH

        roll_cls_result = est_data['roll_cls_result'][idx]
        roll_reg_result = est_data['roll_reg_result'][idx]
        max_roll_ind = torch.argmax(roll_cls_result)
        roll_reg = roll_reg_result[max_roll_ind].cpu().numpy()
        roll = np.mean(bin['roll_bin'][max_roll_ind]) + roll_reg * ROLL_WIDTH

        #------------------------------ground truth layout data-------------------------------
        gt_pitch=np.mean(bin['pitch_bin'][data_batch['pitch_cls'][idx].cpu().numpy()])+data_batch['pitch_reg'][idx].cpu().numpy()*PITCH_WIDTH
        gt_roll = np.mean(bin['roll_bin'][data_batch['roll_cls'][idx].cpu().numpy()]) + data_batch['roll_reg'][
            idx].cpu().numpy() * ROLL_WIDTH
        lo_centroid=avg_layout['avg_centroid']+est_data['lo_centroid_result'][idx].cpu().numpy()
        lo_size=avg_layout['avg_size']+est_data['lo_coeffs_result'][idx].cpu().numpy()
        layout_dict={
            "pitch":pitch,
            "roll":roll,
            "gt_pitch":gt_pitch,
            "gt_roll":gt_roll,
            "lo_centroid":lo_centroid,
            "lo_size":lo_size,
        }
        save_dict={
            "sequence_id":sequence_id,
            "layout":layout_dict,
            "bboxes":bbox_list,
            "gt_bboxes":gt_bbox_list
        }
        save_dict_list.append(save_dict)
    return save_dict_list

def total3d_todevice(cfg,data,device):
    image = data['image'].to(device)
    sequence_id = data["sequence_id"]
    pitch_reg = data['camera']['pitch_reg'].float().to(device)
    pitch_cls = data['camera']['pitch_cls'].long().to(device)
    roll_reg = data['camera']['roll_reg'].float().to(device)
    roll_cls = data['camera']['roll_cls'].long().to(device)
    # lo_ori_reg = data['layout']['ori_reg'].float().to(device)
    # lo_ori_cls = data['layout']['ori_cls'].long().to(device)
    lo_centroid = data['layout']['centroid_reg'].float().to(device)
    lo_coeffs = data['layout']['coeffs_reg'].float().to(device)
    lo_bdb3D = data['layout']['bdb3D'].float().to(device)

    patch = data['boxes_batch']['patch'].float().cuda()
    g_features = data['boxes_batch']['g_feature'].float().cuda()
    size_reg = data['boxes_batch']['size_reg'].float().cuda()
    size_cls = data['boxes_batch']['size_cls'].float().cuda()
    ori_reg = data['boxes_batch']['ori_reg'].float().cuda()
    ori_cls = data['boxes_batch']['ori_cls'].long().cuda()
    centroid_reg = data['boxes_batch']['centroid_reg'].float().cuda()
    centroid_cls = data['boxes_batch']['centroid_cls'].long().cuda()
    offset_2D = data['boxes_batch']['delta_2D'].float().cuda()
    box_feat = data["boxes_batch"]['box_feat'].float().cuda()
    bdb2D = data['boxes_batch']['bdb2D_pos'].float().cuda()
    bdb3D = data['boxes_batch']['bdb3D'].float().cuda()
    #yaw=data["boxes_batch"]["yaw"].float().cuda()
    #gt_yaw=data["boxes_batch"]["yaw"].float().cuda()
    split = data['obj_split']
    # split of relational pairs for batch learning.
    rel_pair_counts = torch.cat([torch.tensor([0]), torch.cumsum(
        torch.pow(data['obj_split'][:, 1] - data['obj_split'][:, 0], 2), 0)], 0)
    K = data['camera']["K"].float().cuda()
    camera_input = {
        "K": K
    }

    layout_input = {'image': image, 'pitch_reg': pitch_reg, 'pitch_cls': pitch_cls, 'roll_reg': roll_reg,
                    'roll_cls': roll_cls,
                    'lo_centroid': lo_centroid,
                    'lo_coeffs': lo_coeffs, 'lo_bdb3D': lo_bdb3D}

    object_input = {'patch': patch, 'g_features': g_features, 'size_reg': size_reg, 'size_cls': size_cls,
                    'ori_reg': ori_reg, 'ori_cls': ori_cls, 'centroid_reg': centroid_reg,
                    'centroid_cls': centroid_cls, "box_feat": box_feat, 'bdb3D': bdb3D,
                    'offset_2D': offset_2D, 'split': split, 'rel_pair_counts': rel_pair_counts, 'bdb2D_pos': bdb2D,"bdb2D":bdb2D}
    return {**layout_input,**object_input,**camera_input,"sequence_id":sequence_id}



