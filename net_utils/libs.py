# Lib functions in data processing and calculation.
# author: ynie
# date: Feb, 2020

import torch
import torch.nn as nn
import copy
import numpy as np
from torch.nn import functional as F
from copy import deepcopy
from net_utils.bins import *

def to_dict_tensor(dicts, if_cuda):
    '''
    Store dict to torch tensor.
    :param dicts:
    :param if_cuda:
    :return:
    '''
    dicts_new = copy.copy(dicts)
    for key, value in dicts_new.items():
        value_new = torch.from_numpy(np.array(value))
        if value_new.type() == 'torch.DoubleTensor':
            value_new = value_new.float()
        if if_cuda:
            value_new = value_new.cuda()
        dicts_new[key] = value_new
    return dicts_new

def num_from_bins(bins, cls, reg):
    """
    :param bins: b x 2 tensors
    :param cls: b long tensors
    :param reg: b tensors
    :return: bin_center: b tensors
    """
    bin_width = (bins[0][1] - bins[0][0])
    bin_center = (bins[cls, 0] + bins[cls, 1]) / 2
    return bin_center + reg * bin_width

def R_from_yaw_pitch_roll(yaw, pitch, roll):
    '''
    get rotation matrix from predicted camera yaw, pitch, roll angles.
    :param yaw: batch_size x 1 tensor
    :param pitch: batch_size x 1 tensor
    :param roll: batch_size x 1 tensor
    :return: camera rotation matrix
    '''
    n = yaw.size(0)
    Rp = torch.zeros((n, 3, 3)).cuda()
    Ry = torch.zeros((n,3,3)).cuda()
    Rr = torch.zeros((n,3,3)).cuda()
    cp=torch.cos(pitch)
    sp=torch.sin(pitch)
    cy=torch.cos(yaw)
    sy=torch.sin(yaw)
    cr=torch.cos(roll)
    sr=torch.sin(roll)
    Rp[:,0,0]=1
    Rp[:,1,1]=cp
    Rp[:,1,2]=-sp
    Rp[:,2,1]=sp
    Rp[:,2,2]=cp

    Ry[:,0,0]=cy
    Ry[:,0,2]=sy
    Ry[:,1,1]=1
    Ry[:,2,0]=-sy
    Ry[:,2,2]=cy

    Rr[:,0,0]=cr
    Rr[:,0,1]=-sr
    Rr[:,1,0]=sr
    Rr[:,1,1]=cr
    Rr[:,2,2]=1

    R=torch.bmm(torch.bmm(Rr,Rp),Ry)

    return R

def get_rotation_matrix_gt(bins_tensor, pitch_cls_gt, pitch_reg_gt, roll_cls_gt, roll_reg_gt):
    '''
    get rotation matrix from predicted camera pitch, roll angles.
    '''
    pitch = num_from_bins(bins_tensor['pitch_bin'], pitch_cls_gt, pitch_reg_gt)
    roll = num_from_bins(bins_tensor['roll_bin'], roll_cls_gt, roll_reg_gt)
    r_ex = R_from_yaw_pitch_roll(torch.zeros_like(pitch), pitch, roll)
    return r_ex, pitch, roll

def get_mask_status(masks, split):
    obj_status_flag = []
    for batch_id, interval in enumerate(split):
        for obj_id in range(interval[1]-interval[0]):
            if masks[batch_id][obj_id]:
                obj_status_flag.append(1)
            else:
                obj_status_flag.append(0)
    return np.array(obj_status_flag)

def layout_basis_from_ori(ori):
    """
    :param ori: orientation angle
    :return: basis: 3x3 matrix
            the basis in 3D coordinates
    """
    n = ori.size(0)

    basis = torch.zeros((n, 3, 3)).cuda()

    basis[:, 0, 0] = torch.sin(ori)
    basis[:, 0, 2] = torch.cos(ori)
    basis[:, 1, 1] = 1
    basis[:, 2, 0] = -torch.cos(ori)
    basis[:, 2, 2] = torch.sin(ori)

    return basis

def get_corners_of_bb3d(coeffs,centroid,ori):
    patch_size=coeffs.shape[0]
    #c_x, c_y, c_z = centroid[:, 0], centroid[:, 1], centroid[:, 2]
    cy = torch.cos(ori)  # B,1
    sy = torch.sin(ori)  # B,1
    #c_x = c_x
    #c_y = cp * c_y - sp * c_z
    #c_z = sp * c_y + cp * c_z
    s_x, s_y, s_z = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]
    c1 = torch.stack([- s_x / 2, - s_y / 2, - s_z / 2], dim=1)
    c2 = torch.stack([- s_x / 2, - s_y / 2, s_z / 2], dim=1)
    c3 = torch.stack([- s_x / 2, s_y / 2, - s_z / 2], dim=1)
    c4 = torch.stack([- s_x / 2, s_y / 2, s_z / 2], dim=1)
    c5 = torch.stack([s_x / 2, - s_y / 2, - s_z / 2], dim=1)
    c6 = torch.stack([s_x / 2, - s_y / 2, s_z / 2], dim=1)
    c7 = torch.stack([s_x / 2, s_y / 2, - s_z / 2], dim=1)
    c8 = torch.stack([s_x / 2, s_y / 2, s_z / 2], dim=1)

    verts = torch.cat(
        [c1[:, None], c2[:, None], c3[:, None], c4[:, None], c5[:, None], c6[:, None], c7[:, None], c8[:, None]], dim=1)
    Ry = torch.zeros((patch_size, 3, 3)).cuda()
    Ry[:, 0, 0] = cy
    Ry[:, 0, 2] = sy
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -sy
    Ry[:, 2, 2] = cy
    verts=torch.einsum("ijk,iqk->iqj",Ry,verts)
    #print(verts.shape,centroid.shape)
    verts=verts+centroid.squeeze(2).unsqueeze(1)

    return verts

def get_layout_bdb_sunrgbd(bins_tensor, lo_ori_reg, lo_ori_cls, centroid_reg, coeffs_reg):
    """
    get the eight corners of 3D bounding box
    :param bins_tensor:
    :param lo_ori_reg: layout orientation regression results
    :param lo_ori_cls: layout orientation classification results
    :param centroid_reg: layout centroid regression results
    :param coeffs_reg: layout coefficients regression results
    :return: bdb: b x 8 x 3 tensor: the bounding box of layout in layout system.
    """

    ori_reg = torch.gather(lo_ori_reg, 1, lo_ori_cls.view(lo_ori_cls.size(0), 1).expand(lo_ori_cls.size(0), 1)).squeeze(1)
    ori = num_from_bins(bins_tensor['layout_ori_bin'], lo_ori_cls, ori_reg)

    basis = layout_basis_from_ori(ori)

    centroid_reg = centroid_reg + bins_tensor['layout_centroid_avg']

    coeffs_reg = (coeffs_reg + 1) * bins_tensor['layout_coeffs_avg']

    bdb = get_corners_of_bb3d(basis, coeffs_reg, centroid_reg)

    return bdb

def get_bdb_form_from_corners(corners, mask_status=None):
    if mask_status is not None:
        corners = corners[mask_status.nonzero()]
    vec_0 = (corners[:, 4, :] - corners[:, 0, :]) / 2.
    vec_1 = (corners[:, 2, :] - corners[:, 0, :]) / 2.
    vec_2 = (corners[:, 1, :] - corners[:, 0, :]) / 2.

    coeffs_0 = torch.norm(vec_0, dim=1)
    coeffs_1 = torch.norm(vec_1, dim=1)
    coeffs_2 = torch.norm(vec_2, dim=1)
    coeffs = torch.cat([coeffs_0.unsqueeze(-1), coeffs_1.unsqueeze(-1), coeffs_2.unsqueeze(-1)], -1)

    centroid = (corners[:, 0, :] + corners[:, 6, :]) / 2.

    basis_0 = torch.mm(torch.diag(1 / coeffs_0), vec_0)
    basis_1 = torch.mm(torch.diag(1 / coeffs_1), vec_1)
    basis_2 = torch.mm(torch.diag(1 / coeffs_2), vec_2)

    basis = torch.cat([basis_0.unsqueeze(1), basis_1.unsqueeze(1), basis_2.unsqueeze(1)], dim=1)

    return {'basis': basis, 'coeffs': coeffs, 'centroid': centroid}


def recover_points_to_world_sys(bdb3D, mesh_coordinates, ldif_center=None, ldif_coef=None):
    '''
    Get 3D point cloud from mesh with estimated position and orientation.
    :param bdb3D: 3D object bounding boxes with keys ['coeffs', 'basis', 'centroid'].
    :param mesh_coordinates: Number_of_objects x Number_of_points x 3.
    :return: points on world system
    '''
    if ldif_center is None or ldif_coef is None:
        mesh_coordinates_in_world_sys = []

        for obj_idx, mesh_coordinate in enumerate(mesh_coordinates):
            #mesh_coordinate = mesh_coordinate.transpose(-1, -2)
            mesh_center = (mesh_coordinate.max(dim=0)[0] + mesh_coordinate.min(dim=0)[0]) / 2.
            mesh_center = mesh_center.detach()
            mesh_coordinate = mesh_coordinate - mesh_center

            mesh_coef = (mesh_coordinate.max(dim=0)[0] - mesh_coordinate.min(dim=0)[0]) / 2.
            mesh_coef = mesh_coef.detach()
            mesh_coordinate = torch.mm(torch.mm(mesh_coordinate, torch.diag(1. / mesh_coef)),
                                       torch.diag(bdb3D['coeffs'][obj_idx]))

            # set orientation
            mesh_coordinate = torch.mm(mesh_coordinate, bdb3D['basis'][obj_idx])

            # move to center
            mesh_coordinate = mesh_coordinate + bdb3D['centroid'][obj_idx].view(1, 3)

            mesh_coordinates_in_world_sys.append(mesh_coordinate)
    else:
        mesh_coordinates = mesh_coordinates - ldif_center.unsqueeze(1)
        mesh_coordinates = torch.matmul(mesh_coordinates, torch.diag_embed(1. / ldif_coef))
        mesh_coordinates = torch.matmul(mesh_coordinates, torch.diag_embed(bdb3D['coeffs']))

        mesh_coordinates[:,1:2] = -mesh_coordinates[:,1:2]

        mesh_coordinates = torch.matmul(mesh_coordinates, bdb3D['basis'])
        mesh_coordinates_in_world_sys = mesh_coordinates + bdb3D['centroid'].unsqueeze(1)

    return mesh_coordinates_in_world_sys

def recover_points_to_obj_sys(bdb3D, obj_sample, ldif_center, ldif_coef):
    '''
    Get 3D point cloud from mesh with estimated position and orientation.
    :param bdb3D: 3D object bounding boxes with keys ['coeffs', 'basis', 'centroid'].
    :param obj_sample: Number_of_objects x Number_of_points x 3.
    :return: points on world system
    '''
    # move to center
    obj_sample_in_obj_sys = obj_sample - bdb3D['centroid'].unsqueeze(1)

    # set orientation
    obj_sample_in_obj_sys = torch.matmul(obj_sample_in_obj_sys, torch.inverse(bdb3D['basis']))

    # scale to unit bbox
    obj_sample_in_obj_sys = torch.matmul(obj_sample_in_obj_sys, torch.diag_embed(1. / bdb3D['coeffs']))

    # scale and centralize to match ldif
    sample = torch.matmul(obj_sample_in_obj_sys, torch.diag_embed(ldif_coef))
    sample[:,1:2]=-sample[:,1:2]
    sample = sample + ldif_center.unsqueeze(1)

    return sample

def get_rotation_matix_result(bins_tensor, pitch_cls_gt, pitch_reg_result, roll_cls_gt, roll_reg_result,
                              return_degrees=False):
    '''
    get rotation matrix from predicted camera pitch, roll angles.
    '''

    pitch_result = torch.gather(pitch_reg_result, 1,
                              pitch_cls_gt.long().view(pitch_cls_gt.size(0), 1).expand(pitch_cls_gt.size(0), 1)).squeeze(1)
    roll_result = torch.gather(roll_reg_result, 1,
                               roll_cls_gt.long().view(roll_cls_gt.size(0), 1).expand(roll_cls_gt.size(0), 1)).squeeze(1)
    pitch = num_from_bins(bins_tensor['pitch_bin'], pitch_cls_gt.long(), pitch_result)
    roll = num_from_bins(bins_tensor['roll_bin'], roll_cls_gt.long(), roll_result)
    cam_R = R_from_yaw_pitch_roll(torch.zeros_like(pitch), pitch, roll)
    cam_R_inv=R_from_yaw_pitch_roll(torch.zeros_like(pitch), -pitch, -roll)
    if return_degrees:
        return cam_R, pitch, roll
    else:
        return cam_R,cam_R_inv


def rgb_to_world(p, depth, K, cam_R, split):
    """
    Given pixel location and depth, camera parameters, to recover world coordinates.
    :param p: n x 2 tensor
    :param depth: b tensor
    :param k: b x 3 x 3 tensor
    :param cam_R: b x 3 x 3 tensor
    :param split: b x 2 split tensor.
    :return: p_world_right: n x 3 tensor in right hand coordinate
    """

    n = p.size(0)

    K_ex = torch.cat([K[index].expand(interval[1] - interval[0], -1, -1) for index, interval in enumerate(split)], 0)
    cam_R_ex = torch.cat([cam_R[index].expand(interval[1] - interval[0], -1, -1) for index, interval in enumerate(split)], 0)

    x_temp = (p[:, 0] - K_ex[:, 0, 2]) / K_ex[:, 0, 0]
    y_temp = (p[:, 1] - K_ex[:, 1, 2]) / K_ex[:, 1, 1]
    z_temp = 1
    ratio = depth / torch.sqrt(x_temp ** 2 + y_temp ** 2 + z_temp ** 2)
    x_cam = x_temp * ratio
    y_cam = y_temp * ratio
    z_cam = z_temp * ratio

    # transform to toward-up-right coordinate system
    x3 = x_cam
    y3 = y_cam
    z3 = z_cam

    p_cam = torch.stack((x3, y3, z3), 1).view(n, 3, 1) # n x 3
    p_world = torch.bmm(cam_R_ex, p_cam)
    return p_world

def basis_from_ori(ori):
    """
    :param ori: torch tensor
            the orientation angle
    :return: basis: 3x3 tensor
            the basis in 3D coordinates
    """
    n = ori.size(0)

    basis = torch.zeros((n, 3, 3)).cuda()

    basis[:, 0, 0] = torch.cos(ori)
    basis[:, 0, 2] = -torch.sin(ori)
    basis[:, 1, 1] = 1
    basis[:, 2, 0] = torch.sin(ori)
    basis[:, 2, 2] = torch.cos(ori)

    return basis

def get_bdb_3d_result(bins_tensor,ori_cls_result,ori_reg_result,centroid_cls_result,centroid_reg_result,
                      size_cls,size_reg_result,P_result,camera_K,camera_R,split):
    patch_size=ori_cls_result.shape[0]
    size_cls_gt=torch.argmax(size_cls,dim=1)
    coeffs=(size_reg_result+1)*bins_tensor['avg_size'][size_cls_gt,:]

    centroid_ind=torch.argmax(centroid_cls_result,dim=1)
    centroid_depth=torch.mean(bins_tensor['centroid_bin'][centroid_ind,:],dim=1)+\
                   centroid_reg_result[torch.arange(patch_size),centroid_ind.long()]*DEPTH_WIDTH
    centroid=rgb_to_world(P_result,centroid_depth,camera_K,camera_R,split)

    ori_ind=torch.argmax(ori_cls_result,dim=1)
    ori=torch.mean(bins_tensor["ori_bin"][ori_ind,:],dim=1)+\
        ori_reg_result[torch.arange(patch_size),ori_ind.long()]*ORI_BIN_WIDTH
    bdb3d=get_corners_of_bb3d(coeffs,centroid,ori)

    return bdb3d


def project_3d_points_to_2d(points3d, cam_R_ex, K_ex):
    """
    project 3d points to 2d
    :param points3d: n x 8 x 3 tensor; n equals to number of boxes.
    :param cam_R_ex: n x 3 x 3 tensor
    :param K_ex: n x 3 x 3 tensor
    :return:
    """
    n = points3d.size(0)

    points_cam_ori = torch.bmm(points3d, cam_R_ex)
    T_cam = torch.FloatTensor([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]]).expand(n, -1, -1).cuda()
    points_cam = torch.bmm(points_cam_ori, torch.transpose(T_cam, 1, 2))
    points_cam_positive = torch.transpose(
        torch.stack((points_cam[:, :, 0], points_cam[:, :, 1], F.threshold(points_cam[:, :, 2], 0.0001, 0.0001)), 2), 1,
        2)  # b x 3 x 8

    points_2d_ori = torch.transpose(torch.bmm(K_ex, points_cam_positive), 1, 2)  # b x 8 x 3
    points_2d = torch.stack(
        (points_2d_ori[:, :, 0] / points_2d_ori[:, :, 2], points_2d_ori[:, :, 1] / points_2d_ori[:, :, 2]),
        2)  # n x 8 x 2
    return points_2d


def get_bdb_2d_result(bdb3d, cam_R, K, split):
    """
    :param bins_tensor:
    :param bdb3d: n x 8 x 3 tensor: n equals to the number of objects in all batches.
    :param cam_R: b x 3 x 3 tensor: b - batch number
    :param K: b x 3 x 3 tensor: b - batch number
    :return:
    """
    n = bdb3d.size(0)
    # convert K to n x 3 x 3
    K_ex = torch.cat([K[index].expand(interval[1] - interval[0], -1, -1) for index, interval in enumerate(split)], 0)
    cam_R_ex = torch.cat(
        [cam_R[index].expand(interval[1] - interval[0], -1, -1) for index, interval in enumerate(split)], 0)

    points_2d = project_3d_points_to_2d(bdb3d, cam_R_ex, K_ex)  # n x 8 x 2

    x1 = torch.min(torch.max(torch.min(points_2d[:, :, 0], dim=1)[0], torch.zeros(n).cuda()),
                   2 * K_ex[:, 0, 2]) / (K_ex[:, 0, 2].float())
    y1 = torch.min(torch.max(torch.min(points_2d[:, :, 1], dim=1)[0], torch.zeros(n).cuda()),
                   2 * K_ex[:, 1, 2]) / (K_ex[:, 1, 2].float())

    x2 = torch.min(torch.max(torch.max(points_2d[:, :, 0], dim=1)[0], torch.zeros(n).cuda()),
                   2 * K_ex[:, 0, 2]) / (K_ex[:, 0, 2].float())
    y2 = torch.min(torch.max(torch.max(points_2d[:, :, 1], dim=1)[0], torch.zeros(n).cuda()),
                   2 * K_ex[:, 1, 2]) / (K_ex[:, 1, 2].float())

    return torch.stack((x1, y1, x2, y2), 1)

def physical_violation(bdb_layout, bdb_3d, split):
    '''
    compute the loss of physical violation
    :param bdb_layout: b x 8 x 3 tensor
    :param bdb_3d: n x 8 x 3 tensor
    :param split: b x 2 tensor
    :return:
    '''
    n = bdb_3d.size(0)
    layout_max = torch.max(bdb_layout, dim=1)[0]  # b x 3
    layout_min = torch.min(bdb_layout, dim=1)[0]  # b x 3

    layout_max_ex = torch.cat([layout_max[index].expand(interval[1] - interval[0], -1) for index, interval in enumerate(split)], 0) # n x 3
    layout_min_ex = torch.cat([layout_min[index].expand(interval[1] - interval[0], -1) for index, interval in enumerate(split)], 0) # n x 3

    bdb_max = torch.max(bdb_3d, dim=1)[0]  # n x 3
    bdb_min = torch.min(bdb_3d, dim=1)[0]  # n x 3

    violation = F.relu(bdb_max - layout_max_ex) + F.relu(layout_min_ex - bdb_min)  # n x 3

    return violation, torch.zeros(n, 3).cuda()

def get_bdb_evaluation(bins_tensor, ori_cls_gt, ori_reg_result, centroid_cls_gt, centroid_reg_result, size_cls_gt,
                       size_reg_result, P, K, cam_R, split, return_bdb=False):

    bdb, bdb_form = get_bdb_3d_result(bins_tensor, ori_cls_gt, ori_reg_result, centroid_cls_gt, centroid_reg_result,
                                    size_cls_gt, size_reg_result, P, K, cam_R, split)

    n = ori_cls_gt.size(0)
    basis = bdb_form['basis']
    coeffs = bdb_form['coeffs']
    centroid = bdb_form['centroid']
    class_id = torch.argmax(size_cls_gt, 1)
    bdb_output = [{'basis': basis[i, :, :].cpu().numpy(), 'coeffs': coeffs[i, :].cpu().numpy(),
                   'centroid': centroid[i, :].squeeze().cpu().numpy(), 'classid': class_id[i].cpu().numpy()} for i in
                  range(n)]
    if not return_bdb:
        return bdb_output
    else:
        return bdb_output, bdb

def get_corners_of_bb3d_no_index(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    coeffs = np.abs(coeffs)
    corners[0, :] = - basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners[1, :] = - basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = + basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = + basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]

    corners[4, :] = - basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners[5, :] = - basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[6, :] = + basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[7, :] = + basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners = corners + np.tile(centroid, (8, 1))
    return corners

def change_key(bbox):
    if 'u1' not in bbox.keys() and 'x1' in bbox.keys():
        bbox = deepcopy(bbox)
        bbox['u1'] = bbox['x1']
        bbox['v1'] = bbox['y1']
        bbox['u2'] = bbox['x2']
        bbox['v2'] = bbox['y2']
        bbox.pop('x1', None)
        bbox.pop('x2', None)
        bbox.pop('y1', None)
        bbox.pop('y2', None)
    return bbox

def get_iou(bb1, bb2):
    """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'u1', 'v1', 'u2', 'v2'}
            The (u1, v1) position is at the top left corner,
            The (u2, v2) position is at the bottom right corner
        bb2 : dict
            Keys: {'u1', 'v1', 'u2', 'v2'}
            The (u1, v1) position is at the top left corner,
            The (u2, v2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
    """
    bb1 = change_key(bb1)
    bb2 = change_key(bb2)

    assert bb1['u1'] <= bb1['u2']
    assert bb1['v1'] <= bb1['v2']
    assert bb2['u1'] <= bb2['u2']
    assert bb2['v1'] <= bb2['v2']

    # determine the coordinates of the intersection rectangle
    u_left = max(bb1['u1'], bb2['u1'])
    v_top = max(bb1['v1'], bb2['v1'])
    u_right = min(bb1['u2'], bb2['u2'])
    v_bottom = min(bb1['v2'], bb2['v2'])

    if u_right < u_left or v_bottom < v_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (u_right - u_left) * (v_bottom - v_top)

    # compute the area of both AABBs
    bb1_area = (bb1['u2'] - bb1['u1']) * (bb1['v2'] - bb1['v1'])
    bb2_area = (bb2['u2'] - bb2['u1']) * (bb2['v2'] - bb2['v1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou