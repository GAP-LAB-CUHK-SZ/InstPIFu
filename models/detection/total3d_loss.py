import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.data_config import cls_reg_ratio
from net_utils.libs import get_layout_bdb_sunrgbd, get_bdb_form_from_corners, \
    recover_points_to_world_sys, get_rotation_matix_result, \
    get_bdb_2d_result, physical_violation, recover_points_to_obj_sys,R_from_yaw_pitch_roll
import numpy as np
from net_utils.bins import *


cls_criterion = nn.CrossEntropyLoss(reduction='mean')
reg_criterion = nn.SmoothL1Loss(reduction='mean')
mse_criterion = nn.MSELoss(reduction='mean')
binary_cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')

class BaseLoss(object):
    '''base loss class'''
    def __init__(self, weight=1, config=None):
        '''initialize loss module'''
        self.weight = weight
        self.config = config

def cls_reg_loss(cls_result, cls_gt, reg_result, reg_gt):
    cls_loss = cls_criterion(cls_result, cls_gt)
    if len(reg_result.size()) == 3:
        reg_result = torch.gather(reg_result, 1, cls_gt.view(reg_gt.size(0), 1, 1).expand(reg_gt.size(0), 1, reg_gt.size(1)))
    else:
        reg_result = torch.gather(reg_result, 1, cls_gt.view(reg_gt.size(0), 1).expand(reg_gt.size(0), 1))
    reg_result = reg_result.squeeze(1)
    reg_loss = reg_criterion(reg_result, reg_gt)
    return cls_loss, cls_reg_ratio * reg_loss

class DetLoss(BaseLoss):
    def __call__(self, est_data, gt_data):
        # calculate loss
        size_reg_loss = reg_criterion(est_data['size_reg_result'], gt_data['size_reg']) * cls_reg_ratio
        ori_cls_loss, ori_reg_loss = cls_reg_loss(est_data['ori_cls_result'], gt_data['ori_cls'].long(), est_data['ori_reg_result'], gt_data['ori_reg'])
        centroid_cls_loss, centroid_reg_loss = cls_reg_loss(est_data['centroid_cls_result'], gt_data['centroid_cls'].long(),
                                                          est_data['centroid_reg_result'], gt_data['centroid_reg'])
        offset_2D_loss = reg_criterion(est_data['offset_2D_result'], gt_data['offset_2D'])

        return {'size_reg_loss':size_reg_loss, 'ori_cls_loss':ori_cls_loss, 'ori_reg_loss':ori_reg_loss,
                'centroid_cls_loss':centroid_cls_loss, 'centroid_reg_loss':centroid_reg_loss,
                'offset_2D_loss':offset_2D_loss}

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

def layout_corner_from_pred(pitch,bbox_centroid,size):
    batch_size=bbox_centroid.shape[0]
    c_x, c_y, c_z = bbox_centroid[:,0], bbox_centroid[:,1], bbox_centroid[:,2]
    cp = torch.cos(pitch)  # B,1
    sp = torch.sin(pitch)  # B,1
    c_x=c_x
    c_y=cp*c_y-sp*c_z
    c_z=sp*c_y+cp*c_z
    s_x, s_y, s_z = size[:,0], size[:,1], size[:,2]
    c1=torch.stack([c_x- s_x / 2, c_y- s_y / 2, c_z- s_z / 2],dim=1)
    c2=torch.stack([c_x- s_x / 2, c_y- s_y / 2, c_z+s_z / 2],dim=1)
    c3 = torch.stack([c_x- s_x / 2, c_y+s_y / 2, c_z- s_z / 2], dim=1)
    c4 = torch.stack([c_x- s_x / 2, c_y+s_y / 2, c_z+s_z / 2], dim=1)
    c5 = torch.stack([c_x+s_x / 2, c_y- s_y / 2, c_z- s_z / 2], dim=1)
    c6 = torch.stack([c_x+s_x / 2, c_y- s_y / 2, c_z+s_z / 2], dim=1)
    c7 = torch.stack([c_x+s_x / 2, c_y+s_y / 2, c_z- s_z / 2], dim=1)
    c8 = torch.stack([c_x+s_x / 2, c_y+s_y / 2, c_z+s_z / 2], dim=1)

    verts=torch.cat([c1[:,None],c2[:,None],c3[:,None],c4[:,None],c5[:,None],c6[:,None],c7[:,None],c8[:,None]],dim=1)

    return verts

def get_layout_bdb(bins_tensor,pitch_cls_result,pitch_reg_result,lo_centroid_reg,lo_coeffs_reg):
    pitch_cls_ind=torch.argmax(pitch_cls_result,axis=1)
    pitch_reg=pitch_reg_result[torch.arange(pitch_cls_result.shape[0]),pitch_cls_ind.long()]
    pitch=torch.mean(bins_tensor["pitch_bin"][pitch_cls_ind],dim=1)+pitch_reg*PITCH_WIDTH

    lo_centroid = torch.tensor(avg_layout['avg_centroid']).float().cuda() + lo_centroid_reg
    lo_coeffs=torch.tensor(avg_layout['avg_size']).float().cuda()+lo_coeffs_reg
    layout_bdb=layout_corner_from_pred(pitch,lo_centroid,lo_coeffs)
    return layout_bdb



class PoseLoss(BaseLoss):
    def __call__(self, est_data, gt_data):
        pitch_cls_loss, pitch_reg_loss = cls_reg_loss(est_data['pitch_cls_result'], gt_data['pitch_cls'].long(), est_data['pitch_reg_result'], gt_data['pitch_reg'])
        roll_cls_loss, roll_reg_loss = cls_reg_loss(est_data['roll_cls_result'], gt_data['roll_cls'].long(), est_data['roll_reg_result'], gt_data['roll_reg'])
        lo_centroid_loss = reg_criterion(est_data['lo_centroid_result'], gt_data['lo_centroid']) * cls_reg_ratio
        lo_coeffs_loss = reg_criterion(est_data['lo_coeffs_result'], gt_data['lo_coeffs']) * cls_reg_ratio

        # layout bounding box corner loss
        lo_bdb3D_result = get_layout_bdb(bins_tensor, est_data['pitch_cls_result'], est_data['pitch_reg_result'],
                                                 est_data['lo_centroid_result'],
                                                 est_data['lo_coeffs_result'])
        # layout bounding box corner loss
        #print(type(lo_bdb3D_result),type(gt_data['lo_bdb3D']))
        lo_corner_loss = cls_reg_ratio * reg_criterion(lo_bdb3D_result, gt_data['lo_bdb3D'])

        return {'pitch_cls_loss':pitch_cls_loss, 'pitch_reg_loss':pitch_reg_loss,
                'roll_cls_loss':roll_cls_loss, 'roll_reg_loss':roll_reg_loss,
                'lo_centroid_loss':lo_centroid_loss, 'lo_coeffs_loss':lo_coeffs_loss,"lo_corner_loss":lo_corner_loss},{"lo_bdb3D_result":lo_bdb3D_result}

class LDIFLoss(BaseLoss):
    def __call__(self, est_data, gt_data,config):
        # calculate loss (ldif.training.loss.compute_loss)
        #print(est_data['uniform_class'].shape,gt_data['uniform_class'].shape)
        nss_sample_loss = config['model']['near_surface_loss_weight']*\
                          nn.MSELoss()(est_data['pred_class'][:,0:2048].squeeze(2), gt_data['gt_class'][:,0:2048])

        uniform_sample_loss=config['model']['uniform_loss_weight']*\
                          nn.MSELoss()(est_data['pred_class'][:,2048:].squeeze(2), gt_data['gt_class'][:,2048:])

        element_centers = est_data['element_centers']

        bounding_box = config['data']['bounding_box']
        lower, upper = -bounding_box, bounding_box
        lower_error = torch.max(lower - element_centers, torch.zeros(1).cuda())
        upper_error = torch.max(element_centers - upper, torch.zeros(1).cuda())
        bounding_box_constraint_error = lower_error * lower_error + upper_error * upper_error
        bounding_box_error = torch.mean(bounding_box_constraint_error)
        inside_box_loss = config['model']['inside_box_loss_weight'] * bounding_box_error

        return {'uniform_sample_loss': uniform_sample_loss,
                "near_surface_sample_loss":nss_sample_loss,
                'fixed_bounding_box_loss': inside_box_loss}

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
    verts=verts+centroid.squeeze(2).unsqueeze(1)

    return verts

def get_bdb_3d_result(bins_tensor,ori_cls,ori_reg_result,centroid_cls,centroid_reg_result,
                      size_cls,size_reg_result,P_result,camera_K,camera_R,split):
    patch_size=ori_cls.shape[0]
    coeffs=(size_reg_result+1)*bins_tensor['avg_size'][torch.argmax(size_cls,dim=1),:]
    #print(bins_tensor['avg_size'][torch.argmax(size_cls,dim=1),:].shape)
    #print(size_reg_result.shape)
    centroid_depth=torch.mean(bins_tensor['centroid_bin'][centroid_cls.long(),:],dim=1)+\
                   centroid_reg_result[torch.arange(patch_size),centroid_cls.long()]*DEPTH_WIDTH
    centroid=rgb_to_world(P_result,centroid_depth,camera_K,camera_R,split.long())

    ori=torch.mean(bins_tensor["ori_bin"][ori_cls.long(),:],dim=1)+\
        ori_reg_result[torch.arange(patch_size),ori_cls.long()]*ORI_BIN_WIDTH
    bdb3d=get_corners_of_bb3d(coeffs,centroid,ori)

    return bdb3d

def get_bdb_2d_result(bdb3D_result,camera_R_inv,camera_K,split):
    K_ex = torch.cat([camera_K[index].expand(interval[1] - interval[0], -1, -1) for index, interval in enumerate(split)], 0)
    cam_R_inv_ex = torch.cat(
        [camera_R_inv[index].expand(interval[1] - interval[0], -1, -1) for index, interval in enumerate(split)], 0)
    bdb3D_incam=torch.einsum("ijk,iqk->iqj",cam_R_inv_ex,bdb3D_result)
    bdb3D_inimg=torch.einsum("ijk,iqk->iqj",K_ex,bdb3D_incam)
    bdb_x=bdb3D_inimg[:,:,0]/bdb3D_inimg[:,:,2]
    bdb_y = bdb3D_inimg[:, :, 1] / bdb3D_inimg[:, :, 2]
    x_min=torch.clamp(torch.min(bdb_x,dim=1)[0],min=0)
    x_max=torch.clamp(torch.max(bdb_x,dim=1)[0],max=1296)
    y_min=torch.clamp(torch.min(bdb_y,dim=1)[0],min=0)
    y_max=torch.clamp(torch.max(bdb_y,dim=1)[0],max=968)
    #print(x_min.shape)
    bdb2d=torch.cat([x_min[:,None],y_min[:,None],x_max[:,None],y_max[:,None]],dim=1)
    return bdb2d

class JointLoss(BaseLoss):
    def __call__(self, est_data, gt_data, bins_tensor, layout_results):
        # predicted camera rotation
        cam_R_result,cam_R_inv = get_rotation_matix_result(bins_tensor,
                                                 gt_data['pitch_cls'], est_data['pitch_reg_result'],
                                                 gt_data['roll_cls'], est_data['roll_reg_result'])
        # projected center
        P_result = torch.stack(
            ((gt_data['bdb2D_pos'][:, 0] + gt_data['bdb2D_pos'][:, 2]) / 2 - (gt_data['bdb2D_pos'][:, 2] - gt_data['bdb2D_pos'][:, 0]) * est_data['offset_2D_result'][:, 0],
             (gt_data['bdb2D_pos'][:, 1] + gt_data['bdb2D_pos'][:, 3]) / 2 - (gt_data['bdb2D_pos'][:, 3] - gt_data['bdb2D_pos'][:, 1]) * est_data['offset_2D_result'][:, 1]), 1)

        # retrieved 3D bounding box
        bdb3D_result = get_bdb_3d_result(bins_tensor,
                                            gt_data['ori_cls'],
                                            est_data['ori_reg_result'],
                                            gt_data['centroid_cls'],
                                            est_data['centroid_reg_result'],
                                            gt_data['size_cls'],
                                            est_data['size_reg_result'],
                                            P_result,
                                            gt_data['K'],
                                            cam_R_result,
                                            gt_data['split'])

        # 3D bounding box corner loss
        corner_loss = 5 * cls_reg_ratio * reg_criterion(bdb3D_result, gt_data['bdb3D'])

        # 2D bdb loss
        bdb2D_result = get_bdb_2d_result(bdb3D_result,cam_R_inv,gt_data['K'],gt_data['split'])
        #print(bdb2D_result[0],gt_data['bdb2D_pos'][0])
        bdb2D_loss = 20*cls_reg_ratio * reg_criterion(bdb2D_result/1296, gt_data['bdb2D_pos']/1296)

        # physical violation loss
        #print(layout_results['lo_bdb3D_result'], bdb3D_result)
        phy_violation, phy_gt = physical_violation(layout_results['lo_bdb3D_result'], bdb3D_result, gt_data['split'].long())
        phy_loss = 20 * mse_criterion(phy_violation, phy_gt)

        return {'phy_loss':phy_loss,"corner_loss":corner_loss,'bdb2D_loss':bdb2D_loss},\
               {'cam_R_result':cam_R_result, 'bdb3D_result':bdb3D_result}

def get_phy_loss_samples(ldif, structured_implicit, ldif_center, ldif_coef, phy_loss_samples,
                         return_range=False, surface_optimize=False):
    # get inside points from blob centers
    centers = structured_implicit.all_centers.clone()
    sample_points = centers

    # get inside points with random sampling
    bbox_samples = (torch.rand([len(centers), phy_loss_samples * (3 if surface_optimize else 10), 3],
                               device=centers.device) - 0.5) * 2 * ldif_coef.unsqueeze(1) + ldif_center.unsqueeze(1)
    sample_points = torch.cat([sample_points, bbox_samples], 1)

    # optimize to get surface points
    if surface_optimize:
        surface_samples = sample_points.clone()
        surface_samples.requires_grad = True
        ldif_grad = []
        for param in ldif.parameters():
            ldif_grad.append(param.requires_grad)
            param.requires_grad = True
        optimizer = torch.optim.SGD([surface_samples], 200)
        with torch.enable_grad():
            for i in range(10):
                optimizer.zero_grad()
                est_sdf = ldif(
                    samples=surface_samples,
                    structured_implicit=structured_implicit.dict(),
                    apply_class_transfer=False,
                )['global_decisions'] + 0.07
                # from external.PIFu.lib import sample_util
                # sample_util.save_samples_truncted_prob(f'out/inside_sample_{i}.ply', surface_samples[0].detach().cpu(),
                #                                        (est_sdf[0] < 0).detach().cpu())
                error = torch.mean(abs(est_sdf))
                error.backward()
                optimizer.step()
                # print(f"({i}) sdf error: {error:.4f}")
        for param, requires_grad in zip(ldif.parameters(), ldif_grad):
            param.requires_grad = requires_grad
        surface_samples.requires_grad = False
        sample_points = torch.cat([sample_points, surface_samples], 1)

    # remove outside points
    est_sdf = ldif(
        samples=sample_points,
        structured_implicit=structured_implicit.dict(),
        apply_class_transfer=True,
    )['global_decisions']
    inside_samples = []
    in_coor_min = []
    in_coor_max = []
    for i, (s, mask) in enumerate(zip(sample_points, est_sdf < 0.5)):
        inside_sample = s[mask.squeeze(), :]
        if return_range:
            if len(inside_sample) > 0:
                inside_min = inside_sample.min(0)[0]
                inside_max = inside_sample.max(0)[0]
            if len(inside_sample) <= 0 or (inside_min == inside_max).sum() > 0:
                inside_min = centers[i].min(0)[0]
                inside_max = centers[i].max(0)[0]
            in_coor_min.append(inside_min)
            in_coor_max.append(inside_max)
        # from external.PIFu.lib import sample_util
        # sample_util.save_samples_truncted_prob('out/inside_sample.ply', s.detach().cpu(),
        #                                        mask.detach().cpu())
        if len(inside_sample) <= 0:
            inside_sample = centers[i]
        # from external.PIFu.lib import sample_util
        # sample_util.save_samples_truncted_prob('out/inside_sample.ply', inside_sample.detach().cpu(),
        #                                        np.zeros(inside_sample.shape[0]))
        p_ids = np.random.choice(len(inside_sample), phy_loss_samples, replace=True)
        inside_sample = inside_sample[p_ids]
        inside_samples.append(inside_sample)
    inside_samples = torch.stack(inside_samples)

    if return_range:
        in_coor_min = torch.stack(in_coor_min)
        in_coor_max = torch.stack(in_coor_max)
        return inside_samples, in_coor_min, in_coor_max
    return inside_samples

class LDIFReconLoss(BaseLoss):
    def __call__(self, est_data, gt_data, extra_results,config):
        ldif = est_data['mgn']
        get_phy_loss = config['loss_weights']['ldif_phy_loss']
        device = gt_data['patch'].device

        loss_settings = config['model']['loss_settings']
        loss_type = loss_settings['type']
        scale_before_func = loss_settings['scale_before_func']
        phy_loss_samples = loss_settings['phy_loss_samples']
        phy_loss_objects = loss_settings['phy_loss_objects']
        surface_optimize = config['model']['loss_settings']['surface_optimize']
        sdf_data = {}

        if get_phy_loss:
            bdb3D_form = get_bdb_form_from_corners(extra_results['bdb3D_result'])
            structured_implicit = est_data['structured_implicit']
            ldif_center, ldif_coef = est_data['obj_center'], est_data['obj_coef']
            if 'ldif_sampling_points' in est_data:
                inside_samples = est_data['ldif_sampling_points']
            else:
                obj_center = ldif_center.clone()
                #obj_center[:, 2] *= -1
                inside_samples = get_phy_loss_samples(ldif, structured_implicit, obj_center, ldif_coef,
                                                      phy_loss_samples, surface_optimize=surface_optimize)

            # put points to other objects' coor
            #inside_samples[:, :, 2] *= -1
            obj_samples = recover_points_to_world_sys(bdb3D_form, inside_samples, ldif_center, ldif_coef)
            max_sample_points = (gt_data['split'][:, 1] - gt_data['split'][:, 0] - 1).max() * obj_samples.shape[1]
            if max_sample_points == 0:
                sdf_data['ldif_phy_loss'] = None
            else:
                est_sdf = []
                for start, end in gt_data['split']:
                    assert end > start
                    if end > start + 1:
                        centroids = bdb3D_form['centroid'][start:end]
                        centroids = centroids.unsqueeze(0).expand(len(centroids), -1, -1)
                        distances = F.pairwise_distance(
                            centroids.reshape(-1, 3), centroids.transpose(0, 1).reshape(-1, 3), 2
                        ).reshape(len(centroids), len(centroids))
                        for obj_ind in range(start, end):
                            other_obj_dis = distances[obj_ind - start]
                            _, nearest = torch.sort(other_obj_dis)
                            other_obj_sample = obj_samples[start:end].index_select(
                                0, nearest[1:phy_loss_objects + 1]).reshape(-1, 3)
                            other_obj_sample = recover_points_to_obj_sys(
                                {k: v[obj_ind:obj_ind + 1] for k, v in bdb3D_form.items()},
                                other_obj_sample.unsqueeze(0),
                                ldif_center[obj_ind:obj_ind + 1],
                                ldif_coef[obj_ind:obj_ind + 1]
                            )
                            #other_obj_sample[:, :, 2] *= -1
                            sdf = ldif(
                                samples=other_obj_sample,
                                structured_implicit=structured_implicit[obj_ind:obj_ind + 1].dict(),
                                apply_class_transfer=False,
                            )['global_decisions']
                            est_sdf.append(sdf.squeeze())
                if len(est_sdf) == 0:
                    sdf_data['ldif_phy_loss'] = None
                else:
                    est_sdf = torch.cat(est_sdf) + 0.07
                    est_sdf[est_sdf > 0] = 0
                    gt_sdf = torch.full(est_sdf.shape, 0., device=device, dtype=torch.float32)
                    sdf_data['ldif_phy_loss'] = (est_sdf, gt_sdf)

        # compute final loss
        loss = {}
        if not isinstance(loss_type, list):
            loss_type = [loss_type] * len(sdf_data)
        for lt, (k, sdf) in zip(loss_type, sdf_data.items()):
            if sdf is None:
                loss[k] = 0.
            else:
                est_sdf, gt_sdf = sdf

                if 'class' in lt:
                    est_sdf = torch.sigmoid(scale_before_func * est_sdf) - 0.5
                    gt_sdf[gt_sdf > 0] = 0.5
                    gt_sdf[gt_sdf < 0] = -0.5
                elif 'sdf' in lt:
                    est_sdf = scale_before_func * est_sdf
                else:
                    raise NotImplementedError

                if 'mse' in lt:
                    point_loss = nn.MSELoss()(est_sdf, gt_sdf)
                elif 'l1' in lt:
                    point_loss = nn.L1Loss()(est_sdf, gt_sdf)
                elif 'sl1' in lt:
                    point_loss = nn.SmoothL1Loss()(est_sdf, gt_sdf)
                else:
                    raise NotImplementedError

                loss[k] = point_loss

        return loss

class PIFuReconLoss(BaseLoss):
    def __call__(self, est_data, gt_data, extra_results, config):
        get_phy_loss = config['loss_weights']['phy_loss']
        device = gt_data['patch'].device

        loss_settings = config['model']['loss_settings']
        loss_type = loss_settings['type']
        scale_before_func = loss_settings['scale_before_func']
        phy_loss_samples = loss_settings['phy_loss_samples']
        phy_loss_objects = loss_settings['phy_loss_objects']
        surface_optimize = config['model']['loss_settings']['surface_optimize']
        sdf_data = {}
        est_data = parse_bbox(est_data, gt_data)
        if get_phy_loss:
            object_samples=est_data['cube_coor']
            samples_in_cam=PIFu_recover_points_to_world_sys(est_data,object_samples)
        est_occ=[]
        overlap_mask=[]
        for idx,(start,end) in enumerate(gt_data['split'].long()):
            assert end>start
            if end>start+1:
                centroids=est_data['obj_cam_center'][start:end]
                centroids=centroids.unsqueeze(0).expand(centroids.shape[0],-1,-1)
                distance=F.pairwise_distance(centroids.reshape(-1,3),centroids.transpose(0,1).reshape(-1,3),2).reshape(centroids.shape[0],centroids.shape[0])
                for obj_ind in range(start,end):
                    other_obj_dis=distance[obj_ind-start]
                    _,nearest=torch.sort(other_obj_dis)
                    other_obj_sample=samples_in_cam[start:end].index_select(
                        0,nearest[1:phy_loss_objects+1]
                    ).reshape(-1,3) #batch id is 0~4?
                    other_obj_sample_in_currobj=PIFu_recover_points_to_obj_sys(est_data['rot_matrix'][obj_ind:obj_ind+1],
                                                                    est_data['obj_cam_center'][obj_ind:obj_ind+1],
                                                                    est_data['bbox_size'][obj_ind:obj_ind+1],
                                                                    other_obj_sample)
                    mask=(other_obj_sample_in_currobj[:,:,0]<1) & (other_obj_sample_in_currobj[:,:,0]>-1) & (other_obj_sample_in_currobj[:,:,1]<1) \
                    & (other_obj_sample_in_currobj[:,:,1]>-1)& (other_obj_sample_in_currobj[:,:,2]<1)& (other_obj_sample_in_currobj[:,:,2]>-1)
                    est_occ.append(est_data['occ_cube'][start:end].index_select(
                        0,nearest[1:phy_loss_objects+1]
                    ).reshape(1,-1).float())
                    overlap_mask.append(mask.float())
        point_loss=0
        for i in range(len(est_occ)):
            print(torch.max(scale_before_func*est_occ[i]*overlap_mask[i]))
            point_loss+=nn.MSELoss(torch.sigmoid(scale_before_func*est_occ[i]*overlap_mask[i])-0.5,0)
        point_loss=point_loss/len(est_occ)
        return {'pifu_phy_loss':point_loss}


def PIFu_recover_points_to_obj_sys(rot_matrix,obj_cam_center,bbox_size,other_obj_sample):
    other_obj_sample=other_obj_sample.unsqueeze(0)
    inv_rot_list=[t.inverse() for t in torch.unbind(rot_matrix,dim=0)]
    inv_rot_matrix=torch.stack(inv_rot_list)
    obj_sample=other_obj_sample-obj_cam_center.unsqueeze(1)
    obj_sample[:,:,0:2]=-obj_sample[:,:,0:2]
    obj_sample=torch.einsum('ijk,ikq->ijq', obj_sample, inv_rot_matrix.transpose(1, 2))
    obj_sample=obj_sample*2/bbox_size.unsqueeze(1)

    return obj_sample

def PIFu_recover_points_to_world_sys(est_data,inside_samples):
    rot_matrix=est_data['rot_matrix']
    obj_cam_center=est_data['obj_cam_center']
    bbox_size=est_data['bbox_size']


    samples_incam = inside_samples * bbox_size.unsqueeze(1) / 2
    samples_incam = torch.einsum('ijk,ikq->ijq', samples_incam, rot_matrix.transpose(1, 2))
    samples_incam[:, :, 0:2] = -samples_incam[:, :, 0:2]  # y down coordinate
    samples_incam[:, :, 0:3] = samples_incam[:, :, 0:3] + obj_cam_center.unsqueeze(1)

    return samples_incam

def parse_bbox(est_data,gt_data):
    pitch_cls,pitch_reg_result=gt_data['pitch_cls'].long(),est_data['pitch_reg_result']
    roll_cls,roll_reg_result=gt_data['roll_cls'].long(),est_data['roll_reg_result']
    ori_cls,ori_reg_result=gt_data['ori_cls'].long(),est_data['ori_cls_result']
    centroid_cls,centroid_reg_result=gt_data['centroid_cls'].long(),est_data['centroid_reg_result']
    size_reg_result=est_data['size_reg_result']
    size_cls=gt_data['size_cls']
    P_result = torch.stack(
        ((gt_data['bdb2D_pos'][:, 0] + gt_data['bdb2D_pos'][:, 2]) / 2 - (
                    gt_data['bdb2D_pos'][:, 2] - gt_data['bdb2D_pos'][:, 0]) * est_data['offset_2D_result'][:, 0],
         (gt_data['bdb2D_pos'][:, 1] + gt_data['bdb2D_pos'][:, 3]) / 2 - (
                     gt_data['bdb2D_pos'][:, 3] - gt_data['bdb2D_pos'][:, 1]) * est_data['offset_2D_result'][:, 1]), 1)
    camera_K=gt_data["K"]
    split=gt_data["split"]

    patch_size = ori_cls.shape[0]
    batch_size=roll_cls.shape[0]

    ori_reg=ori_reg_result[torch.arange(patch_size),ori_cls]
    ori=torch.mean(bins_tensor['ori_bin'][ori_cls],dim=1)+ORI_BIN_WIDTH*ori_reg
    roll_reg=roll_reg_result[torch.arange(batch_size),roll_cls]
    roll=torch.mean(bins_tensor['roll_bin'][roll_cls],dim=1)+ROLL_WIDTH*roll_reg
    pitch_reg=pitch_reg_result[torch.arange(batch_size),pitch_cls]
    pitch=torch.mean(bins_tensor['pitch_bin'][pitch_cls],dim=1)+PITCH_WIDTH*pitch_reg
    centroid_reg=centroid_reg_result[torch.arange(patch_size),centroid_cls]

    coeffs = (size_reg_result + 1) * bins_tensor['avg_size'][torch.argmax(size_cls, dim=1), :]
    centroid_depth = torch.mean(bins_tensor['centroid_bin'][centroid_cls, :], dim=1) + centroid_reg * DEPTH_WIDTH
    cam_R_result = get_rotation_matix_result(bins_tensor,
                                             gt_data['pitch_cls'].long(), est_data['pitch_reg_result'],
                                             gt_data['roll_cls'].long(), est_data['roll_reg_result'])
    obj_cam_center = rgb_to_world(P_result, centroid_depth, camera_K, cam_R_result, list(split.long())).squeeze(2)

    ori = torch.mean(bins_tensor["ori_bin"][ori_cls, :], dim=1) + \
          ori_reg_result[torch.arange(patch_size), ori_cls.long()] * ORI_BIN_WIDTH
    pitch_boardcast=torch.zeros((patch_size)).to(ori.device)
    roll_boardcast=torch.zeros((patch_size)).to(ori.device)
    for index, interval in enumerate(list(split.long())):
        pitch_boardcast[interval[0]:interval[1]]=pitch[index]
        roll_boardcast[interval[0]:interval[1]]=roll[index]

    rot_matrix=R_from_yaw_pitch_roll(ori+np.pi,pitch_boardcast,-roll_boardcast)

    est_data['rot_matrix']=rot_matrix
    est_data['obj_cam_center']=obj_cam_center
    est_data['bbox_size']=coeffs
    return est_data

def rgb_to_cam(p, depth, K, split):
    n = p.size(0)

    K_ex = torch.cat([K[index].expand(interval[1] - interval[0], -1, -1) for index, interval in enumerate(split)], 0)
    x_temp = (p[:, 0] - K_ex[:, 0, 2]) / K_ex[:, 0, 0]
    y_temp = (p[:, 1] - K_ex[:, 1, 2]) / K_ex[:, 1, 1]
    z_temp = 1
    ratio = depth / torch.sqrt(x_temp ** 2 + y_temp ** 2 + z_temp ** 2)
    x_cam = x_temp * ratio
    y_cam = y_temp * ratio
    z_cam = z_temp * ratio
    x3 = x_cam
    y3 = y_cam
    z3 = z_cam

    p_cam = torch.stack((x3, y3, z3), 1).view(n, 3, 1)  # n x 3
    return p_cam

def get_imgcoor_input(samples_incan,rot_matrix,bbox_size,obj_cam_center,K,bdb2D,use_crop=True):
    samples_inrecan = torch.einsum('ijk,ikq->ijq', samples_incan, rot_matrix.transpose(1, 2))
    z_feat = samples_inrecan[:, :, 2:3]
    samples_incam = samples_incan * bbox_size.unsqueeze(1) / 2
    samples_incam = torch.einsum('ijk,ikq->ijq', samples_incam, rot_matrix.transpose(1, 2))
    samples_incam[:, :, 0:2] = -samples_incam[:, :, 0:2]  # y down coordinate
    # print(samples_incam.shape,obj_cam_center.shape)
    samples_incam[:, :, 0:3] = samples_incam[:, :, 0:3] + obj_cam_center.unsqueeze(1)

    img_samples = torch.einsum('ijk,ikq->ijq', samples_incam[:, :, 0:3], K.transpose(1, 2))
    width = K[:, 0, 2] * 2
    height = K[:, 1, 2] * 2
    x_coor = img_samples[:, :, 0] / img_samples[:, :, 2]  # these are image coordinate
    y_coor = img_samples[:, :, 1] / img_samples[:, :, 2]  # these are image coordinate
    if use_crop:
        x_coor = x_coor - (bdb2D[:, None, 0] + bdb2D[:, None, 2]) / 2
        y_coor = y_coor - (bdb2D[:, None, 1] + bdb2D[:, None, 3]) / 2
        x_coor = x_coor / (bdb2D[:, None, 2] - bdb2D[:, None, 0]) * 2
        y_coor = y_coor / (bdb2D[:, None, 3] - bdb2D[:, None, 1]) * 2
    else:
        x_coor = ((x_coor - width / 2) / width) * 2
        y_coor = ((y_coor - height / 2) / height) * 2
    img_coor = torch.cat([x_coor[:, :, None], y_coor[:, :, None]], dim=2)
    return img_coor,z_feat

def pifu_get_phy_loss_samples(est_data,gt_data,use_crop=True,surface_optimize=False,use_gt=True):
    patch_size=gt_data['rot_matrix'].shape[0]
    samples_in_bbox=(torch.rand([patch_size,1000,3]).to(gt_data['rot_matrix'].device)-0.5)*2 #get 128 sample points per objects
    mgn=est_data["mgn"]
    if use_gt:
        rot_matrix=gt_data['rot_matrix']
        bbox_size=gt_data['bbox_size']
        obj_cam_center=gt_data['obj_cam_center']
    else:
        rot_matrix=est_data['rot_matrix']
        bbox_size=est_data['bbox_size']
        obj_cam_center=est_data['obj_cam_center']
    K=gt_data['K']
    cls_codes=gt_data['cls_codes']
    whole_image=gt_data['image']
    patch=gt_data['patch']
    bdb_grid=gt_data['bdb_grid']
    bdb2D=gt_data['bdb2D_pos']
    split=gt_data['obj_split'].long()
    K_array = torch.zeros((patch_size, 3, 3)).to(rot_matrix.device)
    for idx, (start, end) in enumerate(split):
        K_array[start:end] = K[idx:idx + 1]
    if not surface_optimize:
        img_coor,z_feat=get_imgcoor_input(samples_in_bbox,rot_matrix,bbox_size,obj_cam_center,K_array,bdb2D,use_crop=True)
        mgn.obj_split=split
        mgn.filter(whole_image,patch)
        mgn.query(points=samples_in_bbox,bdb_grid=bdb_grid, z_feat=z_feat, transforms=None, cls_codes=cls_codes,
               img_coor=img_coor)
        pred=mgn.get_preds()
        pred_list=[]
        for i in range(samples_in_bbox.shape[0]):
            inside_sample = samples_in_bbox[i, pred[i] > 0.5, :]
            weights = torch.ones([inside_sample.shape[0]]).to(inside_sample.device)
            if inside_sample.shape[0] > 512:
                idx = torch.multinomial(weights, 512, replacement=False)
            else:
                idx = torch.multinomial(weights, 512, replacement=True)
            pred_list.append(inside_sample[idx])
        return torch.stack(pred_list)
    else:
        surface_samples=samples_in_bbox.clone()
        surface_samples.requires_grad=True
        optimizer=torch.optim.SGD([surface_samples],200)
        with torch.enable_grad():
            for i in range(10):
                optimizer.zero_grad()
                img_coor,z_feat=get_imgcoor_input(surface_samples,rot_matrix,bbox_size,obj_cam_center,K_array,bdb2D,use_crop=True)
                mgn.obj_split = split
                mgn.filter(whole_image, patch)
                mgn.query(points=samples_in_bbox, bdb_grid=bdb_grid, z_feat=z_feat, transforms=None,
                          cls_codes=cls_codes,
                          img_coor=img_coor)
                pred = mgn.get_preds()
                error=torch.mean(torch.abs(pred-0.5))
                error.backward()
                optimizer.step()
        pred_list = []
        for i in range(pred.shape[0]):
            inside_sample=surface_samples[i,pred[i]>0.5,:]
            weights=torch.ones([inside_sample.shape[0]]).to(inside_sample.device)
            if inside_sample.shape[0]>128:
                idx=torch.multinomial(weights,128,replacement=False)
            else:
                idx = torch.multinomial(weights, 128, replacement=True)
            pred_list.append(inside_sample[idx])
        return torch.stack(pred_list)




