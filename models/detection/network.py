import torch
from torch import nn
from configs.data_config import obj_cam_ratio
# from external.ldif.representation.structured_implicit_function import StructuredImplicit
# import numpy as np
# from models.loss import get_phy_loss_samples
from models.detection.object_detection import Bdb3DNet
from models.detection.layout_estimation import PoseNet
from models.detection.total3d_loss import DetLoss,PoseLoss,JointLoss,LDIFReconLoss
from net_utils.bins import bin
from models.detection.gcnn import GCNN
from models.detection.mesh_reconstruction import LDIF_joint
from external.ldif.representation.structured_implicit_function import StructuredImplicit
from net_utils.bins import *

class TOTAL3D(nn.Module):

    def __init__(self, cfg):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(TOTAL3D, self).__init__()
        self.config = cfg
        if self.config["phase"] in ["object_detection","joint"]:
            self.object_detection=Bdb3DNet(cfg)
            self.object_detection_loss=DetLoss()
        if self.config["phase"] in ["layout_estimation", "joint"]:
            self.layout_estimation=PoseNet(cfg)
            self.layout_estimation_loss=PoseLoss()
        if self.config["phase"] in ["joint"]:
            self.mesh_reconstruction=LDIF_joint(cfg)
            self.output_adjust=GCNN(cfg)
            self.mesh_reconstruction_loss=LDIFReconLoss()
            self.joint_loss=JointLoss()
        self.freeze_modules(cfg)

    def forward(self, data):
        #for key in data:
        #    print(key,data[key].shape)
        all_output = {}

        if self.config["phase"] in ["layout_estimation","joint"]:
            pitch_reg_result, roll_reg_result, \
            pitch_cls_result, roll_cls_result, \
            lo_centroid_result, lo_coeffs_result, a_features = self.layout_estimation(data['image'])

            layout_output = {'pitch_reg_result': pitch_reg_result, 'roll_reg_result': roll_reg_result,
                             'pitch_cls_result': pitch_cls_result, 'roll_cls_result': roll_cls_result,
                             'lo_centroid_result': lo_centroid_result, 'lo_coeffs_result': lo_coeffs_result,
                             'lo_afeatures': a_features}
            all_output.update(layout_output)

        if self.config['phase'] in ['object_detection', 'joint']:
            size_reg_result, \
            ori_reg_result, ori_cls_result, \
            centroid_reg_result, centroid_cls_result, \
            offset_2D_result, a_features, \
            r_features, a_r_features = self.object_detection(data['patch'], data['size_cls'], data['g_features'],
                                                             data['split'], data['rel_pair_counts'],data["box_feat"])
            object_output = {'size_reg_result':size_reg_result, 'ori_reg_result':ori_reg_result,
                             'ori_cls_result':ori_cls_result, 'centroid_reg_result':centroid_reg_result,
                             'centroid_cls_result':centroid_cls_result, 'offset_2D_result':offset_2D_result,
                             'odn_afeature': a_features, 'odn_rfeatures': r_features, 'odn_arfeatures': a_r_features}
            all_output.update(object_output)
        if self.config['phase'] in ['joint']:
            mesh_output=self.mesh_reconstruction(data['patch'],data['size_cls'],structured_implicit=True)
            if 'structured_implicit' in mesh_output:
                mesh_output['structured_implicit'] = StructuredImplicit(
                    config=self.config, **mesh_output['structured_implicit'])
            mesh_output['mgn']=self.mesh_reconstruction
            all_output.update(mesh_output)
            all_output.update(self.get_extra_results(all_output))
            if hasattr(self,'output_adjust'):
                input = all_output.copy()
                input['size_cls'] = data['size_cls']
                input['cls_codes'] = data['size_cls']
                input['g_features'] = data['g_features']
                input['bdb2D_pos'] = data['bdb2D_pos']
                input['K'] = data['K']
                input['split'] = data['split']
                input['rel_pair_counts'] = data['rel_pair_counts']
                refined_output = self.output_adjust(input)
                all_output.update(refined_output)

        loss_dict=self.loss(all_output,data)

        return all_output,loss_dict

    def get_extra_results(self,all_output):
        extra_results={}
        structured_implicit=all_output['structured_implicit']
        in_coor_min = structured_implicit.all_centers.min(dim=1)[0]
        in_coor_max = structured_implicit.all_centers.max(dim=1)[0]

        obj_center = (in_coor_max + in_coor_min) / 2.
        #obj_center[:, 1] *= -1 #camera coordiante is y down
        obj_coef = (in_coor_max - in_coor_min) / 2.
        extra_results.update({'obj_center':obj_center,'obj_coef':obj_coef})
        return extra_results

    def loss(self, est_data, gt_data):
        '''
        calculate loss of est_out given gt_out.
        '''
        # loss_weights = self.cfg.config.get('loss_weights', {})
        # if self.cfg.config[self.cfg.config['mode']]['phase'] in ['layout_estimation', 'joint']:
        #     layout_loss, layout_results = self.layout_estimation_loss(est_data, gt_data, self.cfg.bins_tensor)
        #     layout_loss_weighted = {k: v * loss_weights.get(k, 1.0) for k, v in layout_loss.items()}
        #     total_layout_loss = sum(layout_loss_weighted.values())
        #     total_layout_loss_unweighted = sum([v.detach() for v in layout_loss.values()])
        #     for key, value in layout_loss.items():
        #         layout_loss[key] = value.item()
        loss_weights = self.config["loss_weights"]
        if self.config["phase"] in ["layout_estimation", "joint"]:
            layout_loss,layout_result = self.layout_estimation_loss(est_data, gt_data)
            layout_loss_weighted = {k: v * loss_weights.get(k, 1.0) for k, v in layout_loss.items()}
            total_layout_loss = sum(layout_loss_weighted.values())
            total_layout_loss_unweighted = sum([v.detach() for v in layout_loss.values()])

            for key, value in layout_loss.items():
                layout_loss[key] = value.item()
        if self.config["phase"] in ["object_detection", "joint"]:
            object_loss = self.object_detection_loss(est_data, gt_data)
            object_loss_weighted = {k: v * loss_weights.get(k, 1.0) for k, v in object_loss.items()}
            total_object_loss = sum(object_loss_weighted.values())
            total_object_loss_unweighted = sum([v.detach() for v in object_loss.values()])
            for key, value in object_loss.items():
                object_loss[key] = value.item()

        if self.config['phase'] in ["joint"]:
            joint_loss,extra_results=self.joint_loss(est_data,gt_data,bins_tensor,layout_result)
            joint_loss_weighted = {k: v * loss_weights.get(k, 1.0) for k, v in joint_loss.items()}
            mesh_loss = self.mesh_reconstruction_loss(est_data, gt_data, extra_results,self.config)
            mesh_loss_weighted = {k: v * loss_weights.get(k, 1.0) for k, v in mesh_loss.items()}

            total_joint_loss = sum(joint_loss_weighted.values()) + sum(mesh_loss_weighted.values())
            total_joint_loss_unweighted = \
                sum([v.detach() for v in joint_loss.values()]) \
                + sum([v.detach() if isinstance(v, torch.Tensor) else v for v in mesh_loss.values()])
            for key, value in mesh_loss.items():
                mesh_loss[key] = float(value)
            for key, value in joint_loss.items():
                joint_loss[key] = value.item()

        if self.config["phase"] in ["layout_estimation"]:
            return {'total':total_layout_loss, **layout_loss, 'total_unweighted': total_layout_loss_unweighted}
        if self.config["phase"] in ["object_detection"]:
            return {'total': total_object_loss, **object_loss, 'total_unweighted': total_object_loss_unweighted}
        if self.config["phase"] in ["joint"]:
            total3d_loss = total_object_loss + total_joint_loss + obj_cam_ratio * total_layout_loss
            total3d_loss_unweighted = total3d_loss + total_joint_loss_unweighted \
                                      + obj_cam_ratio * total_layout_loss_unweighted
            return {'total': total3d_loss, **layout_loss, **object_loss, **mesh_loss, **joint_loss,
                    'total_unweighted': total3d_loss_unweighted}

    def freeze_modules(self, cfg):
        '''
        Freeze modules in training
        '''
        if cfg['mode'] == 'train':
            freeze_layers = cfg['train']['freeze']
            for layer in freeze_layers:
                if not hasattr(self, layer):
                    continue
                for param in getattr(self, layer).parameters():
                    param.requires_grad = False
                print('The module: %s is fixed.' % (layer))