import argparse,os
parser = argparse.ArgumentParser(description="im3d arg parser")
parser.add_argument('--mode', type=str, default='demo', help='train, test, demo or qtrain, qtest')
parser.add_argument("--instpifu_dir",type=str,required=True)
parser.add_argument("--im3d_dir",type=str,required=True)
parser.add_argument("--taskid",type=str,default="2003",help="image id from sunrgbd dataset")
args=parser.parse_args()
parser.add_argument('--config', type=str, default=os.path.join(args.instpifu_dir,"checkpoints/im3d_weight/out_config.yaml"),
                    help='configure file for training or testing.') #add new argument for the config files
args=parser.parse_args()
import sys
sys.path[0]=args.im3d_dir
os.chdir(args.im3d_dir)
import argparse
from net_utils.utils import CheckpointIO
from net_utils.utils import load_device, load_model
import torch
from configs.config_utils import CONFIG,mount_external_config
from demo import load_demo_data

cfg = CONFIG(parser)
cfg = mount_external_config(cfg)
checkpoint = CheckpointIO(cfg)
device=torch.device("cuda:0")
net=load_model(cfg,device=device)
net.train(False)

ckpt_path=os.path.join(args.instpifu_dir,"./checkpoints/im3d_weight/model_best.pth")
ckpt=torch.load(ckpt_path)
net.load_state_dict(ckpt['net'])

input_folder=os.path.join(args.instpifu_dir,"./real_demo/%s"%(args.taskid))
data=load_demo_data(input_folder,device=device)
with torch.no_grad():
    est_data=net(data)

from net_utils.libs import get_layout_bdb_sunrgbd, get_rotation_matix_result, get_bdb_evaluation
from scipy.io import savemat

lo_bdb3D_out = get_layout_bdb_sunrgbd(cfg.bins_tensor, est_data['lo_ori_reg_result'],
                                  torch.argmax(est_data['lo_ori_cls_result'], 1),
                                  est_data['lo_centroid_result'],
                                  est_data['lo_coeffs_result'])
# camera orientation for evaluation
cam_R_out,pitch,roll = get_rotation_matix_result(cfg.bins_tensor,
                                  torch.argmax(est_data['pitch_cls_result'], 1), est_data['pitch_reg_result'],
                                  torch.argmax(est_data['roll_cls_result'], 1), est_data['roll_reg_result'],
                                                 return_degrees=True)

# projected center
P_result = torch.stack(((data['bdb2D_pos'][:, 0] + data['bdb2D_pos'][:, 2]) / 2 -
                    (data['bdb2D_pos'][:, 2] - data['bdb2D_pos'][:, 0]) * est_data['offset_2D_result'][:, 0],
                    (data['bdb2D_pos'][:, 1] + data['bdb2D_pos'][:, 3]) / 2 -
                    (data['bdb2D_pos'][:, 3] - data['bdb2D_pos'][:, 1]) * est_data['offset_2D_result'][:,1]), 1)

bdb3D_out_form_cpu, bdb3D_out = get_bdb_evaluation(cfg.bins_tensor,
                                               torch.argmax(est_data['ori_cls_result'], 1),
                                               est_data['ori_reg_result'],
                                               torch.argmax(est_data['centroid_cls_result'], 1),
                                               est_data['centroid_reg_result'],
                                               data['size_cls'], est_data['size_reg_result'], P_result,
                                               data['K'], cam_R_out, data['split'], return_bdb=True)

output_folder=input_folder
nyu40class_ids = [int(evaluate_bdb['classid']) for evaluate_bdb in bdb3D_out_form_cpu]
#os.makedirs(output_folder,exist_ok=True)
# save layout
savemat(os.path.join(output_folder, 'layout.mat'),
        mdict={'layout': lo_bdb3D_out[0, :, :].cpu().numpy()})
# save bounding boxes and camera poses
interval = data['split'][0].cpu().tolist()
current_cls = nyu40class_ids[interval[0]:interval[1]]

savemat(os.path.join(output_folder, 'bdb_3d.mat'),
        mdict={'bdb': bdb3D_out_form_cpu[interval[0]:interval[1]], 'class_id': current_cls})
savemat(os.path.join(output_folder, 'r_ex.mat'),
        mdict={'cam_R': cam_R_out[0, :, :].cpu().numpy(),"pitch":pitch.cpu().numpy(),"roll":roll.cpu().numpy()})

