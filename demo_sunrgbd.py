import numpy as np
import torch
from configs.config_utils import CONFIG
import argparse
from dataset.sunrgbd_dataset import SUNRGBD_Recon_Dataset
from torch.utils.data import DataLoader
from models.instPIFu.InstPIFu_net import InstPIFu
from models.bg_PIFu.BGPIFu_net import BGPIFu_Net
import datetime
import os
import time
import cv2

def dataset2dataloader(dataset):
    dataloader = DataLoader(dataset,
                            num_workers=1,
                            batch_size=1,
                            shuffle=False
                            )
    return dataloader

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Refer-it-in-RGBD demo')
    parser.add_argument('--testid', type=str, default='1180', help='train, test or demo.')
    return parser.parse_args()

if __name__=="__main__":
    args=parse_args()
    '''need to specify which weight files to load in the .yaml file'''
    instPIFu_config_path="./configs/demo_sunrgbd_instPIFu.yaml"
    bg_config_path="./configs/test_bg_PIFu.yaml"

    instPIFu_config=CONFIG(instPIFu_config_path).config
    bg_config = CONFIG(bg_config_path).config
    instPIFu_config['data']['use_pred_pose']=True #to use predict pose or not
    instPIFu_model=InstPIFu(instPIFu_config).cuda()
    instPIFu_checkpoints=torch.load(instPIFu_config["weight"])
    instPIFu_net_weight=instPIFu_checkpoints['net']
    instPIFu_new_net_weight={}
    for key in instPIFu_net_weight:
        if key.startswith("module."):
            k_ = key[7:]
            instPIFu_new_net_weight[k_] = instPIFu_net_weight[key]
    instPIFu_model.load_state_dict(instPIFu_new_net_weight)
    instPIFu_model.eval()

    bg_model = BGPIFu_Net(bg_config).cuda()
    bg_checkpoints = torch.load(bg_config['weight'])
    bg_net_weight = bg_checkpoints['net']
    bg_new_net_weight = {}
    for key in bg_net_weight:
        if key.startswith("module."):
            k_ = key[7:]
            bg_new_net_weight[k_] = bg_net_weight[key]
    bg_model.load_state_dict(bg_new_net_weight)
    bg_model.eval()

    SUNRGBD_recon_dataset=SUNRGBD_Recon_Dataset(instPIFu_config,"test",testid=args.testid)
    SUNRGBD_recon_loader=dataset2dataloader(SUNRGBD_recon_dataset)

    save_folder=os.path.join("outputs",args.testid)
    if os.path.exists(save_folder)==False:
        os.makedirs(save_folder)
    '''inference all objects'''
    start_t=time.time()
    for batch_id, data_batch in enumerate(SUNRGBD_recon_loader):
        for key in data_batch:
            if isinstance(data_batch[key], list) == False:
                data_batch[key] = data_batch[key].float().cuda()
        with torch.no_grad():
            mesh = instPIFu_model.extract_mesh(data_batch, instPIFu_config['data']['marching_cube_resolution'])
            rot_matrix=data_batch["rot_matrix"][0].cpu().numpy()
            obj_cam_center=data_batch["obj_cam_center"][0].cpu().numpy()
            bbox_size=data_batch["bbox_size"][0].cpu().numpy()
            #pitch=data_batch["pitch"][0].cpu().numpy()

            '''transform mesh to camera coordinate'''
            obj_vert=np.asarray(mesh.vertices)
            obj_vert=obj_vert/2*bbox_size
            obj_vert=np.dot(obj_vert,rot_matrix.T)
            obj_vert[:,0:2]=-obj_vert[:,0:2]
            obj_vert+=obj_cam_center
            mesh.vertices=np.asarray(obj_vert.copy())

            object_id=data_batch["obj_id"][0]
            save_path=os.path.join(save_folder,args.testid+"_%s"%(object_id)+".ply")
            print("saving to %s"%(save_path))
            mesh.export(save_path)
        msg = "{:0>8},[{}/{}]".format(
            str(datetime.timedelta(seconds=round(time.time() - start_t))),
            batch_id + 1,
            len(SUNRGBD_recon_loader),
        )
        print(msg)
    bg_PIFu_input={
        "image":data_batch["bg_image"],
        "intrinsic":data_batch["bg_intrinsic"],
    }
    #print(bg_PIFu_input)
    #print(bg_PIFu_input["image"].shape,bg_PIFu_input["intrinsic"])
    with torch.no_grad():
        bg_mesh = bg_model.extract_mesh(bg_PIFu_input, marching_cube_resolution=256)
    save_path = os.path.join(save_folder, "bg.ply")
    print("saving to %s" % (save_path))
    bg_mesh.export(save_path)
    #print(bg_mesh.vertices)

    whole_image=data_batch["whole_image"][0].cpu()*torch.tensor([0.229,0.224,0.225])[:,None,None]+\
    torch.tensor([0.485,0.456,0.406])[:,None,None]
    whole_image=(whole_image.permute(1,2,0).numpy()*255.0).astype(np.uint8)
    save_path=os.path.join(save_folder,"input.jpg")
    cv2.imwrite(save_path,whole_image)





