import torch
import os
import datetime
import time
import cv2
import numpy as np
import pickle

def totalindoor_tester(cfg,model,loss_func,loader,device,checkpoint):
    start_t = time.time()
    config = cfg.config
    log_dir = os.path.join(config['other']["model_save_dir"], config['exp_name'])
    # print(log_dir)
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    cfg.write_config()
    if config['resume'] == True:
        if isinstance(config['weight'],list):
            for path in config['weight']:
                print("loading from", path)
                if "PIFu" in path:
                    load_parallel_checkpoints(path, model,prefix="mesh_reconstruction.",strict=False)
                else:
                    load_parallel_checkpoints(path,model,strict=False)
    model.eval()
    for batch_id, data_batch in enumerate(loader):
        split = data_batch['obj_split']
        # split of relational pairs for batch learning.
        rel_pair_counts = torch.cat([torch.tensor([0]), torch.cumsum(
            torch.pow(split[:, 1] - split[:, 0], 2), 0)], 0)
        data_batch['split']=split
        data_batch['rel_pair_counts']=rel_pair_counts
        for key in data_batch:
            if isinstance(data_batch[key], list) == False:
                data_batch[key] = data_batch[key].float().cuda()
        with torch.no_grad():
            est_data = model(data_batch)
            # est_data,loss_info=model(data_batch)
        # print(loss_info)
        # print(est_data.keys())
        msg = "{:0>8},[{}/{}]".format(
            str(datetime.timedelta(seconds=round(time.time() - start_t))),
            batch_id + 1,
            len(loader),
        )

def instPIFu_tester(cfg,mode,loader,device,checkpoint):
    start_t = time.time()
    config = cfg.config
    log_dir = os.path.join(config['other']["model_save_dir"], config['exp_name'])
    # print(log_dir)
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    cfg.write_config()
    if config['resume'] == True:
        print("loading from",config['weight'])
        checkpoint.load(config['weight'])
        #load_parallel_checkpoints(config['weight'],model,strict=True)
    model.eval()
    for batch_id, data_batch in enumerate(loader):
        for key in data_batch:
            if isinstance(data_batch[key], list) == False:
                data_batch[key] = data_batch[key].float().cuda()
        with torch.no_grad():
            #try:
            mesh=model.extract_mesh(data_batch,config['data']['marching_cube_resolution'])
            #except:
            #    continue
            #est_data,loss_info=model(data_batch)
        #print(loss_info)
        # print(est_data.keys())
        msg = "{:0>8},[{}/{}]".format(
            str(datetime.timedelta(seconds=round(time.time() - start_t))),
            batch_id + 1,
            len(loader),
        )
        print(msg)
        '''export result of object reconstruction'''
        if config['method']=="instPIFu":
            #jid=data_batch['jid'][0]
            taskid=data_batch['taskid'][0]
            object_id=data_batch["obj_id"][0]
            m_save_path=os.path.join(log_dir,taskid+"_"+str(object_id)+".ply")
            print(m_save_path,data_batch['jid'][0])
            #m_save_path=os.path.join(log_dir,taskid+".ply")
            mesh.export(m_save_path)
        '''export result of background reconstruction'''
        elif config['method']=="PIFu_bg":
            taskid=data_batch['taskid'][0]
            m_save_path = os.path.join(log_dir, taskid + ".ply")
            # m_save_path=os.path.join(log_dir,taskid+".ply")
            mesh.export(m_save_path)