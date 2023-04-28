import torch
import os
import datetime
import time
import pickle
import numpy as np

def Recon_tester(cfg,model,loader,device,checkpoint):
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
    model.eval()
    for batch_id, data_batch in enumerate(loader):
        for key in data_batch:
            if isinstance(data_batch[key], list) == False:
                data_batch[key] = data_batch[key].float().cuda()
        with torch.no_grad():
            #print(data_batch['sequence_id'])
            mesh=model.extract_mesh(data_batch,config['data']['marching_cube_resolution'])
            if config['other']['scale_back']:
                bbox_size = data_batch["bbox_size"][0].cpu().numpy()
                '''transform mesh to camera coordinate'''
                obj_vert = np.asarray(mesh.vertices)
                obj_vert = obj_vert / 2 * bbox_size
                mesh.vertices = np.asarray(obj_vert.copy())
        msg = "{:0>8},[{}/{}]".format(
            str(datetime.timedelta(seconds=round(time.time() - start_t))),
            batch_id + 1,
            len(loader),
        )
        print(msg)
        '''export result of object reconstruction'''
        if config['method']=="instPIFu":
            taskid=data_batch['taskid'][0]
            object_id=data_batch["obj_id"][0]
            m_save_path=os.path.join(log_dir,taskid+"_"+str(object_id)+".ply")
            #print(m_save_path,data_batch['jid'][0])
            print("saving to %s"%(m_save_path))
            mesh.export(m_save_path)
        elif config['method']=="bgPIFu":
            taskid = data_batch['taskid'][0]
            m_save_path = os.path.join(log_dir, taskid + ".ply")
            print("saving to %s" % (m_save_path))
            mesh.export(m_save_path)


def Det_tester(cfg,model,loader,device,checkpoint):
    from net_utils.tools import convert_result,total3d_todevice
    start_t=time.time()
    config = cfg.config
    log_dir = os.path.join(config['other']["model_save_dir"], config['exp_name'])
    #print(log_dir)
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    cfg.write_config()
    if config['resume'] == True:
        if isinstance(config['weight'],list):
            for weight_path in config["weight"]:
                checkpoint.load(weight_path)
    for batch_id, data_batch in enumerate(loader):
        with torch.no_grad():
            object_input=total3d_todevice(cfg,data_batch,device)
            est_data,_=model(object_input)
        #print(est_data.keys())
        msg = "{:0>8},[{}/{}]".format(
            str(datetime.timedelta(seconds=round(time.time() - start_t))),
            batch_id + 1,
            len(loader),
        )
        print(msg)
        K=object_input["K"]
        patch_size=object_input['patch'].shape[0]
        obj_split=object_input['split'].long()
        K_array = torch.zeros((patch_size, 3, 3)).to(object_input['patch'].device)
        for idx, (start, end) in enumerate(obj_split.long()):
            K_array[start:end] = K[idx:idx + 1]
        save_dict_list=convert_result(object_input,est_data)
        for idx,item in enumerate(save_dict_list):
            sequence_id=item['sequence_id']
            item['K']=K_array[idx].cpu().numpy()
            save_path=os.path.join(log_dir,sequence_id+".pkl")
            with open(save_path,'wb') as f:
                pickle.dump(item,f)
