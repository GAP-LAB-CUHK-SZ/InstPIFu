import torch
import os
import datetime
import time

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
            mesh=model.extract_mesh(data_batch,config['data']['marching_cube_resolution'])
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