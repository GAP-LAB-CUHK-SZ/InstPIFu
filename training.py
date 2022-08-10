import torch
from tensorboardX import SummaryWriter
import os
import datetime
import time
import numpy as np
import pickle
import torch.nn as nn
#torch.autograd.set_detect_anomaly(True)

def Recon_trainer(cfg,model,optimizer,scheduler,train_loader,test_loader,device,checkpoint):
    start_t = time.time()
    config = cfg.config
    log_dir = os.path.join(config['other']["model_save_dir"], config['exp_name'])
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    cfg.write_config()
    tb_logger = SummaryWriter(log_dir)
    start_epoch = 0
    if config["resume"] == True:
        checkpoint.load(config["weight"])
        start_epoch = scheduler.last_epoch
    if config['finetune']==True:
        start_epoch=0
    scheduler.last_epoch = start_epoch
    model.train()
    iter = 0
    min_eval_loss = 10000
    for e in range(start_epoch, config['other']['nepoch']):
        cfg.log_string("Switch Phase to Train")
        model.train()
        for batch_id, data_batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            for key in data_batch:
                if isinstance(data_batch[key], list) == False:
                    data_batch[key] = data_batch[key].float().cuda()
            est_data, loss_dict = model(data_batch)
            total_loss = torch.mean(loss_dict["loss"])
            total_loss.backward()
            optimizer.step()
            msg = "{:0>8},{}:{},[{}/{}],{}: {}".format(
                str(datetime.timedelta(seconds=round(time.time() - start_t))),
                "epoch",
                e,
                batch_id + 1,
                len(train_loader),
                "total_loss",
                total_loss.item()
            )
            cfg.log_string(msg)
            # iter += 1
            for loss in loss_dict:
                if "total" not in loss:
                    tb_logger.add_scalar("train/" + loss, torch.mean(loss_dict[loss]).item(), iter)
            tb_logger.add_scalar("train/total_loss", total_loss.item(), iter)
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            tb_logger.add_scalar("train/lr", current_lr, iter)
            if iter%config['other']['visualize_interval']==0:
                rgb = data_batch["image"][0] * torch.tensor([0.229, 0.224, 0.225])[:, None, None].cuda() + torch.tensor(
                    [0.485, 0.456, 0.406])[:, None, None].cuda()
                tb_logger.add_image("rgb", rgb, iter)
            if config['method']=="instPIFu":
                if iter % config['other']['visualize_interval'] == 0 and config['data']['use_instance_mask']:
                    pred_mask=est_data["pred_mask"][0]
                    gt_mask=data_batch['mask'][0]
                    tb_logger.add_image("gt_mask", gt_mask, iter)
                    tb_logger.add_image('pred_mask',pred_mask,iter)
            if config["other"]["dump_result"]==True and iter%config["other"]["dump_interval"]==0 and config["phase"]=="reconstruction":
                #gt_labels=data_batch['inside_class'][0]
                pred_class=est_data['pred_class'][0]
                sample_points=data_batch["samples"][0]
                image=data_batch["image"][0]
                save_dict={
                    "pred_class":pred_class.detach().cpu().numpy(),
                    "sample_points":sample_points.detach().cpu().numpy(),
                    "image":image.detach().cpu().numpy(),
                }
                with open(os.path.join(log_dir,"train_dump_dict_%d.pkl"%(iter)),"wb") as f:
                    pickle.dump(save_dict,f)
            iter += 1
        model.eval()
        eval_loss = 0
        eval_loss_info = {
        }
        cfg.log_string("Switch Phase to Test")
        for batch_id, data_batch in enumerate(test_loader):
            for key in data_batch:
                if isinstance(data_batch[key], list) == False:
                    data_batch[key] = data_batch[key].float().cuda()
            with torch.no_grad():
                est_data, loss_dict = model(data_batch)
            total_loss = torch.mean(loss_dict["loss"])
            msg = "{:0>8},{}:{},[{}/{}],{}: {}".format(
                str(datetime.timedelta(seconds=round(time.time() - start_t))),
                "epoch",
                e,
                batch_id + 1,
                len(test_loader),
                "test_loss",
                total_loss.item()
            )
            for key in loss_dict:
                if "total" not in key:
                    if key not in eval_loss_info:
                        eval_loss_info[key] = 0
                    eval_loss_info[key] += torch.mean(loss_dict[key]).item()

            total_loss = torch.mean(total_loss)
            eval_loss += total_loss.item()
            cfg.log_string(msg)
        avg_eval_loss = eval_loss / (batch_id + 1)
        for key in eval_loss_info:
            eval_loss_info[key] = eval_loss_info[key] / (batch_id + 1)
        print("eval_loss is", avg_eval_loss)
        tb_logger.add_scalar('eval/eval_loss', avg_eval_loss, e)
        for key in eval_loss_info:
            tb_logger.add_scalar("eval/" + key, eval_loss_info[key], e)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_eval_loss)
        else:
            scheduler.step()

        checkpoint.register_modules(epoch=e, min_loss=avg_eval_loss)
        if avg_eval_loss < min_eval_loss:
            checkpoint.save('best')
            min_eval_loss = avg_eval_loss
        else:
            checkpoint.save("latest")
        e += 1