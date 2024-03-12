import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--yolo_dir",type=str,default="/apdcephfs/private_haolinliu/yolov7")
parser.add_argument("--instpifu_dir",type=str,default="/apdcephfs/private_haolinliu/InstPIFu")
parser.add_argument("--taskid",type=str,default="2003",help="image id from sunrgbd dataset")
parser.add_argument("--sunrgbd_dir",type=str,default="/apdcephfs_cq3/share_1330077/haolinliu/data/sunrgbd_train_test_data")
args=parser.parse_args()

import os,sys
yolo_root=args.yolo_dir
instpifu_root=args.instpifu_dir
sys.path[0]=yolo_root #use yolov7 to conduct 2d detectio
os.chdir(yolo_root)
import torch
from models.experimental import attempt_load
import cv2
import pickle as p
import matplotlib
import matplotlib.pyplot as plt
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
import copy
import math
import numpy as np
import json

conf_thres=0.35
iou_thres=0.45

yolo_ckpt_path=os.path.join(instpifu_root,"checkpoints/yolov7.pt")
device=torch.device("cuda:0")
model = attempt_load(yolo_ckpt_path, map_location=device)  # load FP32 model

model.eval()
sunrgbd_dir=args.sunrgbd_dir
image_id=args.taskid
data_path=os.path.join(sunrgbd_dir,"%s.pkl"%(image_id))
with open(data_path,'rb') as f:
    data=p.load(f)
image=data['rgb_img']
org_image=copy.deepcopy(image)
dst_size=640
stride=32 #make sure inputs resolution can be divided by 32 in both sides
oh,ow=image.shape[0:2]
nh,nw= math.ceil(oh/ow*640/stride)*stride,640 #only apply to images with longer width
image=cv2.resize(image,dsize=(nw,nh),interpolation=cv2.INTER_LINEAR)

image_tensor = torch.from_numpy((image / 255.0).transpose(2, 0, 1)).float().to(device).unsqueeze(0)
_, _, height, width = image_tensor.shape
with torch.no_grad():
    out, _ = model(image_tensor, augment=False)

labelnames = model.names
label_map = {  # map the label back to instpifu's system
    "chair": "chair",
    "couch": "sofa",
    "bed": "bed",
    "dining table": "table",
}
color_map={}
for key in label_map:
    color_map[key]=np.random.randint(0,255,size=3).tolist()
#print(color_map)

out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, classes=None, agnostic=False)
det = out[0]
canvas = copy.deepcopy(image)
save_bboxes_dict=[]
for i in range(det.shape[0]):
    bbox = det[i]
    label = int(bbox[-1])
    classname = labelnames[label]
    if classname in label_map:
        map_classname=label_map[classname]
    else:
        continue
    x_min, y_min, x_max, y_max = bbox[0:4]
    x_min, y_min, x_max, y_max = x_min.item(), y_min.item(), x_max.item(), y_max.item()

    #resize the box to original resolution
    bbox_dict={
        "bbox":[int(x_min / nw * ow), int(y_min / nh * oh), int(x_max / nw * ow), int(y_max / nh * oh)],
        "class":map_classname
    }
    save_bboxes_dict.append(bbox_dict)
    cv2.rectangle(canvas, (int(x_min), int(y_min)), (int(x_max), int(y_max)), thickness=5, color=color_map[classname])

save_folder=os.path.join(instpifu_root,"real_demo",image_id)
os.makedirs(save_folder,exist_ok=True)
cv2.imwrite(os.path.join(save_folder,"2d_detection.jpg"),canvas[:,:,::-1])
cv2.imwrite(os.path.join(save_folder,"img.jpg"),org_image[:,:,::-1])

cam_K=np.array([[529.5000,0.0000,365.0000],
                [0.0000,529.5000,265.0000],
                [0.0000,0.0000,1.0000]])
np.savetxt(os.path.join(save_folder,"cam_K.txt"),cam_K)

with open(os.path.join(save_folder,"detections.json"),'w') as f:
    json.dump(save_bboxes_dict,f)