#!/bin/bash

instpifu_dir=/apdcephfs/private_haolinliu/InstPIFu
yolo_dir=/apdcephfs/private_haolinliu/yolov7 #need to firstly install prerequired package for both yolov7 and im3d
im3d_dir=/apdcephfs/private_haolinliu/Implicit3DUnderstanding
taskid="2004" #taskid from sunrgbd dataset

python 1.run_2d_detection.py --yolo_dir $yolo_dir --instpifu_dir $instpifu_dir --taskid $taskid
python 2.run_3d_detection.py --im3d_dir $im3d_dir --instpifu_dir $instpifu_dir --taskid $taskid
python 3.run_object_reconstruction.py --instpifu_dir $instpifu_dir --taskid $taskid