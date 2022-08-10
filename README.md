# Towards High-Fidelity Single-view Holistic Reconstruction of Indoor Scenes (ECCV 2022)
<p align="center"><img src="docs/teaser.png" width="600px"/></p></br>
This is the repository of our paper 'Towards High-Fidelity Single-view Holistic Reconstruction of Indoor Scenes' in ECCV 2022<br>
Paper - <a href="https://arxiv.org/pdf/2207.08656" target="__blank">ArXiv - pdf</a> (<a href="https://arxiv.org/abs/2207.08656" target="__blank">abs</a>) 
<br>

# Environment
prerequsite

1. CUDA 11.1
2. pytorch 1.9.0
3. torchvision 0.10.0

after installing the above software, run the following commands for installing others packages:
```angular2html
pip install -r requirements.txt
```

# Data Preparation
### 3D FRONT dataset
The prepare data for training is in the following link: <a href="https://cuhko365-my.sharepoint.com/:f:/g/personal/115010192_link_cuhk_edu_cn/Eg99g4P1VMVJoZ5fz3lmDkABvj7Gc7yCjq-qBuYNqWjl2w?e=72lix4" target="__blank">training_data</a>.
Download these three .zip files, and unzip them under ./data/3dfront. 
# Code
Codes for single-view object reconstruction is already released.
The codes for 3D object detection and backgrounding reconstruction will be updated in a few days.

## Object Reconstruction
### Training
run the following commands for training:
```angular2html
python main.py --mode train --config ./configs/train_instPIFu.yaml
```
### Testing and Evaluation
The code is still under testing, the weight for testing and evaluation script will be released in a few days