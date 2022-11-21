import os,sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import pickle as p
from PIL import Image
from torchvision import transforms
import random
import numpy as np
from PIL import ImageFile
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms_nocrop = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
])

data_transforms_label = transforms.Compose(
    [
     transforms.ToTensor(),
     ]
)

def q2rot(Q,pos):
    # Extract the values from Q
    x = Q[0]
    y = Q[1]
    z = Q[2]
    w = Q[3]

    x2=x+x
    y2=y+y
    z2=z+z
    xx=x*x2
    xy=x*y2
    xz=x*z2
    yy=y*y2
    yz=y*z2
    zz=z*z2
    wx=w*x2
    wy=w*y2
    wz=w*z2

    # First row of the rotation matrix
    r00 = (1-(yy+zz))
    r01 = xy+wz
    r02 = xz-wy
    r03 = 0

    # Second row of the rotation matrix
    r10 = xy-wz
    r11 = (1-(xx+zz))
    r12 = yz+wx
    r13 = 0

    # Third row of the rotation matrix
    r20 = xz+wy
    r21 = yz-wx
    r22 = 1-(xx+yy)
    r23 = 0

    r30=pos[0]
    r31=pos[1]
    r32=pos[2]
    r33=1

    # 3x3 rotation matrix
    pos = np.array([[r00, r10, r20,r30],
                    [r01, r11, r21,r31],
                    [r02, r12, r22,r32],
                    [r03, r13, r23,r33]])
    return pos

def read_obj_point(obj_path):
    with open(obj_path, 'r') as f:
        content_list = f.readlines()
        point_list = [line.rstrip("\n").lstrip("v ").split(" ") for line in content_list]
        for point in point_list:
            for i in range(3):
                point[i] = float(point[i])
        return np.array(point_list)

class FRONT_bg_dataset(Dataset):
    def __init__(self,config,mode):
        super(FRONT_bg_dataset,self).__init__()
        self.config=config
        self.mode=mode
        self.occ_path=self.config['data']['occ_path']
        self.split_path=os.path.join(self.config['data']['split_path'],mode+'.json')
        with open(self.split_path,'r') as f:
            self.split=json.load(f)
        # self.__load_data()
        # self.render_list=list(self.data.keys())

    def __len__(self):
        return len(self.split)
    #
    # def __load_data(self):
    #     with open(self.data_path,'rb') as f:
    #         self.data=p.load(f)

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def reconstruct_pcd(self, depth, intrinsic):
        valid_Y, valid_X = np.where((depth > 0) & (depth < 10))
        random_ind=np.random.choice(valid_X.shape[0],2500)
        valid_Y=valid_Y[random_ind]
        valid_X=valid_X[random_ind]
        unprojected_Y = valid_Y * depth[valid_Y, valid_X]
        unprojected_X = valid_X * depth[valid_Y, valid_X]
        unprojected_Z = depth[valid_Y, valid_X]
        point_cloud_xyz = np.concatenate(
            [unprojected_X[:, np.newaxis], unprojected_Y[:, np.newaxis], unprojected_Z[:, np.newaxis]], axis=1)
        intrinsic_inv = np.linalg.inv(intrinsic[0:3, 0:3])
        point_cloud_xyz = np.dot(intrinsic_inv, point_cloud_xyz.T).T
        return point_cloud_xyz,unprojected_Z

    def __getitem__(self,index):
        success_flag = False
        while success_flag == False:
            render_id,scene_id=self.split[index]['render_id'],self.split[index]['scene_id']
            index = np.random.randint(0, self.__len__())
            #print(render_id)
            prepare_data_path = os.path.join(self.config['data']['data_path'], self.mode, render_id + ".pkl")
            if os.path.exists(prepare_data_path)==False:
                #print(prepare_data_path,"does not exist")
                continue
            with open(prepare_data_path,'rb') as f:
                prepare_data=p.load(f)
            intrinsic=prepare_data['camera']['K'].copy()
            intrinsic=np.abs(intrinsic)
            intrinsic_matrix=np.zeros((4,4))
            intrinsic_matrix[0:3,0:3]=intrinsic
            intrinsic_matrix[3,3]=1

            inside_path=os.path.join(self.occ_path,render_id,"outside_points.obj")
            outside_path=os.path.join(self.occ_path,render_id,"inside_points.obj")
            if os.path.isfile(inside_path)==False or os.path.isfile(outside_path)==False:
                print(inside_path)
                continue
            inside_samples=read_obj_point(inside_path)
            outside_samples=read_obj_point(outside_path)
            if inside_samples.shape[0]<2500 or outside_samples.shape[0]<2500:
                continue
            inside_random_ind=np.random.choice(inside_samples.shape[0],2500,replace=False)
            outside_random_ind=np.random.choice(outside_samples.shape[0],2500,replace=False)
            sample_points=np.concatenate([inside_samples[inside_random_ind],outside_samples[outside_random_ind]],axis=0)

            label=np.zeros(sample_points.shape[0])
            label[0:2500]=1
            label[2500:5000]=0
            success_flag = True
        scale_intrinsic = intrinsic_matrix

        image=Image.fromarray(prepare_data['rgb_img'])
        #print(scale_intrinsic, image.size)
        width, height = image.size
        image=image.resize((self.config["data"]["image_width"],self.config["data"]["image_height"]))
        input_height,input_width=self.config["data"]["image_height"],self.config["data"]["image_width"]

        scale_intrinsic[0] = scale_intrinsic[0] / (width / input_width) /2  # it has extra divided by 2 since the loaded image is already downsampled by 2
        scale_intrinsic[1] = scale_intrinsic[1] / (height / input_height) /2

        #print(scale_intrinsic,image.size)

        rot_matrix = np.array([[1, 0, 0],
                               [0, 1, 0]])
        M=np.array([[1,0,0],[0,1,0],[0,0,1]])
        if self.config['data']['use_aug'] and self.mode=="train":
            '''add rotation to the image'''
            random_angle = (random.random() - 0.5) * 2 * self.config['data']['rotate_degree']
            rot_matrix = cv2.getRotationMatrix2D(center=(input_width//2, input_height//2), angle=random_angle, scale=1)
            image = np.asarray(image, dtype=np.float32)/255.0
            image=cv2.warpAffine(image,rot_matrix,borderMode=cv2.BORDER_REPLICATE,dsize=(input_width,input_height),flags=cv2.INTER_LINEAR)

            do_flip = random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                sample_points[:,0]=-1*sample_points[:,0].copy()
                rot_matrix = cv2.getRotationMatrix2D(center=(input_width//2, input_height//2), angle=-random_angle, scale=1)

            do_augment = random.random()
            if do_augment > 0.5:
                image = self.augment_image(image)

            '''adding some yaw and pitch rotation to the sample points'''
            up_angle=-random.random()*20+15
            up_angle = up_angle / 180 * np.pi
            cs = np.cos(up_angle)
            ss = np.sin(up_angle)
            M_p = np.array([[1, 0, 0],
                          [0, cs, -ss],
                          [0, ss, cs]])

            roll_angle=-random.random()*20+10
            roll_angle = roll_angle/180*np.pi
            cr=np.cos(roll_angle)
            sr=np.sin(roll_angle)
            M_r=np.array([[cr,-sr,0],
                          [sr,cr,0],
                          [0,0,1]])
            M=np.dot(M_p,M_r)

        else:
            image = np.asarray(image, dtype=np.float32) / 255.0
        image=data_transforms_nocrop(image)

        return_dict = {
            "image": image,
            "jid": scene_id,
            "taskid":render_id,
            "intrinsic": scale_intrinsic,
            "rot_matrix": rot_matrix,
            "M":M,
            "samples":sample_points,
            "inside_class":label}
        return return_dict

def worker_init_fn(worker_id):
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)
def Front3D_bg_Dataloader(cfg,mode):
    dataset=FRONT_bg_dataset(cfg,mode)
    dataloader = DataLoader(dataset=dataset,
                            num_workers=cfg['data']['num_workers'],
                            batch_size=cfg['data']['batch_size'],
                            shuffle=(mode == 'train'),
                            worker_init_fn=worker_init_fn, pin_memory=True
                            )
    return dataloader


if __name__=="__main__":
    from configs.config_utils import CONFIG
    config=CONFIG("./configs/train_bg_PIFu.yaml").config
    dataset=FRONT_bg_dataset(config,"train")
    for i in range(10):
        data_batch=dataset.__getitem__(i)
        for key in data_batch:
            if not isinstance(data_batch[key],str):
                print(data_batch[key].shape)