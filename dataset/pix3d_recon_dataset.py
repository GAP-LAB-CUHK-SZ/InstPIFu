import os
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
from PIL import Image
from pathlib import Path
import json
import random
#from net_utils.bins import *
import trimesh
from tqdm import tqdm
import numpy as np
import pickle as p
from torch.utils.data import DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_rot_from_yaw(yaw):
    cy=np.cos(yaw)
    sy=np.sin(yaw)
    rot=np.array([[cy,0,sy],
                     [0,1,0],
                     [-sy,0,cy]])
    return rot

# category_label_mapping = {"table":0,
#                           "sofa": 1,
#                           "misc": 2,
#                           "bookcase":3,
#                           "chair":4,
#                           "tool":5,
#                           "bed":6,
#                           "desk":7,
#                           "wardrobe":8
#                           }

category_label_mapping = {"table":0,
                          "sofa": 1,
                          "misc": 2,
                          "tool":3,
                          "chair":4,
                          "bookcase":5,
                          "bed":6,
                          "desk":7,
                          "wardrobe":8
                          }

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((500, 500)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_mask_resize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((280, 280)),
])
data_transforms_noresize=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_crop=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((280, 280)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

pil2tensor = transforms.ToTensor()

def read_obj_point(obj_path):
    with open(obj_path, 'r') as f:
        content_list = f.readlines()
        point_list = [line.rstrip("\n").lstrip("v ").split(" ") for line in content_list]
        for point in point_list:
            for i in range(3):
                point[i] = float(point[i])
        return np.array(point_list)

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh

def R_from_yaw_pitch_roll(yaw, pitch, roll):
    '''
    get rotation matrix from predicted camera yaw, pitch, roll angles.
    :param yaw: batch_size x 1 tensor
    :param pitch: batch_size x 1 tensor
    :param roll: batch_size x 1 tensor
    :return: camera rotation matrix
    '''
    Rp = np.zeros((3, 3))
    Ry = np.zeros((3,3))
    Rr = np.zeros((3,3))
    cp=np.cos(pitch)
    sp=np.sin(pitch)
    cy=np.cos(yaw)
    sy=np.sin(yaw)
    cr=np.cos(roll)
    sr=np.sin(roll)
    Rp[0,0]=1
    Rp[1,1]=cp
    Rp[1,2]=-sp
    Rp[2,1]=sp
    Rp[2,2]=cp

    Ry[0,0]=cy
    Ry[0,2]=sy
    Ry[1,1]=1
    Ry[2,0]=-sy
    Ry[2,2]=cy

    Rr[0,0]=cr
    Rr[0,1]=-sr
    Rr[1,0]=sr
    Rr[1,1]=cr
    Rr[2,2]=1

    R=np.dot(np.dot(Rr,Rp),Ry)
    return R

class Pix3d_Recon_Dataset(Dataset):
    def __init__(self,config,mode):
        super(Pix3d_Recon_Dataset,self).__init__()
        self.config=config
        self.mode=mode
        self.split_path = os.path.join(self.config['data']['split_dir'], mode+".json")
        with open(self.split_path,'r') as f:
            self.split=json.load(f)
        #self.split = self.split[0:100]
        self.__load_data()
    def __len__(self):
        return len(self.split)
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

    def __load_data(self):
        self.prepare_data_dict={}
        self.nss_inside_data_dict={}
        self.nss_outside_data_dict = {}
        self.uni_inside_data_dict={}
        self.uni_outside_data_dict={}
        self.image_data_dict={}
        self.mask_data_dict={}
        self.mesh_data_dict={}
        for data_path in tqdm(self.split):
            taskid=data_path.split("/")[-1].split(".")[0]
            if taskid not in self.prepare_data_dict:
                with open(data_path,'rb') as f:
                    sequence=pickle.load(f)
                nss_occ_inside_path = os.path.join(self.config['data']['base_dir'], sequence['occ_inside']).replace(
                    "inside_points", "nss_inside_points")
                nss_occ_outside_path = os.path.join(self.config['data']['base_dir'], sequence['occ_outside']).replace(
                    "outside_points", "nss_outside_points")
                uni_occ_inside_path = os.path.join(self.config['data']['base_dir'], sequence['occ_inside']).replace(
                    "inside_points", "uniform_inside_points")
                uni_occ_outside_path = os.path.join(self.config['data']['base_dir'], sequence['occ_outside']).replace(
                    "outside_points", "uniform_outside_points")
                if os.path.isfile(nss_occ_inside_path) == False or os.path.isfile(nss_occ_outside_path) == False:
                    print(taskid,"is invalid")
                    continue

                self.prepare_data_dict[taskid] = sequence
                self.nss_inside_data_dict[taskid]=read_obj_point(nss_occ_inside_path)
                self.nss_outside_data_dict[taskid]=read_obj_point(nss_occ_outside_path)
                self.uni_inside_data_dict[taskid] = read_obj_point(uni_occ_inside_path)
                self.uni_outside_data_dict[taskid] = read_obj_point(uni_occ_outside_path)

                image_path = sequence['img']
                mesh_path = os.path.join(self.config['data']['base_dir'], sequence['model'])
                img_path = os.path.join(self.config['data']['base_dir'], image_path)
                img_id = img_path.split("/")[-1].split(".")[0]
                mask_path = Path(os.path.join(self.config['data']['base_dir'], image_path.replace("img", "mask")))
                mask_path_wosuffix = mask_path.with_suffix('')
                mask_path = mask_path_wosuffix
                mask_path = mask_path.with_suffix(".png")

                img = Image.open(img_path)
                mask = Image.open(mask_path)

                self.image_data_dict[taskid]=img
                self.mask_data_dict[taskid]=mask

                mesh=as_mesh(trimesh.load(mesh_path))
                self.mesh_data_dict[taskid]=mesh

    def __getitem__(self,index):
        success_flag=False
        while success_flag==False:
            data_path=self.split[index]
            index=np.random.randint(0,self.__len__())
            taskid = data_path.split("/")[-1].split(".")[0]
            if taskid not in self.prepare_data_dict:
                print("cannot find data", taskid)
                continue
            data=self.prepare_data_dict[taskid]
            #print(data)
            image_path=data['img']
            mesh_path=os.path.join(self.config['data']['base_dir'],data['model'])
            img_path=os.path.join(self.config['data']['base_dir'],image_path)
            img_id=img_path.split("/")[-1].split(".")[0]
            mask_path=Path(os.path.join(self.config['data']['base_dir'],image_path.replace("img","mask")))
            mask_path_wosuffix=mask_path.with_suffix('')
            mask_path=mask_path_wosuffix
            mask_path=mask_path.with_suffix(".png")

            img=self.image_data_dict[taskid]
            if self.config['data']['use_instance_mask']:
                mask=self.mask_data_dict[taskid]
            width,height=img.size

            nss_inside_points=self.nss_inside_data_dict[taskid]
            nss_outside_points=self.nss_outside_data_dict[taskid]
            uni_inside_points=self.uni_inside_data_dict[taskid]
            uni_outside_points=self.uni_outside_data_dict[taskid]
            nss_inside_random_ind=np.random.choice(nss_inside_points.shape[0],1024,replace=False)
            nss_outside_random_ind=np.random.choice(nss_outside_points.shape[0],1024,replace=False)
            uni_inside_random_ind=np.random.choice(uni_inside_points.shape[0],1024,replace=False)
            uni_outside_random_ind = np.random.choice(uni_outside_points.shape[0], 1024, replace=False)
            samples=np.concatenate([nss_inside_points[nss_inside_random_ind],nss_outside_points[nss_outside_random_ind],
                                    uni_inside_points[uni_inside_random_ind],uni_outside_points[uni_outside_random_ind]],axis=0)
            inside_class = np.zeros(samples.shape[0])
            inside_class[0:1024] = 1
            inside_class[2048:3096]=1
            bdb2D = np.array(data['bbox'])
            #print(bdb2D)
            if self.config['data']['use_aug'] and self.mode == "train":
                random_crop_bdb=bdb2D.copy()
                random_crop_bdb[0] += np.random.rand() * 0.05 * (bdb2D[2] - bdb2D[0])
                random_crop_bdb[1] += np.random.rand() * 0.05 * (bdb2D[3] - bdb2D[1])
                random_crop_bdb[2] = random_crop_bdb[0] + 0.95 * (bdb2D[2] - bdb2D[0])
                random_crop_bdb[3] = random_crop_bdb[1] + 0.95 * (bdb2D[3] - bdb2D[1])
                bdb2D=random_crop_bdb.copy()

            rot_matrix=data['rot_mat']
            #print(rot_matrix)
            rot_matrix=np.reshape(np.array(rot_matrix),(3,3))
            trans_mat=data['trans_mat']
            sensor_width=32
            sensor_height=32/width*height
            f=data['focal_length']*width/sensor_width
            org_K=np.array([[f,0,width/2],
                            [0,f,height/2],
                            [0,0,1]])
            '''adjust intrinsic considering image padding'''
            max_length = max(width, height)
            intrinsic=np.array([[f,0,max_length/2],
                                [0,f,max_length/2],
                                [0,0,1]])
            points_in_wrd = np.dot(samples, rot_matrix.T) + np.array(trans_mat)
            points_in_wrd[:,0:2]=-points_in_wrd[:,0:2]
            points_in_cam = np.dot(points_in_wrd, org_K.T)
            x_coor = points_in_cam[:, 0] / points_in_cam[:, 2]
            y_coor = points_in_cam[:, 1] / points_in_cam[:, 2]
            x_coor = x_coor - (bdb2D[0] + bdb2D[2]) / 2
            y_coor = y_coor - (bdb2D[1] + bdb2D[3]) / 2
            x_coor = x_coor / (bdb2D[2] - bdb2D[0]) * 2
            y_coor = y_coor / (bdb2D[3] - bdb2D[1]) * 2
            bdb_coor=np.concatenate([x_coor[:,np.newaxis],y_coor[:,np.newaxis]],axis=1)

            '''consider padding'''
            bdb2D_pad=bdb2D.copy()
            bdb2D_pad[0]+=max(0,(height-width)/2)
            bdb2D_pad[1]+=max(0,(width-height)/2)
            bdb2D_pad[2]+=max(0,(height-width)/2)
            bdb2D_pad[3]+=max(0,(width-height)/2)
            #print(bdb2D_pad,bdb2D)

            bdb_x = np.linspace(bdb2D_pad[0], bdb2D_pad[2], 64)
            bdb_y = np.linspace(bdb2D_pad[1], bdb2D_pad[3], 64)
            bdb_X, bdb_Y = np.meshgrid(bdb_x, bdb_y)
            bdb_X = (bdb_X - max_length/2) / max_length*2
            bdb_Y = (bdb_Y - max_length/2) / max_length*2
            bdb_grid = np.concatenate([bdb_X[:, :, np.newaxis], bdb_Y[:, :, np.newaxis]], axis=-1)

            mesh=self.mesh_data_dict[taskid]
            vertices=mesh.vertices
            x_max=np.max(vertices[:,0])
            x_min=np.min(vertices[:,0])
            y_max=np.max(vertices[:,1])
            y_min=np.min(vertices[:,1])
            z_max=np.max(vertices[:,2])
            z_min=np.min(vertices[:,2])
            obj_cam_center=np.array(trans_mat)
            obj_cam_center[0:2]=-obj_cam_center[0:2]
            bbox_size=np.array([x_max-x_min,y_max-y_min,z_max-z_min])
            input_samples=samples.copy()
            input_samples=input_samples
            obj_cam_center=obj_cam_center
            input_samples=input_samples/np.array([x_max-x_min,y_max-y_min,z_max-z_min])*2 #-1 ~ 1
            yaw_rot = get_rot_from_yaw(np.pi)
            input_samples=np.dot(input_samples,yaw_rot.T)
            #input_samples[:,2]=-input_samples[:,2] #invert the z axis,since pix 3d is -z axis as front
            rot_matrix=np.dot(rot_matrix,yaw_rot)
            reverse_canonical_samples=np.dot(input_samples.copy(),rot_matrix.T)
            z_feat=reverse_canonical_samples[:,2:3].copy()

            patch=img.crop((bdb2D[0],bdb2D[1],bdb2D[2],bdb2D[3]))
            patch=np.asarray(patch)/255.0
            image = np.asarray(img) / 255.0
            if len(image.shape)==2:
                image=np.concatenate([image[:,:,np.newaxis],image[:,:,np.newaxis],image[:,:,np.newaxis]],axis=2)
                patch=np.concatenate([patch[:,:,np.newaxis],patch[:,:,np.newaxis],patch[:,:,np.newaxis]],axis=2)
            elif image.shape[2]==2:
                continue
            image=image[:,:,0:3]
            patch = patch[:, :, 0:3]
            '''
            pad the image
            '''
            max_length = max(width, height)
            pad_img = np.zeros([max_length, max_length, 3])

            if width >= height:
                margin=int((width-height)/2)
                #print(image.shape,pad_img.shape)
                pad_img[margin:margin+height,:]=image
            else:
                margin=int((height-width)/2)
                pad_img[:,margin:margin+width]=image
            if self.config['data']['use_instance_mask']:
                mask=mask.crop((bdb2D[0],bdb2D[1],bdb2D[2],bdb2D[3]))
                mask=np.asarray(mask)/255.0
                mask=data_transforms_mask_resize(mask)
            if self.config['data']['use_aug'] and self.mode=="train":
                image = self.augment_image(image)
                pad_img=self.augment_image(pad_img)
            patch=data_transforms_crop(patch).float()
            pad_img=data_transforms(pad_img).float()

            cls_codes=np.zeros([9])
            category=data['category']
            cls_codes[category_label_mapping[category]]=1
            #print(category)
            success_flag=True
        #taskid=data['img'].split("/")[-2]+data['img'].split("/")[-1].split(".")[0]
        data_dict={
            "whole_image":pad_img.float(),"image":patch.float(),"patch":patch.float(),
            "samples":input_samples[:,0:3].astype(np.float32),
            "inside_class":inside_class.astype(np.float32),"bdb2D_pos":bdb2D_pad.astype(np.float32),
            "sequence_id":taskid,"z_feat":z_feat,"K":intrinsic,"rot_matrix":rot_matrix,
            "jid":data['occ_inside'],"img_coor":bdb_coor.astype(np.float32),"taskid":taskid,"obj_id":"0",
            "obj_cam_center":obj_cam_center,"cls_codes":cls_codes.astype(np.float32),
            "bdb_grid":bdb_grid.astype(np.float32),"bbox_size":bbox_size
        }
        if self.config['data']['use_instance_mask']:
            data_dict['mask']=mask
        return data_dict

def worker_init_fn(worker_id):
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

def Pix3D_Recon_dataloader(config, mode='train'):
    dataloader = DataLoader(dataset=Pix3d_Recon_Dataset(config, mode),
                            num_workers=config['data']['num_workers'],
                            batch_size=config['data']['batch_size'],
                            shuffle=(mode == 'train'),
                            worker_init_fn=worker_init_fn, pin_memory=True
                            )
    return dataloader

if __name__=="__main__":
    from configs.config_utils import CONFIG

    cfg = CONFIG("./configs/train_instPIFu_onpix3d.yaml").config
    dataset=Pix3d_Recon_Dataset(cfg,"test")
    for i in range(0,10):
        print("[%d/%d]"%(i,dataset.__len__()))
        item=dataset.__getitem__(i)
        with open("/home/haolin/test_batch_%d_pix3d.pkl"%(i),'wb') as f:
            p.dump(item,f)

