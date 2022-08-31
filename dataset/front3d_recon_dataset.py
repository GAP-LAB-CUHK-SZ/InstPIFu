# Dataloader of InstPIFu.
# author: LiuHaolin
# date: Aug, 2020
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision import transforms
import pickle
from PIL import Image
import numpy as np
import json
import random
from net_utils.bins import *
from tqdm import tqdm

category_label_mapping = {"table": 0,
                          "sofa": 1,
                          "cabinet": 2,
                          "night_stand": 3,
                          "chair": 4,
                          "bookshelf": 5,
                          "bed": 6,
                          "desk": 7,
                          "dresser": 8
                          }

def read_obj_point(obj_path):
    with open(obj_path, 'r') as f:
        content_list = f.readlines()
        point_list = [line.rstrip("\n").lstrip("v ").split(" ") for line in content_list]
        for point in point_list:
            for i in range(3):
                point[i] = float(point[i])
        return np.array(point_list)

def R_from_yaw_pitch_roll(yaw, pitch, roll):
    '''
    get rotation matrix from predicted camera yaw, pitch, roll angles.
    :param yaw: batch_size x 1 tensor
    :param pitch: batch_size x 1 tensor
    :param roll: batch_size x 1 tensor
    :return: camera rotation matrix
    '''
    Rp = np.zeros((3, 3))
    Ry = np.zeros((3, 3))
    Rr = np.zeros((3, 3))
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cr = np.cos(roll)
    sr = np.sin(roll)
    Rp[0, 0] = 1
    Rp[1, 1] = cp
    Rp[1, 2] = -sp
    Rp[2, 1] = sp
    Rp[2, 2] = cp

    Ry[0, 0] = cy
    Ry[0, 2] = sy
    Ry[1, 1] = 1
    Ry[2, 0] = -sy
    Ry[2, 2] = cy

    Rr[0, 0] = cr
    Rr[0, 1] = -sr
    Rr[1, 0] = sr
    Rr[1, 1] = cr
    Rr[2, 2] = 1

    R = np.dot(np.dot(Rr, Rp), Ry)
    return R

def get_centroid_from_proj(centroid_depth, proj_centroid, K):
    x_temp = (proj_centroid[0] - K[0, 2]) / K[0, 0]
    y_temp = (proj_centroid[1] - K[1, 2]) / K[1, 1]
    z_temp = 1
    ratio = centroid_depth / np.sqrt(x_temp ** 2 + y_temp ** 2 + z_temp ** 2)
    x_cam = x_temp * ratio
    y_cam = y_temp * ratio
    z_cam = z_temp * ratio
    p = np.stack([x_cam, y_cam, z_cam])
    return p


data_transforms_patch = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_mask = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])

class Front3D_Recon_Dataset(Dataset):
    def __init__(self, config, mode):
        super(Front3D_Recon_Dataset, self).__init__()
        self.mode = mode
        self.config = config
        if mode=="train":
            classname = self.config['data']['class_name']
        else:
            classname = self.config['data']['test_class_name']
        self.use_pred_pose = self.config['data']['use_pred_pose']
        if isinstance(classname, list):
            self.multi_class = True
            self.split = []
            for class_name in classname:
                self.split_path = os.path.join(self.config['data']['split_dir'], mode, class_name + ".json")
                with open(self.split_path, 'r') as f:
                    self.split += json.load(f)
        else:
            self.multi_class = False
            self.split = []
            self.split_path = os.path.join(self.config['data']['split_dir'], mode, classname + ".json")
            with open(self.split_path, 'r') as f:
                self.split = json.load(f)
        #print(len(self.split))
        #print(self.split)
        '''only test 2000 samples'''
        if mode=="test":
            self.split=self.split[0:2000]

        for i in range(len(self.split)):
            filename=self.split[i][0]
            if ".pkl" in filename:
                taskid=filename.rstrip(".pkl")
            self.split[i][0]=taskid

        if self.config['data']['load_dynamic'] == False:
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
        self.prepare_data_dict = {}
        self.occ_inside_data_dict = {}
        self.occ_outside_data_dict = {}
        self.instance_mask_dict = {}
        jid_list=[]
        print("loading prepare data")
        for (taskid, objid) in tqdm(self.split):
            if taskid not in self.prepare_data_dict:
                prepare_data_path = os.path.join(self.config['data']['data_path'], self.mode, taskid + ".pkl")
                with open(prepare_data_path, 'rb') as f:
                    sequence = pickle.load(f)
                self.prepare_data_dict[taskid] = sequence
            boxes = self.prepare_data_dict[taskid]['boxes']
            object_ind = objid
            jid = boxes['jid'][object_ind]
            jid_list.append(jid)
        jid_list=list(set(jid_list))
        print("loading object occupancy data")
        for jid in tqdm(jid_list):
            if jid not in self.occ_inside_data_dict:
                inside_occ_path = os.path.join(self.config['data']['occ_path'], jid, "inside_points.obj")
                outside_occ_path = os.path.join(self.config['data']['occ_path'], jid, 'outside_points.obj')
                if os.path.isfile(inside_occ_path) and os.path.isfile(outside_occ_path):
                    inside_sample = read_obj_point(inside_occ_path)
                    outside_sample = read_obj_point(outside_occ_path)
                    self.occ_outside_data_dict[jid] = outside_sample
                    self.occ_inside_data_dict[jid] = inside_sample
                else:
                    print(jid, inside_occ_path)
                    continue

    def __getitem__(self, index):
        success_flag = False
        while success_flag == False:
            taskid, objid = self.split[index]
            index = np.random.randint(0, self.__len__())
            '''load the data dynamically or store them in the memory firstly'''
            if self.config['data']['load_dynamic'] == True:
                prepare_data_path = os.path.join(self.config['data']['data_path'], self.mode, taskid + ".pkl")
                with open(prepare_data_path, 'rb') as f:
                    sequence = pickle.load(f)
            else:
                sequence = self.prepare_data_dict[taskid]
            image = Image.fromarray(sequence['rgb_img'])
            width, height = image.size
            layout = sequence["layout"]
            boxes = sequence['boxes']
            object_ind = objid
            jid = boxes['jid'][object_ind]
            if self.config['data']['load_dynamic'] == True:
                inside_occ_path = os.path.join(self.config['data']['occ_path'], jid, "inside_points.obj")
                outside_occ_path = os.path.join(self.config['data']['occ_path'], jid, 'outside_points.obj')
                '''if the occupancy file does not exist, skip'''
                if os.path.isfile(inside_occ_path) and os.path.isfile(outside_occ_path):
                    inside_sample = read_obj_point(inside_occ_path)
                    outside_sample = read_obj_point(outside_occ_path)
                else:
                    print(jid, inside_occ_path)
                    continue
            else:
                if jid not in self.occ_inside_data_dict:
                    continue
                inside_sample = self.occ_inside_data_dict[jid]
                outside_sample = self.occ_outside_data_dict[jid]
            inside_random_ind = np.random.choice(inside_sample.shape[0], 2048, replace=False)
            outside_random_ind = np.random.choice(outside_sample.shape[0], 2048, replace=False)
            samples = np.concatenate([inside_sample[inside_random_ind], outside_sample[outside_random_ind]], axis=0)
            inside_class = np.zeros(samples.shape[0])
            inside_class[0:2048] = 1
            inside_class[2048:] = 0
            scale = boxes['scale'][object_ind]
            samples = samples.copy() * scale
            # print(np.min(samples[0:2048],axis=0),np.max(samples[0:2048],axis=0))
            size_cls = boxes['size_cls'][object_ind]
            # print(size_cls)
            bbox_size = (boxes['size_reg'][object_ind] + 1) * bin['avg_size'][size_cls]
            if np.where(bbox_size == 0)[0].shape[0] > 0:
                print("bbox_size has zero", bbox_size)
                continue
            # canonical_samples = samples/bbox_size*2 #-0.5~0.5
            # canonical_samples[:,1]=canonical_samples[:,1]-1
            homo_points = np.concatenate([samples.copy(), np.ones((samples.shape[0], 1))], axis=1)
            tran_matrix = boxes["tran_matrix"][object_ind]
            tran_matrix[1, 3] = 0
            wrd2cam_matrix = sequence['camera']['wrd2cam_matrix']
            org_K = sequence['camera']['K'].copy()

            '''intrinsic needs to be scaled by 2 since the input image is downsampled'''
            K = org_K.copy()
            K[0] = K[0] / 2
            K[1] = K[1] / 2
            # rot_matrix=np.dot(wrd2cam_matrix[0:3,0:3], tran_matrix)
            cam_samples = np.dot(homo_points, np.dot(wrd2cam_matrix, tran_matrix).T)  # y up coordinate
            cam_samples[:, 0:2] = -cam_samples[:, 0:2]

            obj_cam_center = boxes['cam_center'][object_ind]
            '''convert the input samples to cannonical coordinate'''
            input_samples = cam_samples[:,
                            0:3] - obj_cam_center  # this is now in camera coordinate,origin at (0,0,0)
            rot_matrix = np.dot(wrd2cam_matrix[0:3, 0:3], tran_matrix[0:3, 0:3])
            '''inference using predicted pose'''
            if self.use_pred_pose:
                pred_result_path = os.path.join(self.config['data']['pred_pose_path'], "%s.pkl" % (taskid))
                with open(pred_result_path, 'rb') as f:
                    pred_result = pickle.load(f)
                pitch = pred_result["layout"]['pitch']
                roll = pred_result["layout"]["roll"]
                bboxes = pred_result["bboxes"][object_ind]
                yaw = bboxes["yaw"]
                gt_pitch = layout['pitch']
                gt_roll = layout['roll']
                gt_yaw = boxes['yaw'][object_ind]
                rot_matrix = R_from_yaw_pitch_roll(-yaw, pitch, roll)
                project_center = bboxes['project_center']
                centroid_depth = bboxes['centroid_depth']
                obj_cam_center = get_centroid_from_proj(centroid_depth, project_center, org_K)

            inv_rot = np.linalg.inv(rot_matrix)
            canonical_samples = input_samples[:, 0:3].copy()
            canonical_samples[:, 0:2] = -canonical_samples[:, 0:2]  # y up coordiante
            canonical_samples = np.dot(canonical_samples[:, 0:3], inv_rot.T)
            canonical_samples = canonical_samples / bbox_size * 2

            '''get relative depth of the sample poitns'''
            rot_canonical_samples = np.dot(canonical_samples[:, 0:3].copy(), rot_matrix.T)
            z_feat = rot_canonical_samples[:, 2:3]

            '''compute samples projection on image'''
            input_samples = canonical_samples[:, 0:3].copy()
            img_samples = np.dot(cam_samples[:, 0:3], K.T)
            x_coor = img_samples[:, 0] / img_samples[:, 2]
            y_coor = img_samples[:, 1] / img_samples[:, 2]
            x_coor = ((x_coor - width / 2) / width) * 2
            y_coor = ((y_coor - height / 2) / height) * 2
            img_coor = np.concatenate([x_coor[:, np.newaxis], y_coor[:, np.newaxis], img_samples[:, 2:3]], axis=1)

            '''2d bounding box need to be scaled by 2 since image is downsampled by 2'''
            bdb = boxes['bdb2D_pos'][object_ind]/2
            # print(bdb)
            '''add noise to gt 2d bounding box'''
            if self.config['data']['use_aug'] and self.mode == "train":
                random_crop_bdb = np.array(bdb)
                random_crop_bdb[0] += np.random.rand() * 0.05 * (bdb[2] - bdb[0])
                random_crop_bdb[1] += np.random.rand() * 0.05 * (bdb[3] - bdb[1])
                random_crop_bdb[2] = random_crop_bdb[0] + 0.95 * (bdb[2] - bdb[0])
                random_crop_bdb[3] = random_crop_bdb[1] + 0.95 * (bdb[3] - bdb[1])
                bdb = random_crop_bdb

            '''sampling grid construction during the RoI align operation'''
            bdb_x = np.linspace(bdb[0], bdb[2], 64)
            bdb_y = np.linspace(bdb[1], bdb[3], 64)
            bdb_X, bdb_Y = np.meshgrid(bdb_x, bdb_y)
            bdb_X = (bdb_X - width/2) / width*2 #-1 ~ 1
            bdb_Y = (bdb_Y - height/2) / height*2 #-1 ~ 1
            bdb_grid = np.concatenate([bdb_X[:, :, np.newaxis], bdb_Y[:, :, np.newaxis]], axis=-1)

            '''compute sample coordinate in the bounding box'''
            bdb_xcoor = img_samples[:, 0] / img_samples[:, 2]
            bdb_ycoor = img_samples[:, 1] / img_samples[:, 2]
            bdb_xcoor = bdb_xcoor - (bdb[0] + bdb[2]) / 2
            bdb_ycoor = bdb_ycoor - (bdb[1] + bdb[3]) / 2
            bdb_xcoor = bdb_xcoor / (bdb[2] - bdb[0]) * 2  # -1 ~ 1
            bdb_ycoor = bdb_ycoor / (bdb[3] - bdb[1]) * 2  # -1 ~ 1
            bdb_coor = np.concatenate([bdb_xcoor[:, np.newaxis], bdb_ycoor[:, np.newaxis]], axis=1)

            '''crop the object'''
            patch = image.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
            # image = image.resize(size=(width // 2, height // 2))
            image = np.asarray(image) / 255.0
            if self.config['data']['use_aug'] and self.mode == "train":
                image = self.augment_image(image)
            image = data_transforms_image(image)

            '''spatial-guided supervision GT'''
            if self.config['data']['use_instance_mask']:
                instance_mask_path = os.path.join(self.config['data']['mask_path'], "%s_%s.png" % (taskid, objid))
                instance_mask = Image.open(instance_mask_path)
                crop_mask = instance_mask.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
                instance_mask = np.asarray(instance_mask) / 255.0
                instance_mask = instance_mask[None, :, :]  # 1,H,W
                crop_mask = np.asarray(crop_mask) / 255.0
                crop_mask = crop_mask[:, :]  # H,W
                crop_mask = data_transforms_mask(crop_mask)

            cls_codes = np.zeros([9])
            cls_codes[boxes['size_cls'][object_ind]] = 1

            patch = np.asarray(patch) / 255.0
            patch = patch[:, :, 0:3]
            if self.config['data']['use_aug'] and self.mode == "train":
                patch = self.augment_image(patch)
            patch = data_transforms_patch(patch).float()
            # print(patch.shape,crop_mask.shape)
            data_dict = {"whole_image": image.float(),"image":patch.float(), "patch": patch.float(),
                         "samples": input_samples[:, 0:3].astype(np.float32),
                         "inside_class": inside_class.astype(np.float32), 'bdb2D_pos': bdb.astype(np.float32),
                         "sequence_id": sequence["sequence_id"], "z_feat": z_feat, "K": K, "rot_matrix": rot_matrix,
                         "jid": jid, "bdb_grid": bdb_grid, "taskid": taskid, "obj_id": str(object_ind),
                         "obj_cam_center": obj_cam_center, "cls_codes": cls_codes.astype(np.float32),
                         "img_coor": bdb_coor.astype(np.float32), "bbox_size": bbox_size}
            if self.config['data']['use_instance_mask']:
                data_dict["mask"] = crop_mask
            success_flag = True

        return data_dict

def worker_init_fn(worker_id):
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

def Front3D_Recon_dataloader(config, mode='train'):
    dataloader = DataLoader(dataset=Front3D_Recon_Dataset(config, mode),
                            num_workers=config['data']['num_workers'],
                            batch_size=config['data']['batch_size'],
                            shuffle=(mode == 'train'),
                            worker_init_fn=worker_init_fn, pin_memory=True
                            )
    return dataloader

if __name__=="__main__":
    from configs.config_utils import CONFIG
    cfg=CONFIG("./configs/train_instPIFu.yaml").config
    #=Front3D_Recon_dataloader(cfg,'train')
    dataset=Front3D_Recon_Dataset(cfg,"train")
    for i in range(10):
        item = dataset.__getitem__(i)
        with open("/home/haolin/traindata_front3d_%d.pkl" % (i), 'wb') as f:
            pickle.dump(item, f)