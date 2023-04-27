import trimesh
import numpy as np
import os
import random
import argparse
from tqdm import tqdm

def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()

def get_minmax(vertices, B_MIN, B_MAX):
    bmin_x = min(vertices[:, 0])
    bmin_y = min(vertices[:, 1])
    bmin_z = min(vertices[:, 2])

    bmax_x = max(vertices[:, 0])
    bmax_y = max(vertices[:, 1])
    bmax_z = max(vertices[:, 2])

    B_MIN[0] = bmin_x * 1.1
    B_MIN[1] = bmin_y * 1.1
    B_MIN[2] = bmin_z * 1.1

    B_MAX[0] = bmax_x * 1.1
    B_MAX[1] = bmax_y * 1.1
    B_MAX[2] = bmax_z * 1.1


def run(data_root,target_root):
    num_sample_inout = 30000
    sigma = 5.0  # perturbation standard deviation for positions
    B_MAX = np.array([1.0, 1.0, 1.0])  # now for normalized model
    B_MIN = np.array([-1.0, -1.0, -1.0])

    mc_box_size = 2.6

    random.seed(1991)
    np.random.seed(1991)

    subjects = os.listdir(data_root)
    # subjects = subjects[:5]
    failed_subj = []
    few_inside = []

    for i, subject in tqdm(enumerate(subjects[:])):
        try:
            mesh_path=os.path.join(data_root, subject, 'raw_watertight.obj')
            print(os.path.exists(mesh_path))
            mesh = trimesh.load(mesh_path)
            vertices = np.array(mesh.vertices)
            get_minmax(vertices, B_MIN, B_MAX)
            #print(B_MIN, B_MAX)
            # B_MIN = B_MIN - (B_MAX - B_MIN)*0.2
            # B_MAX = B_MAX + (B_MAX - B_MIN)*0.2
            surface_points, _ = trimesh.sample.sample_surface(mesh, 3 * num_sample_inout)
            # need to adjust 0.01
            sample_points = surface_points + 0.01 * np.random.normal(scale=sigma, size=surface_points.shape)

            # add random points within image space
            b_min=np.amin(B_MIN)
            b_max=np.amax(B_MAX)
            length = b_max - b_min
            scale = length
            random_points = np.random.rand(num_sample_inout * 2, 3) * length + b_min
            np.random.shuffle(sample_points)

            # labeling
            uniform_inside = mesh.contains(random_points)
            uniform_inside_points = random_points[uniform_inside]
            uniform_outside_points = random_points[np.logical_not(uniform_inside)]
            print(uniform_inside.shape)
            nss_inside = mesh.contains(sample_points)
            print(nss_inside.shape)
            nss_inside_points = sample_points[nss_inside]
            nss_outside_points = sample_points[np.logical_not(nss_inside)]

            if not os.path.exists(os.path.join(target_root, subject)):
                os.makedirs(os.path.join(target_root, subject))
            #print(os.path.join(target_root, subject))
            inside_points=np.concatenate([uniform_inside_points,nss_inside_points],axis=0)
            outside_points=np.concatenate([uniform_outside_points,nss_outside_points],axis=0)
            print(inside_points.shape,outside_points.shape)
            save_obj_mesh(os.path.join(target_root, subject, 'inside_points.obj'), inside_points, [])
            save_obj_mesh(os.path.join(target_root, subject, 'outside_points.obj'), outside_points, [])

            print('Finished %d of %d models' % (i + 1, len(subjects)))
            print('failed %d' % len(failed_subj))
            print('inside < 10000 %d' % len(few_inside))
        except:
            failed_subj.append(subject)
            continue

    failed_file = open('./failed_sample.txt', 'w')
    for f in failed_subj:
        failed_file.write(f + '\n')
    failed_file.close()

    few_inside_file = open('./few_inside.txt', 'w')
    for f in few_inside:
        few_inside_file.write(f + '\n')
    few_inside_file.close()

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('sample occupancy of 3D-FUTURE dataset')
    parser.add_argument('--data_root', type=str,
                        help='root path of 3D-FUTURE dataset')
    parser.add_argument('--target_root', type=str, default='train', help='root path where to save the occupancy')
    return parser.parse_args()

if __name__=="__main__":
    args=parse_args()
    data_root=args.data_root
    target_root=args.target_root
    run(data_root,target_root)
