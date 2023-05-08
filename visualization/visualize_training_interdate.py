import open3d
import pickle as p
import numpy as np
import open3d as o3d
import cv2

data_path=r"D://fsdownload//train_dump_dict_38000.pkl"
with open(data_path,'rb') as f:
    data_content=p.load(f)

sample_points=data_content["sample_points"]
print(np.mean(sample_points,axis=0))

ndf=data_content["pred_class"]
print(np.min(ndf))
color=np.zeros((ndf.shape[0],3)).astype(np.float32)
color[ndf<0.5]=np.array([1.0,0,0])
color[ndf>=0.5]=np.array([0,1.0,0])
print(color.shape)

sample_pcd=o3d.geometry.PointCloud()
sample_pcd.points=o3d.utility.Vector3dVector(sample_points[:,0:3])
sample_pcd.colors=o3d.utility.Vector3dVector(color[:,0:3])
#print(np.array(sample_pcd.colors))
coor_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()

rgb=data_content["image"].transpose(1,2,0)*np.array([[0.229,0.224,0.225]])+np.array([[0.485, 0.456, 0.406]])
cv2.imshow('1',rgb)
vis=o3d.visualization.Visualizer()
vis.create_window()
#vis.add_geometry(pcd_whole)
vis.add_geometry(coor_frame)
vis.add_geometry(sample_pcd)
vis.run()