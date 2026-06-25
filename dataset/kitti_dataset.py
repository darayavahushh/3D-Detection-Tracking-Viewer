import numpy as np
import re
from .kitti_data_base import *
import os

class KittiDetectionDataset:
    def __init__(self,root_path,label_path = None):
        self.root_path = root_path
        self.velo_path = os.path.join(self.root_path,"velodyne")
        self.image_path = os.path.join(self.root_path,"image_2")
        self.calib_path = os.path.join(self.root_path,"calib")
        if label_path is None:
            self.label_path = os.path.join(self.root_path, "label_2")
        else:
            self.label_path = label_path

        self.all_ids = os.listdir(self.velo_path)

    def __len__(self):
        return len(self.all_ids)
    def __getitem__(self, item):

        name = str(item).zfill(6)

        velo_path = os.path.join(self.velo_path,name+'.bin')
        image_path = os.path.join(self.image_path, name+'.png')
        calib_path = os.path.join(self.calib_path, name+'.txt')
        label_path = os.path.join(self.label_path, name+".txt")

        P2,V2C = read_calib(calib_path)
        points = read_velodyne(velo_path,P2,V2C)
        image = read_image(image_path)
        labels,label_names = read_detection_label(label_path)
        labels[:,3:6] = cam_to_velo(labels[:,3:6],V2C)[:,:3]

        return P2,V2C,points,image,labels,label_names

class KittiTrackingDataset:
    def __init__(self,root_path,seq_id,label_path=None):
        self.seq_name = str(seq_id).zfill(4)
        self.root_path = root_path
        self.velo_path = os.path.join(self.root_path,"velodyne",self.seq_name)
        self.image_path = os.path.join(self.root_path,"image_02",self.seq_name)
        self.calib_path = os.path.join(self.root_path,"calib",self.seq_name)



        self.all_ids = os.listdir(self.velo_path)
        calib_path = self.calib_path + '.txt'

        if label_path is None:

            label_path = os.path.join(self.root_path, "label_02", self.seq_name+'.txt')


        self.P2, self.V2C = read_calib(calib_path)
        self.labels, self.label_names = read_tracking_label(label_path)

        # per-frame ego pose (velodyne frame at frame t -> fixed world frame),
        # used to draw object trajectories in world coordinates. Requires the
        # KITTI oxts(GPS/IMU) data; stays None when it is not available.
        self.poses = None
        oxts_path = os.path.join(self.root_path, "oxts", self.seq_name + '.txt')
        imu_to_velo = read_imu_to_velo(calib_path)
        if os.path.exists(oxts_path) and imu_to_velo is not None:
            imu_poses = load_oxts_poses(oxts_path)          # T_world_imu
            velo_to_imu = np.linalg.inv(imu_to_velo)         # velodyne -> imu
            self.poses = [p @ velo_to_imu for p in imu_poses]  # T_world_velo


    def __len__(self):
        return len(self.all_ids)-1
    def __getitem__(self, item):

        name = str(item).zfill(6)

        velo_path = os.path.join(self.velo_path,name+'.bin')
        image_path = os.path.join(self.image_path, name+'.png')



        points = read_velodyne(velo_path,self.P2,self.V2C)
        image = read_image(image_path)

        if item in self.labels.keys():
            labels = self.labels[item]
            labels = np.array(labels)
            labels[:,3:6] = cam_to_velo(labels[:,3:6],self.V2C)[:,:3]
            label_names = self.label_names[item]
            label_names = np.array(label_names)
        else:
            labels = None
            label_names = None

        return self.P2,self.V2C,points,image,labels,label_names