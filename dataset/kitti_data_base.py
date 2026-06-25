import os
import cv2
import re
import numpy as np

"""
input: calib txt path
return: P2: (4,4) 3D camera coordinates to 2D image pixels
        vtc_mat: (4,4) 3D velodyne Lidar coordinates to 3D camera coordinates
"""
def read_calib(calib_path):
    with open(calib_path) as f:
        for line in f.readlines():
            if line[:2] == "P2":
                P2 = re.split(" ", line.strip())
                P2 = np.array(P2[-12:], np.float32)
                P2 = P2.reshape((3, 4))
            if line[:14] == "Tr_velo_to_cam" or line[:11] == "Tr_velo_cam":
                vtc_mat = re.split(" ", line.strip())
                vtc_mat = np.array(vtc_mat[-12:], np.float32)
                vtc_mat = vtc_mat.reshape((3, 4))
                vtc_mat = np.concatenate([vtc_mat, [[0, 0, 0, 1]]])
            if line[:7] == "R0_rect" or line[:6] == "R_rect":
                R0 = re.split(" ", line.strip())
                R0 = np.array(R0[-9:], np.float32)
                R0 = R0.reshape((3, 3))
                R0 = np.concatenate([R0, [[0], [0], [0]]], -1)
                R0 = np.concatenate([R0, [[0, 0, 0, 1]]])
    vtc_mat = np.matmul(R0, vtc_mat)
    return (P2, vtc_mat)


"""
description: read lidar data given 
input: lidar bin path "path", cam 3D to cam 2D image matrix (4,4), lidar 3D to cam 3D matrix (4,4)
output: valid points in lidar coordinates (PointsNum,4)
"""
def read_velodyne(path, P, vtc_mat,IfReduce=True):
    max_row = 374  # y
    max_col = 1241  # x
    lidar = np.fromfile(path, dtype=np.float32).reshape((-1, 4))

    if not IfReduce:
        return lidar

    mask = lidar[:, 0] > 0
    lidar = lidar[mask]
    lidar_copy = np.zeros(shape=lidar.shape)
    lidar_copy[:, :] = lidar[:, :]

    velo_tocam = vtc_mat
    lidar[:, 3] = 1
    lidar = np.matmul(lidar, velo_tocam.T)
    img_pts = np.matmul(lidar, P.T)
    velo_tocam = np.mat(velo_tocam).I
    velo_tocam = np.array(velo_tocam)
    normal = velo_tocam
    normal = normal[0:3, 0:4]
    lidar = np.matmul(lidar, normal.T)
    lidar_copy[:, 0:3] = lidar
    x, y = img_pts[:, 0] / img_pts[:, 2], img_pts[:, 1] / img_pts[:, 2]
    mask = np.logical_and(np.logical_and(x >= 0, x < max_col), np.logical_and(y >= 0, y < max_row))

    return lidar_copy[mask]


"""
description: convert 3D camera coordinates to Lidar 3D coordinates.
input: (PointsNum,3)
output: (PointsNum,3)
"""
def cam_to_velo(cloud,vtc_mat):
    mat=np.ones(shape=(cloud.shape[0],4),dtype=np.float32)
    mat[:,0:3]=cloud[:,0:3]
    mat=np.mat(mat)
    normal=np.mat(vtc_mat).I
    normal=normal[0:3,0:4]
    transformed_mat = normal * mat.T
    T=np.array(transformed_mat.T,dtype=np.float32)
    return T

"""
description: convert 3D camera coordinates to Lidar 3D coordinates.
input: (PointsNum,3)
output: (PointsNum,3)
"""
def velo_to_cam(cloud,vtc_mat):
    mat=np.ones(shape=(cloud.shape[0],4),dtype=np.float32)
    mat[:,0:3]=cloud[:,0:3]
    mat=np.mat(mat)
    normal=np.mat(vtc_mat).I
    normal=normal[0:3,0:4]
    transformed_mat = normal * mat.T
    T=np.array(transformed_mat.T,dtype=np.float32)
    return T

def read_image(path):
    im=cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return im

def read_detection_label(path):

    boxes = []
    names = []

    with open(path) as f:
        for line in f.readlines():
            line = line.split()
            this_name = line[0]
            if this_name != "DontCare":
                line = np.array(line[-7:],np.float32)
                boxes.append(line)
                names.append(this_name)

    return np.array(boxes),np.array(names)

def read_tracking_label(path):

    frame_dict={}

    names_dict={}

    with open(path) as f:
        for line in f.readlines():
            line = line.split()
            this_name = line[2]
            frame_id = int(line[0])
            ob_id = int(line[1])

            if this_name != "DontCare":
                line = np.array(line[10:17],np.float32).tolist()
                line.append(ob_id)


                if frame_id in frame_dict.keys():
                    frame_dict[frame_id].append(line)
                    names_dict[frame_id].append(this_name)
                else:
                    frame_dict[frame_id] = [line]
                    names_dict[frame_id] = [this_name]

    return frame_dict,names_dict


def read_imu_to_velo(calib_path):
    """
    read the imu(GPS/IMU) -> velodyne transform (Tr_imu_velo) from a kitti
    tracking calib file.
    input: calib txt path
    output: (4,4) matrix mapping a point in the imu frame to the velodyne frame,
            or None if the entry is not present
    """
    with open(calib_path) as f:
        for line in f.readlines():
            if line[:14] == "Tr_imu_to_velo" or line[:11] == "Tr_imu_velo":
                mat = re.split(" ", line.strip())
                mat = np.array(mat[-12:], np.float32).reshape((3, 4))
                mat = np.concatenate([mat, [[0, 0, 0, 1]]])
                return mat
    return None


def load_oxts_poses(oxts_path):
    """
    read a kitti oxts (GPS/IMU) file and convert each packet to a 4x4 pose
    matrix that maps a point in the imu frame at that frame into a fixed world
    frame (the first frame is used as the origin).
    input: oxts txt path
    output: list of (4,4) numpy arrays, one per frame (T_world_imu)
    """
    poses = []
    scale = None
    origin_inv = None
    earth_radius = 6378137.0  # in meters, WGS-84

    with open(oxts_path) as f:
        for line in f.readlines():
            vals = line.strip().split()
            if len(vals) < 6:
                continue
            lat, lon, alt = float(vals[0]), float(vals[1]), float(vals[2])
            roll, pitch, yaw = float(vals[3]), float(vals[4]), float(vals[5])

            if scale is None:
                scale = np.cos(lat * np.pi / 180.0)

            # mercator projection of the GPS coordinates
            tx = scale * lon * np.pi * earth_radius / 180.0
            ty = scale * earth_radius * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
            tz = alt

            Rx = np.array([[1, 0, 0],
                           [0, np.cos(roll), -np.sin(roll)],
                           [0, np.sin(roll), np.cos(roll)]])
            Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                           [0, 1, 0],
                           [-np.sin(pitch), 0, np.cos(pitch)]])
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                           [np.sin(yaw), np.cos(yaw), 0],
                           [0, 0, 1]])
            T = np.eye(4)
            T[:3, :3] = Rz @ Ry @ Rx
            T[:3, 3] = [tx, ty, tz]

            if origin_inv is None:
                origin_inv = np.linalg.inv(T)
            poses.append(origin_inv @ T)

    return poses


if __name__ == '__main__':
    path = 'H:/dataset/traking/training/label_02/0000.txt'
    labels,a = read_tracking_label(path)
    print(a)

