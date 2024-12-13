import numpy as np
import cv2
import open3d as o3d
import os
import sys
import json
import argparse
from scipy.spatial import KDTree
import copy
from tqdm import tqdm
from typing import List
from einops import rearrange

from prepare_data import *

class CameraInfo:
    '''
    Camera info, storing all the parameters for a camera.
    '''
    def __init__(self, meta_path):
        meta_info = read_meta_data(meta_path)
        self.extrinsics = None
        for key, value in meta_info.items():
            setattr(self, key, value)
    
    def get_cam_id(self):
        return self.cam_series[-4:]
    def get_distortion(self):
        return self.color_distortion
    
    def get_intrinsics(self):
        return self.color_intrinsics
    def get_extrinsics(self):
        return self.color_offset_extrinsics
    

def weighted_sum(distances, coor_3d):
    num = distances.shape[0]
    dist = distances.reshape((num, -1, 1))
    weights = 1 / (dist + 1e-8) 
    weighted_coor = np.sum(weights * coor_3d, axis=1)
    average_coor = weighted_coor / np.sum(weights, axis=1)
    return average_coor


class ViewFrame:
    '''
    One frame of image, including depth, color, mask, and corresponding camera info.
    '''
    def __init__(self, color_raw_path, cam_info: CameraInfo, mask_path=None):
        depth_path = color_raw_path.replace('COLOR', 'DEPTH')
        self.color_img = read_color_image(color_raw_path, cam_info.color_height, cam_info.color_width)
        self.depth_img = read_depth_image(depth_path, cam_info.depth_scale, cam_info.depth_height, cam_info.depth_width)
        self.mask = read_mask_image(mask_path) if mask_path is not None else None
        self.cam_info = cam_info
        frame_str = color_raw_path.split('/')[-1].split('.')[1]
        self.frame_id = frame_str
        
        self.depth_xyz = None
        self.color_xyz = None
        self.depth_to_color_warped_pixels = None
        self.kd_tree = None
    
    def get_cam_id(self):
        return self.cam_info.cam_series[-4:]
    
    def get_color_xyz(self):
        '''
        Get the point cloud coordinates in the color camera space.
        '''
        if self.color_xyz is None:
            if self.depth_xyz is None:
                self.depth_xyz = build_point_cloud_from_depth(self.depth_img, self.cam_info.depth_intrinsics)
            depth_xyz = self.depth_xyz
            depth_xyz_homogeneous = np.hstack((self.depth_xyz, np.ones((self.depth_xyz.shape[0], 1))))
            depth_to_color_cam_xyz = np.dot(depth_xyz_homogeneous, self.cam_info.color_offset_extrinsics.T)
            depth_to_color_cam_xyz = depth_to_color_cam_xyz[:, :3]
            self.color_xyz = depth_to_color_cam_xyz
        return self.color_xyz
    
    def get_pcd(self, near_clip=None, far_clip=None, downsample_step=None):
        depth_mask = np.ones_like(self.depth_img, dtype=bool)
        
        if near_clip is not None and far_clip is not None:
            depth_mask = depth_mask & (self.depth_img >= near_clip) & (self.depth_img <= far_clip)
        if downsample_step is not None:
            downsample_mask = np.zeros_like(self.depth_img, dtype=bool)
            downsample_mask[::downsample_step, ::downsample_step] = True
            depth_mask = depth_mask & downsample_mask
            
        depth_mask = depth_mask.flatten()
        warped_pixels = self.get_warped_pixels()
        warped_pixels_int = warped_pixels.astype(int)
        x_coords = warped_pixels_int[:, 0]
        y_coords = warped_pixels_int[:, 1]
        warped_mask = (x_coords >= 0) & (x_coords < self.cam_info.color_width) & \
                      (y_coords >= 0) & (y_coords < self.cam_info.color_height)
        valid_mask = warped_mask & depth_mask
        
        if self.mask is not None:
            y_coords_clip = np.clip(y_coords, 0, self.cam_info.color_height - 1)
            x_coords_clip = np.clip(x_coords, 0, self.cam_info.color_width - 1)
            valid_mask = valid_mask & self.mask[y_coords_clip, x_coords_clip]
        
        colors = self.color_img[y_coords[valid_mask], x_coords[valid_mask]]
        depth_to_color_cam_xyz = self.get_color_xyz()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(depth_to_color_cam_xyz[valid_mask])
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.)
        return pcd
    
    def get_warped_pixels(self):
        '''
        Get the depth 2d coordinates warped to the color camera pixel space.
        '''
        if self.depth_to_color_warped_pixels is None:
            depth_to_color_cam_xyz = self.get_color_xyz()
            depth_to_color_warped_pixels = np.dot(self.cam_info.color_intrinsics, depth_to_color_cam_xyz.T).T
            depth_to_color_warped_pixels = depth_to_color_warped_pixels[:, :] / depth_to_color_warped_pixels[:, 2:]
            self.depth_to_color_warped_pixels = depth_to_color_warped_pixels[:, :2]
        return self.depth_to_color_warped_pixels
    
    def project_points_to_3d(self, points_2d, blending_k = 4):
        if self.kd_tree is None:
            self.kd_tree = KDTree(self.get_warped_pixels())
        tree = self.kd_tree
        distances, indices = tree.query(points_2d, k=blending_k)
        indices = indices.squeeze()
        indice_coors = self.get_color_xyz()[indices]
        coor_3d = weighted_sum(distances, indice_coors)
        return coor_3d
    
    def get_chessboard_corners_3d(self, board_size=(7,4)):
        corners = self.detect_chessboard_corners_2d(board_size)
        if corners is None:
            return None
        return self.project_points_to_3d(corners)
    
    def detect_chessboard_corners_2d(self, board_size=(7,4)):
        gray_img = cv2.cvtColor(self.color_img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_img, board_size, None)
        if not ret:
            return None
        return corners
    
    @staticmethod
    def get_view_infos(root_path, frame_id, cam_list, mask_root=None):
        view_infos = []
        for cam_id in cam_list:
            cam_info = CameraInfo(get_meta_path(root_path, cam_id))
            mask_path = get_mask_path(mask_root, frame_id, cam_id) if mask_root is not None else None
            view_infos.append(ViewFrame(get_color_raw_path(root_path, frame_id, cam_id), cam_info, mask_path))
        return view_infos

class TransformCollection:
    def __init__(self):
        self.transform_data = {} # (s_id, t_id) -> trans_mat
    
    def add_transform(self, s_id, t_id, trans_mat):
        self.transform_data[(s_id, t_id)] = trans_mat
        
    def add_transform_dict(self, trans_dict):
        for (s_id, t_id), trans_mat in trans_dict.items():
            self.add_transform(s_id, t_id, trans_mat)
        
    def get_transform(self, s_id, t_id):
        if (s_id, t_id) in self.transform_data.keys():
            return self.transform_data[(s_id, t_id)]
        else:
            return None
    
    def get_transform_to_target(self, t_id):
        trans_keys = [key for key in self.transform_data.keys() if key[1] == t_id]
        target_trans_dict = {key[0]: self.transform_data[key] for key in trans_keys}
        return target_trans_dict
    
    def mat_mul(self, a_id, b_id, c_id):
        a2b_mat = self.get_transform(a_id, b_id)
        b2c_mat = self.get_transform(b_id, c_id)
        a2c_mat = np.dot(b2c_mat, a2b_mat)
        self.add_transform(a_id, c_id, a2c_mat)
        
    def merge_to_target(self, t_id):
        trans_pairs = list(self.transform_data.keys())
        cameras = list(set([pair[0] for pair in trans_pairs] + [pair[1] for pair in trans_pairs]))
        merged_cameras = [pair[0] for pair in trans_pairs if pair[1] == t_id]
        unmerged_cameras = [cam for cam in cameras if cam not in merged_cameras]
        
        continue_flag = True
        while continue_flag:
            continue_flag = False
            for um_cam in unmerged_cameras:
                for m_cam in merged_cameras:
                    if self.get_transform(um_cam, m_cam) is not None:
                        self.mat_mul(um_cam, m_cam, t_id)
                        merged_cameras.append(um_cam)
                        unmerged_cameras.remove(um_cam)
                        continue_flag = True
                        break
        return merged_cameras
    
    def load_collection(self, root_path):
        trans_list = os.listdir(root_path)
        for trans_fn in trans_list:
            fn_split = trans_fn.split('.')[0].split('_')
            s_id = fn_split[0]
            t_id = fn_split[2]
            trans_mat = np.loadtxt(os.path.join(root_path, trans_fn))
            self.add_transform(s_id, t_id, trans_mat)

    def save_collection(self, root_path):
        for (s_id, t_id), trans_mat in self.transform_data.items():
            trans_fn = os.path.join(root_path, f'{s_id}_to_{t_id}.txt')
            np.savetxt(trans_fn, trans_mat)
    
def merge_pcd(trans_collection: TransformCollection, pcd_list, cam_id_list, target_cam_id):
    if target_cam_id not in cam_id_list:
        print(f'Error! Target camera {target_cam_id} not in the camera list. Quit.')
        return
    cam_dict = {cam_id: pcd for cam_id, pcd in zip(cam_id_list, pcd_list)}
    combined_pcd = o3d.geometry.PointCloud()
    for source_cam_id in cam_id_list:
        trans_mat = trans_collection.get_transform(source_cam_id, target_cam_id)
        if trans_mat is None:
            print(f'Warning! No transform found between {source_cam_id} and {target_cam_id}. Skip.')
            continue
        source_temp = copy.deepcopy(cam_dict[source_cam_id])
        source_temp.transform(trans_mat)
        combined_pcd += source_temp
    return combined_pcd
    
def camera_calibration_pnp(source_frame_info:ViewFrame, target_frame_info:ViewFrame, k=4):
    obj_points = target_frame_info.get_chessboard_corners_3d()
    source_corners = source_frame_info.detect_chessboard_corners_2d()
    if source_corners is None:
        print(f'Warning! No chessboard detected in source {source_frame_info.get_cam_id()} on frame {source_frame_info.frame_id}. Quit.')
        return None
    success, rvec, tvec = cv2.solvePnP(obj_points, source_corners, source_frame_info.cam_info.color_intrinsics, source_frame_info.cam_info.color_distortion)
    # success, rvec, tvec, _ = cv2.solvePnPRansac(obj_points, source_corners, source_frame_info.cam_info.color_intrinsics, source_frame_info.cam_info.color_distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    if success:
        homo_mat = np.eye(4)
        R, _ = cv2.Rodrigues(rvec)
        homo_mat[:3, :] = np.hstack((R, tvec))
        trans_mat = np.linalg.inv(homo_mat)
        return trans_mat

def calibration_icp_adjust(source_info: ViewFrame, target_info: ViewFrame, init_trans_mat, board_size=(7,4), threshold=0.02):
    # find chessboard corners in target image
    target_corners = target_info.get_chessboard_corners_3d(board_size)
    if target_corners is None:
        print(f'Error! No chessboard detected in target {target_info.get_cam_id()} on frame {target_info.frame_id}. Quit.')
        return None
    
    source_corners = source_info.get_chessboard_corners_3d(board_size)
    if source_corners is None:
        print(f'Error! No chessboard detected in source {source_info.get_cam_id()} on frame {source_info.frame_id}. Quit.')
        return None
    
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_corners)

    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_corners)
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, threshold, init_trans_mat,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return reg_p2p.transformation

class Scene:
    # one frame, multiple views
    class SceneCamera:
        def __init__(self, cam_id, intrinsics, extrinsics):
            self.cam_id = cam_id
            self.intrinsics = intrinsics
            self.extrinsics = extrinsics
        def rotate_90(self, H):
            theta = np.pi / 2
            R_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
            intr = self.intrinsics
            new_intr = np.eye(3)
            new_intr[0, 0] = intr[1, 1]
            new_intr[1, 1] = intr[0, 0]
            new_intr[0, 2] = H - intr[1, 2]
            new_intr[1, 2] = intr[0, 2]
            new_extr = R_mat @ self.extrinsics[:3, :]
            self.intrinsics = new_intr
            self.extrinsics[:3, :] = new_extr
            
    def __init__(self, view_infos: List[ViewFrame], transform_collection: TransformCollection, target_camera = '1246',frame_id = '000000'):
        self.views = view_infos
        self.transforms = transform_collection
        self.target_camera = target_camera
        self.frame_id = str(frame_id).zfill(7)
        self.cameras = None
        self.images = None
        self.masks = None
        self.restart_scene()
        # print(self.cameras)
    
    def restart_scene(self):
        self.cameras = []
        self.images = []
        self.masks = []
        extrinsics_dict = self.transforms.get_transform_to_target(self.target_camera)
        for view in self.views: 
            cam_id = view.cam_info.get_cam_id()
            intrinsics = view.cam_info.get_intrinsics()
            extrinsics = extrinsics_dict[cam_id]
            self.cameras.append(self.SceneCamera(cam_id, intrinsics, extrinsics))
            self.images.append(view.color_img)
            self.masks.append(view.mask)
    def get_hw(self):
        return self.images[0].shape[0], self.images[0].shape[1]
        
    def get_cam_idx(self, cam_id):
        return [cam.cam_id for cam in self.cameras].index(cam_id)
    
    def fetch_cam(self, cam_id):
        cam_idx = self.get_cam_idx(cam_id)
        return self.cameras[cam_idx]
    
    def fetch_img(self, cam_id):
        cam_idx = self.get_cam_idx(cam_id)
        return self.images[cam_idx]
    
    def fetch_mask(self, cam_id):
        cam_idx = self.get_cam_idx(cam_id)
        return self.masks[cam_idx]
    
    def fetch_view(self, cam_id):
        cam_idx = self.get_cam_idx(cam_id)
        return self.views[cam_idx]
    
    def build_pcd(self, cam_list, near_clip=None, far_clip=None, downsample_step=None):
        pcd_list = []
        for cam_id in cam_list:
            view_info = self.fetch_view(cam_id)
            pcd_list.append(view_info.get_pcd(near_clip, far_clip, downsample_step))
        return merge_pcd(self.transforms, pcd_list, cam_list, self.target_camera)

    def rotate_90(self):
        self.images = [np.rot90(img, k=-1) for img in self.images]
        if self.masks is not None:
            self.masks = [np.rot90(mask, k=-1) for mask in self.masks]
        h = self.images[0].shape[0]
        new_cameras = []
        for cam in self.cameras:
            cam.rotate_90(h)
            new_cameras.append(cam)
        self.cameras = new_cameras