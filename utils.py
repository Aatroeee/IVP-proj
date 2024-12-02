import numpy as np
import cv2
import open3d as o3d
import os
import sys
import json
import argparse
import torch
from scipy.spatial import KDTree
import copy
from tqdm import tqdm

cam_series = {
    "1362": "049122251362",
    "1634": "043422251634",
    "4320": "151422254320",
    "1000": "215122251000",
    "1318": "213622251318",
    "1169": "035622251169",
    "2129": "152522252129",
    "1171": "213622301171",
    "0028": "035322250028",
    "8540": "234322308540",
    "1246": "043422251246",
    "1265": "035322251265",
    "1705": "105322251705",
    "1753": "234222301753",
    "1973": "235422301973",
    "0244": "035322250244",
    "1516": "138322251516",
    "1228": "035322251228",
    "1487": "043422251487",
    "1116": "035322251116",
    "0385": "038122250385",
    "2543": "043422252543",
    "0879": "046122250879",
    "0406": "035722250406",
    "2448": "117222252448",
    "1285": "035322251285",
    "1100": "046122251100",
    "1040": "213622301040",
    "0103": "234222300103"
}

camera_set = {
    0 : ["0385", "2543", "1246", "1973"],
    1 : ["4320", "1040", "1634"],
    2 : ["1705", "1318", "1100"],
    3 : ["1285", "1753", "8540"],
    4 : ["1116", "1265", "0103"],
    5 : ["1169", "1516", "2448"],
    6 : ["2129", "0028", "0244"],
    7 : ["1228", "0879", "1362"],
    8 : ["1171", "1000", "1487", "0406"],
}

keyframe_list = np.array([240, 220, 200, 175, 145, 110, 80, 50])

def read_meta_data(meta_path):
    cam_series_id = meta_path.split('/')[-1].split('-')[0]
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    color_intrinsics = meta['color_intrinsics']
    color_intrinsic_matrix = np.array(
        [[color_intrinsics['fx'], 0, color_intrinsics['ppx']],
            [0, color_intrinsics['fy'], color_intrinsics['ppy']],
            [0, 0, 1]])
    color_height = meta['color_intrinsics']['height']
    color_width = meta['color_intrinsics']['width']
    
    distortion_coeffs = [color_intrinsics['k1'],color_intrinsics['k2'],color_intrinsics['p1'],color_intrinsics['p2'],color_intrinsics['k3']]
    distortion_coeffs = np.array(distortion_coeffs)
    intrinsics = meta['depth_intrinsics']
    intrinsic_matrix = np.array([[intrinsics['fx'], 0, intrinsics['ppx']],
                                    [0, intrinsics['fy'], intrinsics['ppy']],
                                    [0, 0, 1]])
    depth_height = meta['depth_intrinsics']['height']
    depth_width = meta['depth_intrinsics']['width']
    depth_scale = meta['depth_scale']
    color_offset_extrinsics = meta['color_offset_extrinsics']
    R = np.array([[color_offset_extrinsics['r00'], color_offset_extrinsics['r01'], color_offset_extrinsics['r02']],
                    [color_offset_extrinsics['r10'], color_offset_extrinsics['r11'], color_offset_extrinsics['r12']],
                    [color_offset_extrinsics['r20'], color_offset_extrinsics['r21'], color_offset_extrinsics['r22']]])
    T = np.array([color_offset_extrinsics['t0'], color_offset_extrinsics['t1'], color_offset_extrinsics['t2']])
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = T
    
    return dict(
        cam_series = cam_series_id,
        
        color_intrinsics = color_intrinsic_matrix,
        color_distortion = distortion_coeffs,
        color_height = color_height,
        color_width = color_width,
        color_offset_extrinsics = extrinsics,
        
        depth_height = depth_height,
        depth_width = depth_width,
        depth_scale = depth_scale,
        depth_intrinsics = intrinsic_matrix,
    )

def read_depth_image(raw_depth_path, depth_scale, height=700, width=1280):
    depth = np.fromfile(raw_depth_path, dtype=np.uint16).reshape(height, width)
    depth = depth.astype(np.float32)
    depth *= depth_scale
    return depth

def read_color_image(raw_color_path, height=800, width=1280):
    raw_img = np.fromfile(raw_color_path, dtype=np.uint8).reshape(height, width, 2)
    bgr_img = cv2.cvtColor(raw_img, cv2.COLOR_YUV2BGR_YUY2)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img

def read_mask_image(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 128
    return mask

def build_point_cloud_from_depth(depth_img: np.ndarray,
                                 depth_intrinsics_matrix: np.ndarray):
    '''Build point cloud from depth image.
    Args:
        depth_img: np.ndarray, depth image.
        depth_intrinsics_matrix: np.ndarray, 3x3 matrix of depth intrinsics, as specified in the meta data.

    Returns:
        xyz: np.ndarray, point cloud coordinates in the camera space.
    '''
    depth = depth_img.astype(np.float32)
    # build point cloud
    HEIGHT, WIDTH = depth_img.shape[:2]
    x = np.arange(0, WIDTH)
    y = np.arange(0, HEIGHT)
    x, y = np.meshgrid(x, y)
    # apply intrinsic matrix to get camera space coordinates
    x = (x - depth_intrinsics_matrix[0, 2]) * depth / depth_intrinsics_matrix[0, 0]
    y = (y - depth_intrinsics_matrix[1, 2]) * depth / depth_intrinsics_matrix[1, 1]

    x = x.flatten()
    y = y.flatten()
    z = depth.flatten()
    xyz = np.vstack((x, y, z)).T
    xyz = xyz.astype(np.float32)
    return xyz

def get_meta_path(root_path, cam_id):
    return os.path.join(root_path, 'meta_data', f'{cam_id}-MODEL.json')

def get_color_raw_path(root_path, frame_id, cam_id):
    frame_str = str(frame_id).zfill(7)
    return os.path.join(root_path, 'raw_data', f'{frame_str}', f'{cam_id}-COLOR.{frame_str}.raw')

class CameraInfo:
    '''
    Camera info, storing all the parameters for a camera.
    '''
    def __init__(self, meta_path):
        meta_info = read_meta_data(meta_path)
        for key, value in meta_info.items():
            setattr(self, key, value)
    
    def get_cam_id(self):
        return self.cam_series[-4:]

def weighted_sum(distances, coor_3d):
    num = distances.shape[0]
    dist = distances.reshape((num, -1, 1))
    weights = 1 / (dist + 1e-8) 
    weighted_coor = np.sum(weights * coor_3d, axis=1)
    average_coor = weighted_coor / np.sum(weights, axis=1)
    return average_coor


class FrameInfo:
    '''
    One frame of images, including depth, color, mask, and corresponding camera info.
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
    
    def get_depth_xyz(self):
        '''
        Get the point cloud coordinates in the depth camera space.
        '''
        if self.depth_xyz is None:
            self.depth_xyz = build_point_cloud_from_depth(self.depth_img, self.cam_info.depth_intrinsics)   
        return self.depth_xyz
    
    def get_color_xyz(self):
        '''
        Get the point cloud coordinates in the color camera space.
        '''
        if self.color_xyz is None:
            depth_xyz = self.get_depth_xyz()
            depth_xyz_homogeneous = np.hstack((depth_xyz, np.ones((depth_xyz.shape[0], 1))))
            depth_to_color_cam_xyz = np.dot(depth_xyz_homogeneous, self.cam_info.color_offset_extrinsics.T)
            depth_to_color_cam_xyz = depth_to_color_cam_xyz[:, :3]
            self.color_xyz = depth_to_color_cam_xyz
        return self.color_xyz
    
    
    def get_downsample_mask(self, downsample_step=2):
        depth_img = self.depth_img
        downsample_mask = np.zeros_like(depth_img, dtype=bool)
        downsample_mask[::downsample_step, ::downsample_step] = True
        return downsample_mask
    
    def get_clip_mask(self, near_clip=0.1, far_clip=3):
        depth_mask = np.ones_like(self.depth_img, dtype=bool)
        depth_mask = depth_mask & (self.depth_img >= near_clip) & (self.depth_img <= far_clip)
        return depth_mask
    
    def get_warped_mask(self):
        warped_pixels = self.get_warped_pixels()
        warped_pixels_int = warped_pixels.astype(int)
        x_coords = warped_pixels_int[:, 0]
        y_coords = warped_pixels_int[:, 1]
        warped_mask = (x_coords >= 0) & (x_coords < self.cam_info.color_width) & \
                      (y_coords >= 0) & (y_coords < self.cam_info.color_height)
        if self.mask is not None:
            warped_mask = warped_mask & self.mask[y_coords, x_coords]
        return warped_mask, x_coords, y_coords
    
    def get_pcd(self, cfg):
        if cfg.near_clip is not None and cfg.far_clip is not None:
            depth_mask = self.get_clip_mask(cfg.near_clip, cfg.far_clip)
        else:
            depth_mask = np.ones_like(self.depth_img, dtype=bool)
        if cfg.downsample_step is not None:
            downsample_mask = self.get_downsample_mask(cfg.downsample_step)
            depth_mask = depth_mask & downsample_mask
        depth_mask = depth_mask.flatten()
        warped_mask, x_coords, y_coords = self.get_warped_mask()
        valid_mask = warped_mask & depth_mask
        
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
    
    def build_kd_tree(self):
        if self.kd_tree is None:
            self.kd_tree = KDTree(self.get_warped_pixels())
        return self.kd_tree
    
    def project_points_to_3d(self, points_2d, blending_k = 4):
        tree = self.build_kd_tree()
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
    
    def merge_pcd(self, pcd_list, cam_id_list, target_cam_id):
        if target_cam_id not in cam_id_list:
            print(f'Error! Target camera {target_cam_id} not in the camera list. Quit.')
            return
        cam_dict = {cam_id: pcd for cam_id, pcd in zip(cam_id_list, pcd_list)}
        source_list = [cam_id for cam_id in cam_id_list if cam_id != target_cam_id]
        combined_pcd = copy.deepcopy(cam_dict[target_cam_id])
        for source_cam_id in source_list:
            source_temp = copy.deepcopy(cam_dict[source_cam_id])
            source_temp.transform(self.get_transform(source_cam_id, target_cam_id))
            combined_pcd += source_temp
        return combined_pcd
    

def camera_calibration_pnp_v2(source_infos: list[FrameInfo], target_info: FrameInfo, k=4, board_size=(7,4)):
    # find chessboard corners in target image
    corners = target_info.get_chessboard_corners_3d(board_size)
    if corners is None:
        print(f'Error! No chessboard detected in target {target_info.get_cam_id()} on frame {target_info.frame_id}. Quit.')
        return
    
    trans_dict = {}
    for source_info in source_infos:
        # find 3d keypoints in source space
        corners_2d = source_info.detect_chessboard_corners_2d(board_size)
        if corners_2d is None:
            print(f'No chessboard detected in source {source_info.get_cam_id()} on frame {source_info.frame_id}. Continue.')
            continue
        success, rvec, tvec = cv2.solvePnP(corners, corners_2d, source_info.cam_info.depth_intrinsics, source_info.cam_info.color_distortion)
        if success:
            homo_mat = np.eye(4)
            R, _ = cv2.Rodrigues(rvec)
            # note: camera to world(1246)
            homo_mat[:3, :] = np.hstack((R, tvec))
            trans_dict[(source_info.get_cam_id(), target_info.get_cam_id())] = np.linalg.inv(homo_mat)
    
    return trans_dict

def camera_calibration_pnp(source_frame_info:utils.FrameInfo, target_frame_info:utils.FrameInfo, k=4):
    obj_points = target_frame_info.get_chessboard_corners_3d()
    source_corners = source_frame_info.detect_chessboard_corners_2d()
    success, rvec, tvec = cv2.solvePnP(obj_points, source_corners, source_frame_info.cam_info.color_intrinsics, source_frame_info.cam_info.color_distortion)
    if success:
        homo_mat = np.eye(4)
        R, _ = cv2.Rodrigues(rvec)
        homo_mat[:3, :] = np.hstack((R, tvec))
        trans_mat = np.linalg.inv(homo_mat)
        return trans_mat

def calibration_icp_adjust(source_info: FrameInfo, target_info: FrameInfo, init_trans_mat, board_size=(7,4), threshold=0.02):
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

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='data/cali_data')
    parser.add_argument('--task', type=str, default='cali')
    parser.add_argument('--trans_path', type=str, default='trans_icp_data')
    parser.add_argument('--pcd_path', type=str, default='pcd_output')
    parser.add_argument('--frame_id', type=str, default='50')
    parser.add_argument('--near_clip', type=float, default=0.1)
    parser.add_argument('--far_clip', type=float, default=3)
    parser.add_argument('--downsample_step', type=int, default=4)
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    root_path = args.root_path
    task = args.task
    trans_path = os.path.join(root_path, args.trans_path)
    pcd_path = os.path.join(root_path, args.pcd_path)
    cam_id_list = [cam_id for cam_id in list(cam_series.keys()) if cam_id not in camera_set[8]]
    cam_series_list = [cam_series[cam_id] for cam_id in cam_id_list]
    cam_info_dict = {cam_id: CameraInfo(get_meta_path(root_path, cam_series[cam_id])) for cam_id in cam_id_list}
    
    target_cam_id = '1246'
    
    if task == 'cali':
        frame_info_dict = {} # (cam_id, frame_id) -> FrameInfo
        for i in range(len(keyframe_list)):
            frame_id = str(keyframe_list[i]).zfill(7)
            for cam_id in cam_id_list:
                frame_info = FrameInfo(get_color_raw_path(root_path, frame_id, cam_series[cam_id]), cam_info_dict[cam_id])
                frame_info_dict[(cam_id, frame_id)] = frame_info
        cali_sequence = []  
        for i in range(8):
            cali_sequence.append(
                dict(
                    target = camera_set[i][-2],
                    frame = keyframe_list[i],
                    source = camera_set[i] + camera_set[(i+1) % 8] + camera_set[(i-1) % 8]
                )
            )
        
        refine_diff = []
        trans_collection = TransformCollection()
        for cali_set in tqdm(cali_sequence):
            current_frame = str(cali_set['frame']).zfill(7)
            source_infos = [frame_info_dict[(i, current_frame)] for i in cali_set['source']]
            target_info = frame_info_dict[(cali_set["target"], current_frame)]
            target_cam_id = target_info.get_cam_id()
            source_cam_ids = [s_cam_id for s_cam_id in cali_set['source']]
            
            trans_dict = camera_calibration_pnp(source_infos, target_info, k=4)
            trans_collection.add_transform_dict(trans_dict)
            
            # valid_source_cam_ids = [s_cam_id for s_cam_id in source_cam_ids if (s_cam_id, target_cam_id) in trans_dict.keys()]
            # for s_cam_id in valid_source_cam_ids:
            #     s_cam_info = frame_info_dict[(s_cam_id, current_frame)]
            #     trans_mat = calibration_icp_adjust(s_cam_info, target_info, trans_dict[(s_cam_id, target_cam_id)])
            #     trans_collection.add_transform(s_cam_id, target_cam_id, trans_mat)
            #     refine_diff.append(np.linalg.norm(trans_mat - trans_dict[(s_cam_id, target_cam_id)]))
        
        if not os.path.exists(trans_path):
            os.mkdir(trans_path)
        
        cali_set = trans_collection.merge_to_target(target_cam_id)
        trans_collection.save_collection(trans_path)
        
        # print(f'Refine diff: {np.mean(refine_diff)}')
    elif task == 'merge':
        trans_collection = TransformCollection()
        trans_collection.load_collection(trans_path)
        # cali_set = trans_collection.merge_to_target(target_cam_id)
        cali_set = ['1246', '0879']
        frame_id = args.frame_id
        save_path = os.path.join(root_path, 'pcd_output', f'{frame_id}')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        frame_info_dict = {}
        pcd_list = []
        for cam_id in cali_set:
            frame_info_dict[cam_id] = FrameInfo(get_color_raw_path(root_path, frame_id, cam_series[cam_id]), cam_info_dict[cam_id])
            pcd_list.append(frame_info_dict[cam_id].get_pcd(args))
        combined_pcd = trans_collection.merge_pcd(pcd_list, cali_set, target_cam_id)
        if not os.path.exists(pcd_path):
            os.mkdir(pcd_path)
        save_fn = os.path.join(pcd_path, f'combined_0879.ply')
        o3d.io.write_point_cloud(save_fn, combined_pcd)
        
        
        