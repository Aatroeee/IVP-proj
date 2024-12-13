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
from cam_info import *

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

def get_meta_path(root_path, cam_id):
    cam_series_id = cam_series[cam_id]
    return os.path.join(root_path, 'meta_data', f'{cam_series_id}-MODEL.json')

def get_color_raw_path(root_path, frame_id, cam_id):
    frame_str = str(frame_id).zfill(7)
    cam_series_id = cam_series[cam_id]
    return os.path.join(root_path, 'raw_data', f'{frame_str}', f'{cam_series_id}-COLOR.{frame_str}.raw')
# billy_data/masks_data/0000001/masks
def get_mask_path(mask_root, frame_id, cam_id):
    frame_str = str(frame_id).zfill(7)
    cam_series_id = cam_series[cam_id]
    cam_mask_path = os.path.join(mask_root, f'{cam_series_id}.png')
    return cam_mask_path

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