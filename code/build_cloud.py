import numpy as np
import cv2
import open3d as o3d
import os
import sys
import json
import argparse
import torch
import copy
from scipy.spatial import KDTree
from baseline import *

def build_point_cloud_from_depth_downsampled(info,                              
                                 downsample_step = 10,near_clip=0.5,far_clip=2, output_path = 'tmp'):
    '''Build point cloud from depth image.
    Args:
        depth_img: np.ndarray, depth image.
        depth_intrinsics_matrix: np.ndarray, 3x3 matrix of depth intrinsics, as specified in the meta data.
        depth_scale: float, depth scale, as specified in the meta data.

    Returns:
        xyz: np.ndarray, point cloud coordinates in the camera space.
    '''
    depth_img = info['depth_img']
    depth_intrinsics_matrix = info['depth_intrinsics']
    depth_scale = info['depth_scale']
    
    depth = depth_img.astype(np.float32)
    depth = depth * depth_scale

    # build point cloud
    HEIGHT, WIDTH = depth_img.shape[:2]
    x = np.arange(0, WIDTH)
    y = np.arange(0, HEIGHT)
    x, y = np.meshgrid(x, y)
    # apply intrinsic matrix to get camera space coordinates
    x = (x - depth_intrinsics_matrix[0, 2]) * depth / depth_intrinsics_matrix[0, 0]
    y = (y - depth_intrinsics_matrix[1, 2]) * depth / depth_intrinsics_matrix[1, 1]
    
    downsample_mask = np.zeros_like(depth, dtype=bool)
    downsample_mask[::downsample_step, ::downsample_step] = True
    clip_mask = (depth >= near_clip) & (depth <= far_clip)
    dc_mask = clip_mask & downsample_mask
    
    x = x[dc_mask].flatten()
    y = y[dc_mask].flatten()
    z = depth[dc_mask].flatten()
    xyz_downsampled = np.vstack((x, y, z)).T
    xyz_downsampled = xyz_downsampled.astype(np.float32)
    # xyz = xyz.astype(np.float32)
    
    offset_extrinsics = info['color_offset_extrinsics']
    color_intrinsic_matrix = info['intrinsics']

    xyz_homogeneous = np.hstack((xyz_downsampled, np.ones((xyz_downsampled.shape[0], 1))))
    # Apply extrinsics and transform to color camera space
    depth_to_color_cam_xyz = np.matmul(xyz_homogeneous, offset_extrinsics.T)
    depth_to_color_cam_xyz = depth_to_color_cam_xyz[:, :3]
    
    depth_to_color_warped_pixels = np.dot(color_intrinsic_matrix, depth_to_color_cam_xyz.T).T
    depth_to_color_warped_pixels = depth_to_color_warped_pixels[:, :] / depth_to_color_warped_pixels[:, 2:]
    depth_to_color_warped_pixels = depth_to_color_warped_pixels[:, :2]
    
    depth_to_color_warped_pixels_int = depth_to_color_warped_pixels.astype(int)
    
    y_coords = depth_to_color_warped_pixels_int[:, 1]
    x_coords = depth_to_color_warped_pixels_int[:, 0]
    

    image = info['rgb_img']
    img_mask = info['mask']
    valid_mask = (x_coords >= 0) & (x_coords < image.shape[1]) & (y_coords >= 0) & (y_coords < image.shape[0])
    # Combine valid_mask with img_mask
    if img_mask is not None:
        valid_mask = valid_mask & img_mask[y_coords, x_coords]

    colors = image[y_coords[valid_mask], x_coords[valid_mask]]
    
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(depth_to_color_cam_xyz[valid_mask])
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.)

    # breakpoint()

    
    cam_id = info['cam_id']
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    o3d.io.write_point_cloud(os.path.join(output_path,f'pcd-{cam_id}.ply'), pcd)
    return pcd

if __name__ == "__main__":
    # build pcd directly

    root_path = '/scratch/projects/fouheylab/dma9300/recon3d/data'
    output_path = '/scratch/projects/fouheylab/dma9300/recon3d/data/point_clouds'
    downsample_step = 4
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i in range(len(keyframe_list)):
                
        frame_id = str(keyframe_list[i]).zfill(7)
        for cam in list(cam_series.keys()):
            test_serie = cam_series[cam]
            info = read_data(root_path, test_serie, frame_id)
            build_point_cloud_from_depth_downsampled(info, far_clip=2, output_path=os.path.join(output_path, frame_id), downsample_step=downsample_step)

