import numpy as np
import cv2
import open3d as o3d
import os
import sys
import json
import argparse
import torch
from scipy.spatial import KDTree


from read_depth_and_build_pcd import read_depth_image, read_color_image, build_point_cloud_from_depth
from cam_settings import cam_series

# some camera settings
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
id2cam = ["1246","2543","0385","1973","1634","1040","4320","0879","1362","1228","0028","0244","2129","2448","1516","1169","0103","1265","1116","1753","8540","1285","1100","1318","1705"]
cam2id = {}
for i, cam in enumerate(id2cam):
    cam2id[cam] = i
keyframe_list = np.array([240, 220, 200, 175, 145, 110, 80, 50]) # keyframe - each group can capture front view of chessboard
verbose = 0


# read data from raw and meta data
def read_data(root_path, cam_series_id, frame_id, need_depth = True, mask_path = None, near_clip=0.5, far_clip=3.0):
    meta_path = os.path.join(root_path, 'meta_data', f'{cam_series_id}-MODEL.json')
    frame_str = str(frame_id).zfill(7)
    depth_path = os.path.join(root_path, 'raw_data', frame_str, f'{cam_series_id}-DEPTH.{frame_str}.raw')
    color_path = depth_path.replace('DEPTH', 'COLOR')
    
    # read data related to color camera
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
    
    rgb_img = read_color_image(color_path, width=color_width, height=color_height, verbose=verbose)
    
    depth_to_color_warped_pixels = None
    depth_to_color_cam_xyz = None
    intrinsic_matrix = None
    depth = None
    depth_clipped = None
    depth_scale = None
    extrinsics = None
    
    # read data related to depth camera
    if need_depth:
        intrinsics = meta['depth_intrinsics']
        intrinsic_matrix = np.array([[intrinsics['fx'], 0, intrinsics['ppx']],
                                        [0, intrinsics['fy'], intrinsics['ppy']],
                                        [0, 0, 1]])
        depth = read_depth_image(depth_path, verbose=verbose)
        depth_scale = meta['depth_scale']
        camera_pts = build_point_cloud_from_depth(
                depth, intrinsic_matrix, depth_scale,
                near_clip=near_clip, far_clip=far_clip)
        depth_clipped = camera_pts[:, -1]
        # Align depth with color pixels
        # build camera to depth extrinsics transform
        color_offset_extrinsics = meta['color_offset_extrinsics']
        R = np.array([[color_offset_extrinsics['r00'], color_offset_extrinsics['r01'], color_offset_extrinsics['r02']],
                        [color_offset_extrinsics['r10'], color_offset_extrinsics['r11'], color_offset_extrinsics['r12']],
                        [color_offset_extrinsics['r20'], color_offset_extrinsics['r21'], color_offset_extrinsics['r22']]])
        T = np.array([color_offset_extrinsics['t0'], color_offset_extrinsics['t1'], color_offset_extrinsics['t2']])
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = T
        
        # Apply the transform to depth xyz
        depth_xyz = camera_pts
        # Extend to homogeneous coordinates
        depth_xyz = np.hstack((depth_xyz, np.ones((depth_xyz.shape[0], 1))))
        # Apply extrinsics and transform to color camera space
        depth_to_color_cam_xyz = np.dot(depth_xyz, extrinsics.T)
        depth_to_color_cam_xyz = depth_to_color_cam_xyz[:, :3]
        
        depth_to_color_warped_pixels = np.dot(color_intrinsic_matrix, depth_to_color_cam_xyz.T).T
        depth_to_color_warped_pixels = depth_to_color_warped_pixels[:, :] / depth_to_color_warped_pixels[:, 2:]
        depth_to_color_warped_pixels = depth_to_color_warped_pixels[:, :2]
    
    mask = None
    if mask_path is not None:
        img_id = cam2id[cam_series_id[-4:]]
        cam_mask_path = os.path.join(mask_path, f'mask_{img_id:02d}.png')
        mask = cv2.imread(cam_mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask>128
      
  
    return dict(
        cam_id = cam_series_id,
        rgb_img = rgb_img,
        intrinsics = color_intrinsic_matrix,
        distortion = distortion_coeffs,
        height = color_height,
        width = color_width,
        
        depth_img = depth,
        depth_clipped = depth_clipped,
        depth_scale = depth_scale,
        depth_warp_pixels = depth_to_color_warped_pixels,
        depth_warp_xyz = depth_to_color_cam_xyz,
        depth_intrinsics = intrinsic_matrix,
        
        color_offset_extrinsics = extrinsics,
        mask = mask
    )


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
    
    cam_id = info['cam_id']
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    o3d.io.write_point_cloud(os.path.join(output_path,f'pcd-{cam_id}.ply'), pcd)
    return pcd


if __name__ == "__main__":
    # build masked pcd
    root_path = 'data/'
    print("runnung!!")
    output_path = os.path.join('data', 'pcd_data')
    frame_id = str(50).zfill(7)
    cam_list = list(cam_series.keys())
    cam_list = [cam for cam in cam_list if cam not in camera_set[8]]
    print(len(cam_list))
    for cam in cam_list:
        print(cam)
        test_serie = cam_series[cam]
        info = read_data(root_path, cam_series[cam], frame_id, need_depth=True, mask_path=os.path.join('data', 'masked_imgs','masks'))
        build_point_cloud_from_depth_downsampled(info, far_clip=2, downsample_step=2, output_path=os.path.join('data', 'masked_pcd_data',))
