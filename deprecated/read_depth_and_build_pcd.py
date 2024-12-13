# read Intel RealSense raw depth image and the meta data with intrinsics
# build point cloud from depth image
import numpy as np
import cv2
import open3d as o3d
import os
import sys
import json
import argparse
import torch


def read_depth_image(depth_path, width=1280, height=720,
                     verbose=False):
    '''Read depth image from raw file.
    Args:
        depth_path: str, path to the raw depth image.
        width: int, width of the depth image.
        height: int, height of the depth image.'''
    depth = np.fromfile(depth_path, dtype=np.uint16).reshape(height, width)

    # Visualize depth image if verbose
    if verbose:
        print('[Verbose] Save depth image to depth_img.png.')
        depth_img = depth.astype(np.float32) / np.max(depth) * 255
        depth_img = depth_img.astype(np.uint8)
        depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
        cv2.imwrite('depth_img.png', depth_img)
    return depth


def read_color_image(color_path, width=1280, height=800,
                     verbose=False):
    '''Read color image from raw file and convert to RGB.
    Args:
        color_path: str, path to the raw color image.
        width: int, width of the color image.
        height: int, height of the color image.'''
    raw_img = np.fromfile(color_path, dtype=np.uint8).reshape(height, width, 2)
    bgr_img = cv2.cvtColor(raw_img, cv2.COLOR_YUV2BGR_YUY2)
    if verbose:
        print('[Verbose] Save color image to color_img.png.')
        cv2.imwrite('color_img.png', bgr_img)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img


def build_point_cloud_from_depth(depth_img: np.ndarray,
                                 depth_intrinsics_matrix: np.ndarray,
                                 depth_scale: float,
                                 near_clip: float = 0.5,
                                 far_clip: float = 5.0):
    '''Build point cloud from depth image.
    Args:
        depth_img: np.ndarray, depth image.
        depth_intrinsics_matrix: np.ndarray, 3x3 matrix of depth intrinsics, as specified in the meta data.
        depth_scale: float, depth scale, as specified in the meta data.

    Returns:
        xyz: np.ndarray, point cloud coordinates in the camera space.
    '''
    depth = depth_img.astype(np.float32)
    depth = depth * depth_scale

    # Filter out (inaccurate) depth values.
    depth[depth < near_clip] = np.nan
    depth[depth > far_clip] = np.nan

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
    xyz = xyz[~np.isnan(xyz).any(axis=1)]
    xyz = xyz.astype(np.float32)
    return xyz


def build_point_cloud_from_raw(depth_path, meta_path,
                               near_clip=0.5, far_clip=5.0,
                               verbose=False):
    '''Build point cloud from raw depth image and meta data.
    Args:
        depth_path: str, path to the raw depth image.
        meta_path: str, path to the meta data file.
        verbose: bool, whether to dump debug information.'''
    # read meta data for intrinsics
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    intrinsics = meta['depth_intrinsics']
    intrinsic_matrix = np.array([[intrinsics['fx'], 0, intrinsics['ppx']],
                                 [0, intrinsics['fy'], intrinsics['ppy']],
                                 [0, 0, 1]])

    # read depth image
    depth = read_depth_image(depth_path, verbose=verbose)
    depth_scale = meta['depth_scale']
    camera_pts = build_point_cloud_from_depth(
        depth, intrinsic_matrix, depth_scale,
        near_clip=near_clip, far_clip=far_clip)

    # read color image
    rgb_fn = depth_path.replace('DEPTH', 'COLOR')
    color_height = meta['color_intrinsics']['height']
    color_width = meta['color_intrinsics']['width']
    rgb_img = read_color_image(rgb_fn, width=color_width, height=color_height, verbose=verbose)

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

    # Apply color camera intrinsics
    color_intrinsics = meta['color_intrinsics']
    color_intrinsic_matrix = np.array(
        [[color_intrinsics['fx'], 0, color_intrinsics['ppx']],
         [0, color_intrinsics['fy'], color_intrinsics['ppy']],
         [0, 0, 1]])
    depth_to_color_warped_pixels = np.dot(color_intrinsic_matrix, depth_to_color_cam_xyz.T).T
    depth_to_color_warped_pixels = depth_to_color_warped_pixels[:, :] / depth_to_color_warped_pixels[:, 2:]
    depth_to_color_warped_pixels = depth_to_color_warped_pixels[:, :2]

    # Organize the depth_to_color_warped_pixels to 2D map
    # 1. pad the depth_to_color_warped_pixels length to multiple of 1024
    # H: -1 -> 0, 1 -> 800
    # W: -1 -> 0, 1 -> 1280
    norm_flow = depth_to_color_warped_pixels.copy()
    norm_flow[:, 0] = norm_flow[:, 0] / 1280 * 2 - 1
    norm_flow[:, 1] = norm_flow[:, 1] / 800 * 2 - 1
    padded_length = (norm_flow.shape[0] + 1023) // 1024 * 1024
    norm_flow_padded = np.zeros((padded_length, 2), dtype=np.float32)
    norm_flow_padded[...] = -1
    norm_flow_padded[:norm_flow.shape[0]] = norm_flow
    norm_flow_padded_h = 1024
    norm_flow_padded_w = norm_flow_padded.shape[0] // 1024
    norm_flow_padded = norm_flow_padded.reshape(
        1, norm_flow_padded_h, norm_flow_padded_w, 2)

    # 2. Convert flow and color map to torch tensor
    norm_flow_padded = torch.from_numpy(norm_flow_padded)
    rgb_img = torch.from_numpy(rgb_img)
    rgb_img = rgb_img.permute(2, 0, 1).unsqueeze(0).float()

    # 3. Warp color image
    warped_color = torch.nn.functional.grid_sample(rgb_img, norm_flow_padded, mode='bilinear', padding_mode='zeros')
    warped_color = warped_color.permute(0, 2, 3, 1).squeeze()
    warped_color = warped_color.cpu().numpy().squeeze()

    # Build color point cloud
    warped_color = warped_color.reshape(-1, 3)[:norm_flow.shape[0]] / 255.

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(camera_pts)
    pcd.colors = o3d.utility.Vector3dVector(warped_color)
    rgb = np.asarray(pcd.colors)
    rgb = rgb.reshape(1, -1, 3) * 255
    rgb = rgb.astype(np.uint8)

    # Convert RGB to HSV
    hsv_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    xyz = np.asarray(pcd.points)

    # Define the range of green color in HSV
    lower_green = np.array([25, 10, 10])  # Adjust these values as needed
    upper_green = np.array([85, 255, 255])  # Adjust these values as needed

    # Create masks for hue, saturation, and value channels
    mask_hue = (hsv_img[:, :, 0] >= lower_green[0]) & (hsv_img[:, :, 0] <= upper_green[0])
    mask_saturation = (hsv_img[:, :, 1] >= lower_green[1]) & (hsv_img[:, :, 1] <= upper_green[1])
    mask_value = (hsv_img[:, :, 2] >= lower_green[2]) & (hsv_img[:, :, 2] <= upper_green[2])

    # Combine masks to get final mask
    mask = mask_hue & mask_saturation & mask_value
    mask = ~mask

    indices = np.where(mask)[1]
    rgb = rgb.reshape(-1, 3)

    xyz = xyz[indices]
    rgb = rgb[indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.)


    if verbose:
        o3d.visualization.draw_geometries([pcd])
    return pcd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Read Intel RealSense raw depth image and the meta data with intrinsics, build point cloud from depth image')
    parser.add_argument('--depth', type=str, help='path to the raw depth image')
    parser.add_argument('--meta', type=str, help='path to the meta data file')
    parser.add_argument('--output', type=str, help='path to the output point cloud file')
    parser.add_argument('--verbose', action='store_true', help='dump debug information')
    parser.add_argument('--near_clip', type=float, default=0.5, help='near clip distance')
    parser.add_argument('--far_clip', type=float, default=5.0, help='far clip distance')
    args = parser.parse_args()

    depth_path = args.depth
    meta_path = args.meta
    output_path = args.output

    pcd = build_point_cloud_from_raw(
        depth_path, meta_path,
        near_clip=args.near_clip, far_clip=args.far_clip,
        verbose=args.verbose)
    o3d.io.write_point_cloud(output_path, pcd)