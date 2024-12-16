import numpy as np
import cv2
import open3d as o3d
import os
import sys
import json
import argparse
import torch
from scipy.spatial import KDTree
from baseline import *
import os, pdb

def convert_depth_2_world_points(info, near_clip=0.5,far_clip=2, RT= None):
    '''Build point cloud from depth image.
    Args:
        depth_img: np.ndarray, depth image.
        depth_intrinsics_matrix: np.ndarray, 3x3 matrix of depth intrinsics, as specified in the meta data.
        depth_scale: float, depth scale, as specified in the meta data.

    Returns:
        xyz: np.ndarray, point cloud coordinates in the camera space.
    '''
    # breakpoint()
    depth_img = info['depth_img']
    depth_scale = info['depth_scale']
    
    depth = depth_img.astype(np.float32)
    depth = depth * depth_scale

    # build point cloud
    HEIGHT, WIDTH = depth_img.shape[:2]
    x = np.arange(0, WIDTH)
    y = np.arange(0, HEIGHT)
    x, y = np.meshgrid(x, y)


    
    x = x.flatten()
    y = y.flatten()
    z = depth.flatten()
    xy = np.vstack((x, y))
    xy_homogeneous = np.concatenate([xy, np.ones((1,xy.shape[1]))], axis = 0)
    intrinsic_matrix = info['intrinsics']

    K_inv = np.linalg.inv(intrinsic_matrix)
    camera_points  = K_inv @ xy_homogeneous 
    camera_points = camera_points.T * z[:, None] 
    camera_points_homogeous  =  np.concatenate([camera_points.T, np.ones((1,camera_points.T.shape[1]))], axis =0)

    RT_inv = np.linalg.inv(RT)
    world_points = RT_inv @ camera_points_homogeous

    return world_points[0:3,:].T


def convert_world_points_2_image_pts(world_pts, K, RT):
    """
    Converts world points to 2D image points using the camera intrinsics and the transformation matrix.
    
    Args:
        world_pts: np.ndarray, 3D world points of shape (N, 3), where N is the number of points.
        K: np.ndarray, Camera intrinsic matrix of shape (3, 3).
        RT: np.ndarray, Camera-to-world transformation matrix of shape (4, 4).
    
    Returns:
        image_pts: np.ndarray, 2D image points in pixel coordinates of shape (N, 2).
    """
    # Convert world points to homogeneous coordinates (Nx4)
    world_pts_homogeneous = np.hstack((world_pts, np.ones((world_pts.shape[0], 1))))  # Shape: (N, 4)
    
    # Transform world points to camera coordinates using RT
    camera_pts_homogeneous = np.dot(RT, world_pts_homogeneous.T).T  # Shape: (N, 4)
    
    # Extract 3D camera points (without homogeneous coordinate)
    camera_pts = camera_pts_homogeneous[:, :3]  # Shape: (N, 3)
    
    # Project camera points to image plane using intrinsic matrix K
    image_pts_homogeneous = np.dot(K, camera_pts.T).T  # Shape: (N, 3)

    z = image_pts_homogeneous[:,2]
    
    # Normalize by the third (homogeneous) coordinate
    image_pts = image_pts_homogeneous[:, :2] / image_pts_homogeneous[:, 2][:, np.newaxis]  # Shape: (N, 2)
    # breakpoint()

    image_pts_z = np.hstack([image_pts, z.reshape(-1, 1)])
    # breakpoint()
    
    return image_pts_z


def check_point_visibility(image_points, depth_map, depth_scale, near_clip = 0.50, far_clip = 2.0):
    """
    Given 3D image points in camera coordinates, check their visibility in the depth map.

    Args:
        image_points_3d (np.ndarray): Array of 3D points in camera coordinates (N, 3), where each row is a point (x, y, z).
        depth_map (np.ndarray): Depth map (HxW) where each value corresponds to the depth at each pixel.
        depth_scale (float): Scale to convert depth values into real-world units (e.g., meters).
        near_clip (float): Near clipping plane distance (in meters).
        far_clip (float): Far clipping plane distance (in meters).
        K (np.ndarray): Camera intrinsic matrix (3x3).

    Returns:
        visibility (np.ndarray):  mask with True meaning point is visible and False meaning point is occluded.
    """
    
    # Extract the 3D points in camera coordinates (image_points_3d = [x, y, z])
    u_coords = np.round(image_points[:, 0]).astype(np.int64)
    v_coords = np.round(image_points[:, 1]).astype(np.int64)
    z = image_points[:, 2]

    visibility_mask  = np.zeros((image_points.shape[0]), dtype = bool)
    # Check if the projected points are within the image boundaries
    valid_mask = (u_coords >= 0) & (u_coords < depth_map.shape[1]) & (v_coords >= 0) & (v_coords < depth_map.shape[0]) & (z>=near_clip) & (z<=far_clip)
     
    u_coords, v_coords,z = u_coords[valid_mask], v_coords[valid_mask], z[valid_mask]

    
    # Convert the depths to real-world units using depth_scale
    depth_map = depth_map * depth_scale

    visibility_mask[valid_mask] = (depth_map[v_coords, u_coords]>=near_clip) & (depth_map[v_coords, u_coords]<=far_clip) & (depth_map[v_coords, u_coords]>(z+1e-8))

    return visibility_mask







def weighted_sum(distances, coor_3d):
    dist = distances.reshape((-1,1))
    weights = 1 / (dist + 1e-8) 
    weighted_coor = np.sum(weights * coor_3d, axis=0)
    average_coor = weighted_coor / np.sum(weights)
    return average_coor

# calibrate all source cameras to target camera
def camera_calibration(root_path, source_cam, target_cam, frame_id, k=4):
    frame_id = str(frame_id).zfill(7)
    target_info = read_data(root_path, target_cam, frame_id,True)
    
    # find chessboard corners in target image
    cb_size = (7,4)
    gray_target_img = cv2.cvtColor(target_info['rgb_img'], cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_target_img, cb_size, None)
    if not ret:
        print(f'Error! No chessboard  in detected in target {target_cam} on frame {frame_id}. Quit.')
        return
    corners = corners.squeeze()
    
    # calculate obj points in target space
    tree = KDTree(target_info['depth_warp_pixels'])
    obj_points = []
    for new_point in corners:
        distances, indices = tree.query(new_point, k=k)
        coor_3d = target_info['depth_warp_xyz'][indices]
        obj_points.append(weighted_sum(distances, coor_3d))
    obj_points = np.array(obj_points)
    
    # TBA: check rvec and tvec
    
    source_mat = {}
    for s_cam in source_cam:
        # find chessboard corners in source image
        source_info = read_data(root_path, s_cam, frame_id,True )
        # cv2.imwrite('output_image.jpg', source_info['rgb_img'])

        gray_source_img = cv2.cvtColor(source_info['rgb_img'], cv2.COLOR_BGR2GRAY)
        ret, source_corners = cv2.findChessboardCorners(gray_source_img, cb_size, None)
        if not ret:
            print(f'Camera {s_cam} in detect chessboard in source frame {frame_id} fail! Continue.')
            continue
        # solve pnp to get rvec and tvec   
        success, rvec, tvec = cv2.solvePnP(obj_points, source_corners, source_info['intrinsics'], source_info['distortion'])
        if success:

            
            homo_mat = np.eye(4)
            R, _ = cv2.Rodrigues(rvec)
            # note: camera to world(1246)
            homo_mat[:3, :] = np.hstack((R, tvec))
            source_mat[s_cam[-4:]] = np.linalg.inv(homo_mat)
            world_pts1 = convert_depth_2_world_points(target_info, near_clip=0.5,far_clip=2, RT= homo_mat)
            world_pts2 = convert_depth_2_world_points(source_info, near_clip=0.5,far_clip=2, RT= homo_mat)

            # breakpoint()
            img_pts1 = convert_world_points_2_image_pts(world_pts2, target_info["intrinsics"], RT = homo_mat)
            img_pts2 = convert_world_points_2_image_pts(world_pts1, source_info["intrinsics"], RT = homo_mat)

            visibility_mask_1 = check_point_visibility(image_points = img_pts2, depth_map = target_info["depth_img"], depth_scale = target_info["depth_scale"], near_clip = 0.50, far_clip = 2.0)
            visibility_mask_2 = check_point_visibility(image_points = img_pts1, depth_map = source_info["depth_img"], depth_scale = source_info["depth_scale"], near_clip = 0.50, far_clip = 2.0)

            visibility_mask = np.logical_and(visibility_mask_1, visibility_mask_2)

            # apply visibility mask on depth map
            # source_info["depth_img"][~visibility_mask.reshape(720,1280)] = 0.00
            # target_info["depth_img"][~visibility_mask.reshape(720,1280)] = 0.00


            breakpoint()




        
    trans_dict = dict(
        target = target_cam[-4:],
        trans_mat = source_mat,
        depth_mask = visibility_mask.reshape(source_info["depth_image"].shape[0],source_info["depth_image"].shape[1])
    )
    return trans_dict
    
def save_trans_dict(td, wb_path):
    if not os.path.exists(wb_path):
        os.mkdir(wb_path)
    target_id = td['target']
    trans_dict = td['trans_mat']
    for sid in trans_dict.keys():
        trans_mat = trans_dict[sid]
        trans_fn = f'{sid}_to_{target_id}_H_fine.txt'
        with open(os.path.join(wb_path, trans_fn), 'w') as wb_f:
            np.savetxt(wb_f, trans_mat)
        

    
if __name__ == "__main__":
    cali_sequence = []
    for i in range(8):
        cali_sequence.append(
            dict(
                target = camera_set[i][-2],
                frame = keyframe_list[i],
                source = camera_set[i] + camera_set[(i+1) % 8] + camera_set[(i-1) % 8]
            )
        )
    root_path = '/scratch/projects/fouheylab/dma9300/recon3d/data'
    output_path = '/scratch/projects/fouheylab/dma9300/recon3d/data/output_data'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for cali_set in cali_sequence:
        source_series = [cam_series[i] for i in cali_set['source']]
        target_serie = cam_series[cali_set['target']]
        trans_dict = camera_calibration(root_path, source_series, target_serie, frame_id=cali_set['frame'], k=4)
        save_trans_dict(trans_dict, output_path)

