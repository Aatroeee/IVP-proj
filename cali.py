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


from read_depth_and_build_pcd import read_depth_image, read_color_image, build_point_cloud_from_depth
from cam_settings import cam_series

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
        mask = mask > 128
        
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

def weighted_sum(distances, coor_3d):
    dist = distances.reshape((-1,1))
    weights = 1 / (dist + 1e-8) 
    weighted_coor = np.sum(weights * coor_3d, axis=0)
    average_coor = weighted_coor / np.sum(weights)
    return average_coor

# calibrate all source cameras to target camera
def camera_calibration(root_path, source_cam, target_cam, frame_id, k=4):
    frame_id = str(frame_id).zfill(7)
    target_info = read_data(root_path, target_cam, frame_id)
    
    # find chessboard corners in target image
    cb_size = (7,4)
    gray_target_img = cv2.cvtColor(target_info['rgb_img'], cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_target_img, cb_size, None)
    if not ret:
        print(f'Error! No chessboard detected in target {target_cam} on frame {frame_id}. Quit.')
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
        source_info = read_data(root_path, s_cam, frame_id, False)
        gray_source_img = cv2.cvtColor(source_info['rgb_img'], cv2.COLOR_BGR2GRAY)
        ret, source_corners = cv2.findChessboardCorners(gray_source_img, cb_size, None)
        if not ret:
            print(f'Camera {s_cam} detect chessboard in frame {frame_id} fail! Continue.')
            continue
        # solve pnp to get rvec and tvec   
        success, rvec, tvec = cv2.solvePnP(obj_points, source_corners, source_info['intrinsics'], source_info['distortion'])
        if success:
            homo_mat = np.eye(4)
            R, _ = cv2.Rodrigues(rvec)
            # note: camera to world(1246)
            homo_mat[:3, :] = np.hstack((R, tvec))
            source_mat[s_cam[-4:]] = np.linalg.inv(homo_mat)
    trans_dict = dict(
        target = target_cam[-4:],
        trans_mat = source_mat
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
        

# multiply two transformation matrix to get a new one
def trans_mat_mul(root_path, a_id, b_id, c_id):  
    a2b_fn = os.path.join(root_path, f'{a_id}_to_{b_id}_H_fine.txt')
    b2c_fn = os.path.join(root_path, f'{b_id}_to_{c_id}_H_fine.txt')
    a2c_fn = os.path.join(root_path, f'{a_id}_to_{c_id}_H_fine.txt')
    
    if os.path.exists(a2c_fn):
        print(f'{a_id} to {c_id} transform exist. Finish.')
        return True
    
    if not os.path.exists(a2b_fn) or not os.path.exists(b2c_fn):
        return False
    
    a2b_mat = np.loadtxt(a2b_fn)
    b2c_mat = np.loadtxt(b2c_fn)
    
    a2c_mat = np.dot(b2c_mat, a2b_mat)
    
    np.savetxt(a2c_fn, a2c_mat, fmt='%e')
    return True


def read_pcd_and_trans(cam_id, pcd_path, trans_path, target_id='1246'):
    trans_fn = os.path.join(trans_path, f'{cam_id}_to_{target_id}_H_fine.txt')
    cam_serie = cam_series[cam_id]
    pcd_fn = os.path.join(pcd_path, f'pcd-{cam_serie}.ply')
    pcd = o3d.io.read_point_cloud(pcd_fn)
    trans = np.loadtxt(trans_fn) if cam_id != target_id else np.eye(4)
    return dict(
        pcd = pcd,
        trans = trans
    )

def save_registration_result_color_multi(sources, target, transformations, output_path):
    geometries = [copy.deepcopy(target)]
    combined_point_cloud = copy.deepcopy(target)
    transformations = np.array(transformations)
    if transformations.ndim == 2:
        transformations = np.expand_dims(transformations, axis=0)

    for i in range(len(sources)):
        source_temp = copy.deepcopy(sources[i])
        source_temp.transform(transformations[i])
        geometries.append(copy.deepcopy(source_temp))
        combined_point_cloud += source_temp

    o3d.io.write_point_cloud(output_path, combined_point_cloud)

if __name__ == '__main__':
    root_path = 'data/cali_data'
    pcd_output_path = os.path.join(root_path, 'pcd_output')
    downsample_step = 4
    if not os.path.exists(pcd_output_path):
        os.mkdir(pcd_output_path)
    for i in range(len(keyframe_list)):
        frame_id = str(keyframe_list[i]).zfill(7)
        for cam in list(cam_series.keys()):
            test_serie = cam_series[cam]
            info = read_data(root_path, test_serie, frame_id)
            output_fn = os.path.join(pcd_output_path, frame_id)
            build_point_cloud_from_depth_downsampled(info, far_clip=2, output_path=output_fn, downsample_step=downsample_step)

    cali_sequence = []
    for i in range(8):
        cali_sequence.append(
            dict(
                target = camera_set[i][-2],
                frame = keyframe_list[i],
                source = camera_set[i] + camera_set[(i+1) % 8] + camera_set[(i-1) % 8]
            )
        )
        
    trans_path = os.path.join(root_path, 'output_data')
    if not os.path.exists(trans_path):
        os.mkdir(trans_path)
    for cali_set in cali_sequence:
        source_series = [cam_series[i] for i in cali_set['source']]
        target_serie = cam_series[cali_set['target']]
        trans_dict = camera_calibration(root_path, source_series, target_serie, frame_id=cali_set['frame'], k=4)
        save_trans_dict(trans_dict, trans_path)
    target_id = '1246'
    raw_cam_list = list(cam_series.keys())
    cali_cam_list = camera_set[0] # have xxx_to_1246 directly
    raw_cam_list = [cam for cam in raw_cam_list if cam not in cali_cam_list + camera_set[8]]

    # calibrate all cameras to central camera(1246)
    while len(raw_cam_list):
        trans_list = os.listdir(trans_path)
        trans_list = [t.split('_') for t in trans_list]
        trans_list = [(t[0], t[2]) for t in trans_list]
        
        rm_list = []
        for raw_cam in raw_cam_list:
            for trans in trans_list:
                if trans[0] == raw_cam and trans[1] in cali_cam_list:
                    trans_mat_mul(trans_path, raw_cam, trans[1], target_id)
                    rm_list.append(raw_cam)
                    break
        raw_cam_list = [cam for cam in raw_cam_list if cam not in rm_list]
        cali_cam_list.extend(rm_list)
        print(f'Current left : {len(raw_cam_list)} cameras.')
        
    frame_id = str(50).zfill(7)
    pcd_path = os.path.join(pcd_output_path, frame_id)
    source_id = [cam_id for cam_id in list(cam_series.keys()) if cam_id not in camera_set[8]]
    target_input = read_pcd_and_trans(target_id, pcd_path, trans_path)
    source_input = [read_pcd_and_trans(sid, pcd_path, trans_path, target_id) for sid in source_id]
    source_pcd = [sinfo['pcd'] for sinfo in source_input]
    source_trans = [sinfo['trans'] for sinfo in source_input]
    save_registration_result_color_multi(source_pcd, target_input['pcd'],source_trans, f'data/combined_pcd_{target_id}.ply')