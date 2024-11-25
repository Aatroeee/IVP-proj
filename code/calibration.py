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
        source_info = read_data(root_path, s_cam, frame_id, False)
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
    root_path = 'data'
    output_path = 'data/output_data_best'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for cali_set in cali_sequence:
        print("printing run")
        source_series = [cam_series[i] for i in cali_set['source']]
        target_serie = cam_series[cali_set['target']]
        trans_dict = camera_calibration(root_path, source_series, target_serie, frame_id=cali_set['frame'], k=4)
        save_trans_dict(trans_dict, output_path)

