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

import utils
import cali

# from cali import read_data, build_point_cloud_from_depth_downsampled, weighted_sum, save_registration_result_color_multi, camera_calibration_kp_icp
# from cam_settings import camera_set, cam_series

source_cam = utils.cam_series['0879']
target_cam = utils.cam_series['1246']

root_path = '/home/gongran/Code/ivp_proj/data/cali_data'
frame_id = str(50).zfill(7)

source_meta_path = os.path.join(root_path, source_cam, f'{frame_id}.json')
target_meta_path = os.path.join(root_path, target_cam, f'{frame_id}.json')

source_cam_info = utils.CameraInfo(utils.get_meta_path(root_path, source_cam))
target_cam_info = utils.CameraInfo(utils.get_meta_path(root_path, target_cam))

source_frame_info = utils.FrameInfo(utils.get_color_raw_path(root_path, frame_id, source_cam), source_cam_info)
target_frame_info = utils.FrameInfo(utils.get_color_raw_path(root_path, frame_id, target_cam), target_cam_info)

trans_dict = utils.camera_calibration_pnp([source_frame_info], target_frame_info, k=4)
# trans_mat = utils.calibration_icp_adjust(source_frame_info, target_frame_info, trans_dict['trans_mat'][source_cam_info.cam_id[-4:]])

trans_cali = cali.camera_calibration(root_path, [source_cam], target_cam, frame_id, k=4)
print(trans_cali.keys())
# print(target_frame_info.get_chessboard_corners_3d())

print(trans_dict[(source_cam_info.get_cam_id(), target_cam_info.get_cam_id())])
print(trans_cali['trans_mat'])

