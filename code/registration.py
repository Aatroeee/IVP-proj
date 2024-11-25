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
# register local pcds to world view
import copy

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
        # breakpoint()
        source_temp.transform(transformations[i])
        geometries.append(copy.deepcopy(source_temp))
        combined_point_cloud += source_temp

    o3d.io.write_point_cloud(output_path, combined_point_cloud)


if __name__ =="__main__":
    frame_id = str(50).zfill(7)
    pcd_path = f'point_clouds/{frame_id}'
    trans_path = 'data/output_data_calib_pnpr2d2'
    target_id = '1246'
    source_id = [cam_id for cam_id in list(cam_series.keys()) if cam_id not in camera_set[8]]
    target_input = read_pcd_and_trans(target_id, pcd_path, trans_path)
    source_input = [read_pcd_and_trans(sid, pcd_path, trans_path, target_id) for sid in source_id]
    source_pcd = [sinfo['pcd'] for sinfo in source_input]
    source_trans = [sinfo['trans'] for sinfo in source_input]
    save_registration_result_color_multi(source_pcd, target_input['pcd'],source_trans, f'data/combined_pcd_pnpr2d2{target_id}.ply')