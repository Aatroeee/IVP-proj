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
import glob
import re
# register local pcds to world view
import copy

def read_pcd_and_trans(cam_id, pcd_path, trans_path, target_id='1246'):
    trans_fn = os.path.join(trans_path, f'{cam_id}_to_{target_id}_H_fine.txt')
    pcd_fn = os.path.join(pcd_path, f'pcd-{target_id}.ply')
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




def valid_camera_series(file_path):
    """
    Filters cam_series and camera_set based on numbers extracted from JSON file names in the given file path.

    Args:
        cam_series (dict): Dictionary of camera series data.
        camera_set (dict): Dictionary of camera set data.
        file_path (str): Path to the directory containing JSON files.

    Returns:
        tuple: Filtered cam_series and camera_set dictionaries.
    """
    # Step 1: Read file names using glob
    file_paths = glob.glob(os.path.join(file_path, "*.json"))
    filenames = [os.path.basename(path) for path in file_paths]

    # Step 2: Extract numbers from file names
    valid = {re.search(r"\d+", filename).group() for filename in filenames}

    return  valid 

def extract_second_numbers(folder_path):
    # List to store the extracted second numbers
    second_numbers = []

    # Regular expression pattern to extract the second number
    pattern = r'_(\d+)_H_fine'  # Matches the second number between _ and _H_fine

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('_H_fine.txt'):  # Ensure correct naming format
            match = re.search(pattern, filename)
            if match:
                second_numbers.append(match.group(1))  # Add second number to the list

    return second_numbers


if __name__ =="__main__":
    root_path = "/scratch/projects/fouheylab/dma9300/recon3d/data_old/billy/"
    save_pose = "camera_poses"
    os.makedirs(save_pose, exist_ok=True)
    pcd_path = f'/scratch/projects/fouheylab/dma9300/recon3d/masked_pcs/'
    trans_path = '/scratch/projects/fouheylab/dma9300/recon3d/camera_poses/'

    target_cams  = extract_second_numbers(trans_path)
# _to_035322250244_H_fine

    cam_series = valid_camera_series("/scratch/projects/fouheylab/dma9300/recon3d/data_old/billy/meta_data/")
    cam_list  = sorted(list(cam_series))
    source_cam = cam_list[0]
    # target_cams  = cam_list[1:]
    target_pts = o3d.io.read_point_cloud("/scratch/projects/fouheylab/dma9300/recon3d/masked_pcs/pcd-035322250028.ply")
    target_hmat = np.eye(4)
    source_input = [read_pcd_and_trans("035322250028", pcd_path, trans_path, tid) for tid in target_cams]

    source_pcd = [sinfo['pcd'] for sinfo in source_input]
    source_trans = [sinfo['trans'] for sinfo in source_input]

    # breakpoint()
    save_registration_result_color_multi(source_pcd, target_pts ,source_trans, f'output_masked_cloud.ply')