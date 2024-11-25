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


if __name__ =="__main__":
    trans_path = 'data/output_data_calib_pnpr2d2'
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
    
