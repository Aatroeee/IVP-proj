# create camera.json given transformation
import numpy as np
import json
import os
import sys
import glob
import typing as T
import shutil
import cv2
import argparse
import open3d as o3d
from cam_settings import cam_series,camera_set
from read_color_raw422_multi import generate_png_from_raw

# calculate fov from intrinsic matrix
def calculate_fov(intrinsic_matrix, image_width, image_height):
    # Extract focal lengths from the intrinsic matrix
    f_x = intrinsic_matrix[0, 0]
    f_y = intrinsic_matrix[1, 1]
    
    # Calculate the horizontal FOV
    fov_x = 2 * np.arctan(image_width / (2 * f_x)) * (180 / np.pi)
    
    # Calculate the vertical FOV
    fov_y = 2 * np.arctan(image_height / (2 * f_y)) * (180 / np.pi)
    
    return fov_x, fov_y

def generate_camera_json(
        meta_path='./meta_data/043422251246-MODEL.json',
        transform_path: T.Optional[str] = None):
    
    if transform_path is None:
        extrinsics = np.eye(4)
    else:
        extrinsics = np.loadtxt(transform_path, delimiter=' ', dtype=np.float32)

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    intrinsics = meta['color_intrinsics']
    intrinsic_matrix = np.array([[intrinsics['fx'], 0, intrinsics['ppx']],
                                [0, intrinsics['fy'], intrinsics['ppy']],
                                [0, 0, 1]])
    intrinsic_matrix = intrinsic_matrix.astype(np.float32)
    H_c2w = extrinsics

    fov_x, fov_y = calculate_fov(intrinsic_matrix, intrinsics['width'], intrinsics['height'])
    return H_c2w, intrinsic_matrix, fov_x, fov_y

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='3dgs format generation')
    parser.add_argument('--input', type=str, default='cali1')
    parser.add_argument('--output', type=str, default='cali-train')
    parser.add_argument('--pcd', type=str, default='combined_pcd_1246.ply')
    parser.add_argument('--frame', type=int, default=50)
    args = parser.parse_args()
    
    file_list = []
    root_path = args.input
    frame_id = args.frame
    current_frame = str(frame_id).zfill(7)
    raw_path = os.path.join(root_path,'raw_data',f'{current_frame}')
    output_path = args.output
    pcd_path = args.pcd
    pcd = o3d.io.read_point_cloud(pcd_path)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    train_output_path = os.path.join(output_path,'train')
    test_output_path = os.path.join(output_path,'test')
    os.mkdir(train_output_path)
    pcd = o3d.io.read_point_cloud(pcd_path)
    o3d.io.write_point_cloud(os.path.join(output_path,'pcd_0.ply'), pcd)
    
    meta_path = os.path.join(root_path, 'meta_data')
    trans_pth = os.path.join(root_path, 'output_data')

    meta_list = os.listdir(meta_path)
    meta_list = [f for f in meta_list if 'MODEL.json' in f]
    for meta_data in meta_list:
        serie_id = meta_data.split('-')[0]
        cam_id = serie_id[-4:]
        id_trans_pth = os.path.join(trans_pth, f'{cam_id}_to_1246_H_fine.txt')
        if os.path.exists(id_trans_pth) and cam_id not in camera_set[8]:
            file_list.append((os.path.join(meta_path, meta_data), id_trans_pth, serie_id))
        if cam_id == '1246':
            file_list.append((os.path.join(meta_path, meta_data), None, serie_id))
    H_c2w_list = []
    intrinsic_matrix_list = []
    for i in range(len(file_list)):
        meta_path, transform_path, _ = file_list[i]
        H_c2w, intrinsic_matrix, fov_x, fov_y = generate_camera_json(
            meta_path, transform_path)
        H_c2w_list.append(H_c2w)
        intrinsic_matrix_list.append(intrinsic_matrix)

    # Sort cameras starting from 1246, gradually adding the nearest ones
    
    camera_dict = {
        "H_c2w": [
            [
                x.tolist() for x in H_c2w_list
            ]
        ],
        "intrinsic": [
            [
                x.tolist() for x in intrinsic_matrix_list
            ]
        ],
        "width_px": 1280,
        "height_px": 800,
        "fov_x_deg": fov_y,
        "fov_y_deg": fov_y,
    }

    # dump
    camera_json_path = os.path.join(train_output_path, 'camera.json')
    print(camera_json_path)
    with open(camera_json_path, 'w') as f:
        json.dump(camera_dict, f, indent=4)

    # create ground truth views
    def process_img(img_path):
        img = cv2.imread(img_path)
        # pad to 1280x1280
        pad_h = (1280 - img.shape[0]) // 2
        pad_w = (1280 - img.shape[1]) // 2
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        return padded

    train_input_path = os.path.join(train_output_path, 'input')
    os.mkdir(train_input_path)
    for i in range(len(file_list)):
        meta_fn, _, serie_id = file_list[i]
        camera_id = meta_fn.split('/')[-1].split('-')[0]
        color_view = os.path.join(train_input_path, f'rgb_{i}.png')
        generate_png_from_raw(os.path.join(raw_path, f'{serie_id}-COLOR.{current_frame}.raw'),\
            800,1280, color_view)
    
    shutil.copytree(train_output_path, test_output_path)