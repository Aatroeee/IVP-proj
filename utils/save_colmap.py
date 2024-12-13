import os
import cv2
import open3d as o3d
import numpy as np

from structure import *
from convert import *
def save_colmap_data(scene:Scene, context_cam_list, pcd, save_folder, enable_mask=True):
    img_path = os.path.join(save_folder, "images")
    os.makedirs(img_path, exist_ok=True)
    intrinsic_list = []
    extrinsic_list = []
    pcd_list = []
    h, w = scene.get_hw()
    for i, cam in enumerate(context_cam_list):
        img = scene.fetch_img(cam)
        img_name = f'{cam:0>6}.jpg'
        current_img_path = os.path.join(img_path, img_name)
        img_wb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if enable_mask:
            mask = scene.fetch_mask(cam).reshape(h, w, 1)
            img_wb = mask * img_wb
            cv2.imwrite(current_img_path, img_wb)
        else:
            cv2.imwrite(current_img_path, img_wb)
        
        scene_cam = scene.fetch_cam(cam)
        intrinsic, extrinsic = convert_colmap_pose(scene_cam.intrinsics, scene_cam.extrinsics)
        intrinsic_str = " ".join(map(str, intrinsic))
        extrinsic_str = " ".join(map(str, extrinsic))
        intrinsic_list.append(f"{i+1} PINHOLE {w} {h} {intrinsic_str}")
        extrinsic_list.append(f"{i+1} {extrinsic_str} {i+1} {img_name}")
        
        # cam_idx = scene.get_cam_idx(cam)
        # pcd = scene.views[cam_idx].get_pcd()
        # pcd_list.append(pcd)
    
      
    pose_path = os.path.join(save_folder, "sparse/0")
    os.makedirs(pose_path, exist_ok=True)
    with open(os.path.join(pose_path, "cameras.txt"), 'w') as f:
        f.write("\n".join(intrinsic_list))
    with open(os.path.join(pose_path, "images.txt"), 'w') as f:
        f.write("\n".join(extrinsic_list))
    pcd_path = os.path.join(pose_path, "points3D.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    
if __name__ == '__main__':
    trans_collection = TransformCollection()
    trans_collection.load_collection('data/cali_data/output_data')
    # context_cam_list = ['1246','0028','1318','1265']
    mask_root = 'data/billy_data/masks_data/0000001/masks'
    context_cam_list = []
    for i in range(8):
        context_cam_list.append(camera_set[i][-2])
        context_cam_list.append(camera_set[i][0])
    view_list = ViewFrame.get_view_infos('data/billy_data', 1, context_cam_list, mask_root)
    scene = Scene(view_list, trans_collection, '1246', frame_id = 1)
    # scene.rotate_90()
    pcd = scene.build_pcd(context_cam_list, near_clip=0.1, far_clip=2.0, downsample_step=None)
    # print(scene.cameras)
    save_colmap_data(scene, context_cam_list, pcd, 'data/colmap', enable_mask=True)