from transformers import pipeline
from PIL import Image
import argparse
import numpy as np
import cv2
import IPython
import sys
import json
import open3d as o3d
from read_depth_and_build_pcd import build_point_cloud_from_raw
from cam_settings import cam_series, camera_set
from cali import save_registration_result_color_multi, read_pcd_and_trans
# read mask
# estimate depth
# align depth
# generate aligned 
# register
UINT8_MAX = 255
def read_raw_depth(depth_path, width=1280, height=720, save_path=None):
    depth = np.fromfile(depth_path, dtype=np.uint16).reshape(height, width)
    if save_path is not None:
        write_depth(depth, save_path)
    return depth

def write_depth(depth, save_path):
    depth_img = depth.astype(np.float32) / np.max(depth) * 255
    depth_img = depth_img.astype(np.uint8)
    cv2.imwrite(save_path, depth_img)

def read_meta(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    color_intrinsics = meta['color_intrinsics']
    color_intrinsic_matrix = np.array(
        [[color_intrinsics['fx'], 0, color_intrinsics['ppx']],
            [0, color_intrinsics['fy'], color_intrinsics['ppy']],
            [0, 0, 1]])
    color_height = meta['color_intrinsics']['height']
    color_width = meta['color_intrinsics']['width']
    
    intrinsics = meta['depth_intrinsics']
    intrinsic_matrix = np.array([[intrinsics['fx'], 0, intrinsics['ppx']],
                                    [0, intrinsics['fy'], intrinsics['ppy']],
                                    [0, 0, 1]])
    depth_scale = meta['depth_scale']
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
    return dict(
        color_intrinsic_matrix=color_intrinsic_matrix,
        color_height=color_height,
        color_width=color_width,
        depth_intrinsic_matrix=intrinsic_matrix,
        depth_scale=depth_scale,
        color_offset_extrinsics=extrinsics
    )

def get_depth_warped_pixels(depth_img: np.ndarray, info_dict: dict):
    depth_intrinsics_matrix = info_dict['depth_intrinsic_matrix']
    color_intrinsic_matrix = info_dict['color_intrinsic_matrix']
    color_offset_extrinsics = info_dict['color_offset_extrinsics']
    
    depth = depth_img.astype(np.float32)

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
    xyz = xyz.astype(np.float32)
    
    depth_xyz = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
    # Apply extrinsics and transform to color camera space
    depth_to_color_cam_xyz = np.dot(depth_xyz, color_offset_extrinsics.T)
    depth_to_color_cam_xyz = depth_to_color_cam_xyz[:, :3]
    
    depth_to_color_warped_pixels = np.dot(color_intrinsic_matrix, depth_to_color_cam_xyz.T).T
    depth_to_color_warped_pixels = depth_to_color_warped_pixels[:, :] / depth_to_color_warped_pixels[:, 2:]
    depth_to_color_warped_pixels = depth_to_color_warped_pixels[:, :2]
    depth_to_color_warped_pixels = depth_to_color_warped_pixels.reshape(HEIGHT, WIDTH, 2)
    
    return depth_to_color_warped_pixels


def apply_colormask_to_depth(depth, mask, save_path=None):
    depth_img = depth.astype(np.float32) / np.max(depth) * 255
    depth_img = depth_img.astype(np.uint8)
    depth_img[mask == 0] = 0
    if save_path is not None:
        cv2.imwrite(save_path, depth_img)
    return depth_img

def get_align_info(original_depth, estimated_depth, mask = None):
    ho, wo = original_depth.shape
    original_depth_reshaped = original_depth.reshape(-1)  # (ho*wo)
    estimated_depth_reshaped = estimated_depth.reshape(-1)  # (ho*wo) 
    if mask is not None:
        mask_reshaped = mask.reshape(-1)  # (ho*wo)
    else:
        mask_reshaped = np.ones_like(original_depth_reshaped)
    
    valid_mask = mask_reshaped > 0.5  # (ho*wo)
    
    valid_original_depth = original_depth_reshaped[valid_mask]
    valid_estimated_depth = estimated_depth_reshaped[valid_mask] / UINT8_MAX
    
    # IPython.embed()
    # sys.exit()
    valid_original_depth_disp = valid_original_depth - np.median(valid_original_depth) + 1e-8
    valid_estimated_depth_disp = valid_estimated_depth - np.median(valid_estimated_depth) + 1e-8
    
    scale = np.median(valid_original_depth_disp / valid_estimated_depth_disp)
    shift = np.median(valid_original_depth_disp - scale * valid_estimated_depth_disp)
    
    # aligned_depth_disp = scale * valid_estimated_depth_disp + shift 
    # aligned_depth = aligned_depth_disp.reshape(ho, wo)
    # IPython.embed()
    # sys.exit()
    return dict(scale=scale, shift=shift, original_median = np.median(valid_original_depth), estimated_median = np.median(valid_estimated_depth))

def align_depth(estimated_depth, align_info):
    estimated_depth_reshaped = estimated_depth.reshape(-1) / UINT8_MAX  # (ho*wo)
    estimated_depth_disp = estimated_depth_reshaped - align_info['estimated_median'] + 1e-8
    aligned_depth_disp = align_info['scale'] * estimated_depth_disp + align_info['shift']
    aligned_depth = aligned_depth_disp.reshape(estimated_depth.shape) + align_info['original_median']
    return aligned_depth

def build_point_cloud_from_depth(depth, color_img, info_dict, mask = None):
    h,w = depth.shape
    intrinsics_matrix = info_dict['color_intrinsic_matrix']
    x = np.arange(0, w)
    y = np.arange(0, h)
    x, y = np.meshgrid(x, y)
    
    x = (x - intrinsics_matrix[0, 2]) * depth / intrinsics_matrix[0, 0]
    y = (y - intrinsics_matrix[1, 2]) * depth / intrinsics_matrix[1, 1]

    x = x.flatten()
    y = y.flatten()
    z = depth.flatten()
    xyz = np.vstack((x, y, z)).T
    if mask is not None:
        mask = mask.flatten()
        xyz = xyz[mask > 0.5]
    xyz = xyz.astype(np.float32)
    
    # IPython.embed()
    
    rgb = color_img.reshape(-1, 3)
    if mask is not None:
        rgb = rgb[mask > 0.5]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.)
    
    return pcd

def get_aligned_estimated_depth(meta_path, color_path, depth_path, mask_path, depth_estimation_pipe):
    info_dict = read_meta(meta_path)
    depth = read_raw_depth(depth_path, save_path=f'res/depth_raw_{series_id}.png')
    depth = depth * info_dict['depth_scale']
    depth_to_color_warped_pixels = get_depth_warped_pixels(depth, info_dict)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    warped_mask = cv2.remap(mask, depth_to_color_warped_pixels.astype(np.float32), None, cv2.INTER_NEAREST)
    warped_mask_bool = warped_mask > 128
    masked_depth = warped_mask_bool * depth
    
    image = Image.open(color_path)
    estimated_depth = depth_estimation_pipe(image)["depth"]
    estimated_depth = np.array(estimated_depth)
    write_depth(estimated_depth, save_path=f'res/estimated_depth_{series_id}.png')
    color_img = np.array(image)
    
    warped_estimated_depth = cv2.remap(estimated_depth, depth_to_color_warped_pixels.astype(np.float32), None, cv2.INTER_LINEAR)
    
    aligned_info = get_align_info(depth, warped_estimated_depth, warped_mask)
    aligned_depth = align_depth(estimated_depth, aligned_info)
    
    pcd = build_point_cloud_from_depth(aligned_depth, color_img, info_dict, mask)
    
    ori_pcd_masked = build_point_cloud_from_depth_downsampled(info_dict, depth, color_img, img_mask=mask)
    
    return dict(aligned_depth=aligned_depth,
                pcd=pcd,
                ori_pcd_masked=ori_pcd_masked)
    
def build_point_cloud_from_depth_downsampled(info_dict, depth_img, color_img,                             
                                 downsample_step = 1,near_clip=0.5,far_clip=3, img_mask=None):
    depth_intrinsics_matrix = info_dict['depth_intrinsic_matrix']
    depth_scale = info_dict['depth_scale']
    
    depth = depth_img.astype(np.float32)

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
    
    offset_extrinsics = info_dict['color_offset_extrinsics']
    color_intrinsic_matrix = info_dict['color_intrinsic_matrix']

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
    

    image = color_img
    valid_mask = (x_coords >= 0) & (x_coords < image.shape[1]) & (y_coords >= 0) & (y_coords < image.shape[0])
    # Combine valid_mask with img_mask
    img_mask = img_mask > 0.5
    if img_mask is not None:
        valid_mask = valid_mask & img_mask[y_coords, x_coords]

    colors = image[y_coords[valid_mask], x_coords[valid_mask]]
    
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(depth_to_color_cam_xyz[valid_mask])
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.)
    
    return pcd

if __name__ == "__main__":
    
    # cam_id = '1246'
    # series_id = '043422251246'
    frame_id = '0000001'
    target_id = '1246'
    # meta_path = f'data/billy_data/meta_data/{series_id}-MODEL.json'
    # color_path = f'data/billy_data/color_data/{frame_id}/{series_id}.png'
    # mask_path = f'data/billy_data/masks_data/{frame_id}/masks/{series_id}.png'
    depth_estimation_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf",device="cuda")
    # source_id = [cam_id for cam_id in list(cam_series.keys()) if cam_id not in camera_set[8]]
    # target_input = None
    # for sid in source_id:
    #     series_id = cam_series[sid]
    #     meta_path = f'data/billy_data/meta_data/{series_id}-MODEL.json'
    #     color_path = f'data/billy_data/color_data/{frame_id}/{series_id}.png'
    #     mask_path = f'data/billy_data/masks_data/{frame_id}/masks/{series_id}.png'
    #     depth_path = f'data/billy_data/raw_data/{frame_id}/{series_id}-DEPTH.{frame_id}.raw'
        
    #     aligned_dict = get_aligned_estimated_depth(meta_path, color_path, depth_path, mask_path, depth_estimation_pipe)
    #     o3d.io.write_point_cloud(f'res/pcds/pcd-{series_id}.ply', aligned_dict['pcd'])
    #     o3d.io.write_point_cloud(f'res/ori_pcds/pcd-{series_id}.ply', aligned_dict['ori_pcd_masked'])
    
    source_id = ['1265', '0028','1318']
    # source_id = [cam_id for cam_id in list(cam_series.keys()) if cam_id not in camera_set[8]]
    pcd_path = 'res/ori_pcds'
    trans_path = 'data/cali_data/output_data'
    target_input = read_pcd_and_trans(target_id, pcd_path, trans_path)
    source_input = [read_pcd_and_trans(sid, pcd_path, trans_path, target_id) for sid in source_id]
    source_pcd = [sinfo['pcd'] for sinfo in source_input]
    source_trans = [sinfo['trans'] for sinfo in source_input]
    save_registration_result_color_multi(source_pcd, target_input['pcd'],source_trans, f'res/combined_pcd_ori_4view_{target_id}.ply')
        
        
        
    
    # depth_path = f'data/billy_data/raw_data/{frame_id}/{series_id}-DEPTH.{frame_id}.raw'
    # info_dict = read_meta(meta_path)
    # save_path = f'res/depth_raw_{series_id}.png'
    # depth = read_raw_depth(depth_path, save_path=save_path)
    # depth = depth * info_dict['depth_scale']
    # depth_to_color_warped_pixels = get_depth_warped_pixels(depth, info_dict)
    
    
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # warped_mask = cv2.remap(mask, depth_to_color_warped_pixels.astype(np.float32), None, cv2.INTER_NEAREST)
    
    # cv2.imwrite(f'res/warped_mask_{series_id}.png', warped_mask)
    # warped_mask_bool = warped_mask > 128
    # masked_depth = warped_mask_bool * depth
    # write_depth(masked_depth, save_path=f'res/masked_depth_{series_id}.png')
    
    
    
    # pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf",device="cuda")
    # image = Image.open(color_path)
    # estimated_depth = pipe(image)["depth"]
    # estimated_depth = np.array(estimated_depth)
    # rgb_img = np.array(image)
    
    # warped_estimated_depth = cv2.remap(estimated_depth, depth_to_color_warped_pixels.astype(np.float32), None, cv2.INTER_LINEAR)
    # write_depth(warped_estimated_depth*warped_mask, save_path=f'res/warped_estimated_depth_{series_id}.png')
    
    # aligned_info = get_align_info(depth, warped_estimated_depth)
    # print(aligned_info)
    # aligned_depth = align_depth(estimated_depth, aligned_info)
    
    # # # write_depth(aligned_depth, save_path=f'res/aligned_depth_{series_id}.png')
    # # pcd_total_estimated = build_point_cloud_from_depth(estimated_depth, rgb_img, info_dict)
    # # o3d.io.write_point_cloud(f'res/pcd_total_estimated_{series_id}.ply', pcd_total_estimated)
    
    # pcd = build_point_cloud_from_depth(aligned_depth, rgb_img, info_dict)
    # o3d.io.write_point_cloud(f'res/pcd_total_aligned_{series_id}.ply', pcd)
    
    
    # ori_pcd = build_point_cloud_from_raw(depth_path, meta_path, near_clip=0.5, far_clip=5.0, verbose=False)
    # o3d.io.write_point_cloud(f'res/ori_pcd_{series_id}.ply', ori_pcd)
    
    # ori_pcd_masked = build_point_cloud_from_depth_downsampled(info_dict, depth, rgb_img, img_mask=mask)
    # o3d.io.write_point_cloud(f'res/ori_pcd_masked_{series_id}.ply', ori_pcd_masked)
