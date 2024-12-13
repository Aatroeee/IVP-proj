import numpy as np
from cali import read_data

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Build depth data from raw files")
    parser.add_argument("--input", type=str, required=True, help="Path to the input directory containing raw_data and meta_data")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--frame", type=int, required=True, help="Frame number to process")
    return parser.parse_args()
args = parse_args()


frame_str = str(args.frame).zfill(7)

def build_depth(input_path, output_path, frame_str):
    data = read_data(root_path=input_path, cam_series_id=frame_str, frame_id=frame_str, need_depth=True, clip_enable=False)
    color_img = data["rgb_img"]
    depth_img = data["depth_img"]
    color_intrinsics = data["intrinsics"]
    depth_intrinsics = data["depth_intrinsics"]
    depth_offset_extrinsics = data["depth_offset_extrinsics"]
    depth_scale = data["depth_scale"]
    
    depth = depth_img.astype(np.float32) * depth_scale
    height, width = depth_img.shape[:2]
    x = np.arange(0, width)
    y = np.arange(0, height)
    x, y = np.meshgrid(x, y)
    x = (x - depth_intrinsics[0, 2]) * depth / depth_intrinsics[0, 0]
    y = (y - depth_intrinsics[1, 2]) * depth / depth_intrinsics[1, 1]
    
    x = x.flatten()
    y = y.flatten()
    z = depth.flatten()
    
    xyz = np.vstack((x, y, z)).T
    xyz = xyz.astype(np.float32)
    
    return xyz
    
if __name__ == "__main__":
    build_depth(args.input, args.output, frame_str)