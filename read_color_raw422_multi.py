# read raw color image as YUV422

import numpy as np
import cv2
import sys
import os
import argparse

def generate_png_from_raw(raw_pth, height, width, png_pth, verbose = 0):
    raw_img = np.fromfile(raw_pth, dtype=np.uint8).reshape((height, width, 2))
    # Convert YUV422 to BGR
    bgr_img = cv2.cvtColor(raw_img, cv2.COLOR_YUV2BGR_YUY2)
    # Save output image
    cv2.imwrite(png_pth, bgr_img)
    if verbose:
        print(f"Processed {raw_file} -> {png_pth.split('/')[-1]}")
        
#usage python cali-test/read_color_raw422_multi.py --input_dir Zulu/Calibration/MarkMorris3.Calibration.ec.take.001 --output_dir cali-test/output-1 --id 0028 --frame 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read Intel RealSense raw color image as YUV422')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing raw color images')
    parser.add_argument('--width', default=1280, help='width of the color image')
    parser.add_argument('--height', default=800, help='height of the color image')
    parser.add_argument('--id', default=None, help='serial number of the camera')
    parser.add_argument('--frame',default=0, help='frame number of the camera, 0 for all')
    parser.add_argument('--type', default="COLOR", help='type of the camera')
    parser.add_argument('--verbose', action='store_true', help='dump debug information')
    parser.add_argument('--output_dir', type=str, default='output', help='Path to the output directory')

    # Parse arguments
    args = parser.parse_args()
    # List all files in the input directory
    files = os.listdir(args.input_dir)
    # Filter raw files with the given serial number of the camera
    files = [f for f in files if f[-3:] == "raw" and args.type in f]
    files_frame = [f.split('.')[1] for f in files]
    if int(args.frame) != 0:
        target_idx = [id for id,val in enumerate(files_frame) if int(val) == int(args.frame)]
        files = [files[id] for id in target_idx]
    if args.id is not None:
        id_list = str(args.id).split(",")
        files_id = [f.split('-')[0][-4:] for f in files]
        target_idx = [idx for idx, val in enumerate(files_id) if val in id_list]
        raw_files = [files[idx] for idx in target_idx]
    else:
        raw_files = files
    print(len(raw_files))
    # Process each raw file
    for raw_file in raw_files:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        input_path = os.path.join(args.input_dir, raw_file)
        output_file_name = os.path.splitext(raw_file)[0] + '.png'
        output_path = os.path.join(output_dir, output_file_name)

        # Read raw image data
        height = 800 if args.type == 'COLOR' else 720
        generate_png_from_raw(input_path, height, args.width, output_path,args.verbose)
        

    print("All images processed.")




