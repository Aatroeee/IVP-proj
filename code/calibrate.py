import numpy as np
import cv2
import open3d as o3d
import os
import sys
import json
import argparse
import torch
import glob 
import re
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


from read_depth_and_build_pcd import read_depth_image, read_color_image, build_point_cloud_from_depth

def setup_maskrcnn():
    """
    Configure Mask R-CNN with Detectron2 for COCO-trained models.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set confidence threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)


predictor =  setup_maskrcnn()

def generate_image_mask(image_array, predictor):
    """
    Generate and return the mask for a single person in the image provided as a NumPy array.

    Args:
        image_array (np.ndarray): The input image as a NumPy array (H, W, C).
        predictor: Mask R-CNN predictor configured with Detectron2.

    Returns:
        np.ndarray: A binary mask (NumPy array) for the detected person.
    """
    if image_array is None or len(image_array.shape) != 3:
        raise ValueError("Invalid image input. Provide a valid 3-channel NumPy array.")

    # Perform inference
    outputs = predictor(image_array)

    # Extract the mask for the single person (class 0 in COCO)
    instances = outputs["instances"]
    human_indices = (instances.pred_classes == 0).nonzero(as_tuple=True)[0]

    # breakpoint()

    if len(human_indices) == 0:
        return None

    # Use the first detected human mask
    mask = instances.pred_masks[human_indices[0]].cpu().numpy()

    return mask

def apply_mask(image, mask):
    """
    Apply a binary mask to an image, keeping only the foreground.

    Args:
        image (np.ndarray): The input image (H, W, C).
        mask (np.ndarray): A binary mask (H, W) where the foreground is True and the background is False.

    Returns:
        np.ndarray: The image with the background masked out, keeping only the foreground.
    """
    # Apply the mask to the image (keeping the foreground)
    mask = mask.astype(np.uint8)*255
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image


# read data from raw and meta data
def read_data(root_path, cam_series_id,  need_depth = True, mask =True , near_clip=0.5, far_clip=3.0):
    meta_path = os.path.join(root_path, 'meta_data', f'{cam_series_id}-MODEL.json')
    frame_str = "0000001"
    depth_path = os.path.join(root_path, 'raw_data', frame_str, f'{cam_series_id}-DEPTH.{frame_str}.raw')
    color_path = depth_path.replace('DEPTH', 'COLOR')
    
    # read data related to color camera
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    color_intrinsics = meta['color_intrinsics']
    color_intrinsic_matrix = np.array(
        [[color_intrinsics['fx'], 0, color_intrinsics['ppx']],
            [0, color_intrinsics['fy'], color_intrinsics['ppy']],
            [0, 0, 1]])
    color_height = meta['color_intrinsics']['height']
    color_width = meta['color_intrinsics']['width']
    
    distortion_coeffs = [color_intrinsics['k1'],color_intrinsics['k2'],color_intrinsics['p1'],color_intrinsics['p2'],color_intrinsics['k3']]
    distortion_coeffs = np.array(distortion_coeffs)
    
    rgb_img = read_color_image(color_path, width=color_width, height=color_height)
    
    depth_to_color_warped_pixels = None
    depth_to_color_cam_xyz = None
    intrinsic_matrix = None
    depth = None
    depth_clipped = None
    depth_scale = None
    extrinsics = None
    
    # read data related to depth camera
    if need_depth:
        intrinsics = meta['depth_intrinsics']
        intrinsic_matrix = np.array([[intrinsics['fx'], 0, intrinsics['ppx']],
                                        [0, intrinsics['fy'], intrinsics['ppy']],
                                        [0, 0, 1]])
        depth = read_depth_image(depth_path)
        depth_scale = meta['depth_scale']
        camera_pts = build_point_cloud_from_depth(
                depth, intrinsic_matrix, depth_scale,
                near_clip=near_clip, far_clip=far_clip)
        depth_clipped = camera_pts[:, -1]
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
        
        # Apply the transform to depth xyz
        depth_xyz = camera_pts
        # Extend to homogeneous coordinates
        depth_xyz = np.hstack((depth_xyz, np.ones((depth_xyz.shape[0], 1))))
        # Apply extrinsics and transform to color camera space
        depth_to_color_cam_xyz = np.dot(depth_xyz, extrinsics.T)
        depth_to_color_cam_xyz = depth_to_color_cam_xyz[:, :3]
        
        depth_to_color_warped_pixels = np.dot(color_intrinsic_matrix, depth_to_color_cam_xyz.T).T
        depth_to_color_warped_pixels = depth_to_color_warped_pixels[:, :] / depth_to_color_warped_pixels[:, 2:]
        depth_to_color_warped_pixels = depth_to_color_warped_pixels[:, :2]
    
    if mask:
        image_mask = generate_image_mask(rgb_img, predictor)
        if image_mask is not None:
            image_mask = image_mask.astype("uint8") * 255
            # img_id = cam2id[cam_series_id[-4:]]
            # cam_mask_path = os.path.join(mask_path, f'mask_{img_id:02d}.png')
            # mask = cv2.imread(cam_mask_path, cv2.IMREAD_GRAYSCALE)
            image_mask = image_mask>0
        
  
    return dict(
        cam_id = cam_series_id,
        rgb_img = rgb_img,
        intrinsics = color_intrinsic_matrix,
        distortion = distortion_coeffs,
        height = color_height,
        width = color_width,
        
        depth_img = depth,
        depth_clipped = depth_clipped,
        depth_scale = depth_scale,
        depth_warp_pixels = depth_to_color_warped_pixels,
        depth_warp_xyz = depth_to_color_cam_xyz,
        depth_intrinsics = intrinsic_matrix,
        
        color_offset_extrinsics = extrinsics,
        mask = image_mask
    )


def build_point_cloud_from_depth_downsampled(info,                              
                                 downsample_step = 10,near_clip=0.5,far_clip=2, output_path = 'tmp'):
    '''Build point cloud from depth image.
    Args:
        depth_img: np.ndarray, depth image.
        depth_intrinsics_matrix: np.ndarray, 3x3 matrix of depth intrinsics, as specified in the meta data.
        depth_scale: float, depth scale, as specified in the meta data.

    Returns:
        xyz: np.ndarray, point cloud coordinates in the camera space.
    '''
    depth_img = info['depth_img']
    depth_intrinsics_matrix = info['depth_intrinsics']
    depth_scale = info['depth_scale']
    
    depth = depth_img.astype(np.float32)
    depth = depth * depth_scale

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
    
    offset_extrinsics = info['color_offset_extrinsics']
    color_intrinsic_matrix = info['intrinsics']

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
    

    image = info['rgb_img']
    img_mask = info['mask']
    valid_mask = (x_coords >= 0) & (x_coords < image.shape[1]) & (y_coords >= 0) & (y_coords < image.shape[0])
    # Combine valid_mask with img_mask
    if img_mask is not None:
        valid_mask = valid_mask & img_mask[y_coords, x_coords]

    colors = image[y_coords[valid_mask], x_coords[valid_mask]]
    
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(depth_to_color_cam_xyz[valid_mask])
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.)
    
    cam_id = info['cam_id']
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    o3d.io.write_point_cloud(os.path.join(output_path,f'pcd-{cam_id}.ply'), pcd)
    # breakpoint()
    # return pcd


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


def compute_camera_pose(source_image, target_image, K_source, K_target, plot_matches=False):
    """
    Compute the camera pose between a source and target camera using SIFT.
    
    Args:
        source_image (np.ndarray): Image from the source camera (grayscale).
        target_image (np.ndarray): Image from the target camera (grayscale).
        K_source (np.ndarray): Intrinsic matrix of the source camera.
        K_target (np.ndarray): Intrinsic matrix of the target camera.
        plot_matches (bool): Whether to plot matches between the two images.
    
    Returns:
        tuple: Relative rotation (R) and translation (T) between the source and target cameras.
    """
    # Initialize SIFT
    sift = cv2.SIFT_create()

    source_image_gray  = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    target_image_gray  = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    
    # Detect and compute keypoints and descriptors
    keypoints_source, descriptors_source = sift.detectAndCompute(source_image_gray, None)
    keypoints_target, descriptors_target = sift.detectAndCompute(target_image_gray, None)
    
    # Use BFMatcher to find matches between the descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_source, descriptors_target)
    
    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract points from matches
    points_source = np.float32([keypoints_source[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points_target = np.float32([keypoints_target[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute the Essential Matrix
    E, mask = cv2.findEssentialMat(points_source, points_target, K_source, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # draw_epipolar_lines(source_image, target_image, points_source, points_target, E)
    
    # Recover the pose (R and T) from the Essential Matrix
    _, R, T, mask_pose = cv2.recoverPose(E, points_source, points_target, K_source)
    
    # Plot matches if requested
    if plot_matches:
        # Draw the matches between the two images
        matched_image = cv2.drawMatches(
            source_image, keypoints_source, target_image, keypoints_target, matches[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Plot the matched image using Matplotlib
        plt.figure(figsize=(15, 7))
        plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        plt.title("Keypoint Matches")
        plt.axis("off")
        plt.savefig("vis_matches.png")
    
    return R, T

def transformation_matrix(R, t):
    """
    Compute the 4x4 transformation matrix [R | T; 0 0 0 1].

    Args:
        R (np.ndarray): Rotation matrix (3x3).
        t (np.ndarray): Translation vector (3x1 or 1x3).

    Returns:
        np.ndarray: Transformation matrix (4x4).
    """
    T = np.eye(4)  # Initialize a 4x4 identity matrix
    T[:3, :3] = R  # Set the rotation matrix in the top-left 3x3 block
    T[:3, 3] = t.reshape(3)  # Set the translation vector in the last column
    return T

def plot_image_and_mask(image, masked_image, title1="Image", title2="Masked Image", save_name = None):
    """
    Plot an image and its masked version side by side using subplots.

    Args:
        image (np.ndarray): Original image.
        masked_image (np.ndarray): Masked version of the image.
        title1 (str): Title for the first subplot (original image).
        title2 (str): Title for the second subplot (masked image).
    """
    plt.figure(figsize=(10, 5))

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(title1)
    plt.axis("off")

    # Plot masked image
    plt.subplot(1, 2, 2)
    plt.imshow(masked_image)
    plt.title(title2)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_name}.png")

def stitch_images(source_image, target_image, K_source, K_target, RT):
    """
    Stitch source and target images using a 4x4 homogeneous transformation matrix RT.

    Args:
        source_image (np.ndarray): The source image (HxWx3).
        target_image (np.ndarray): The target image (HxWx3).
        K_source (np.ndarray): Intrinsic matrix of the source camera.
        K_target (np.ndarray): Intrinsic matrix of the target camera.
        RT (np.ndarray): 4x4 transformation matrix from source to target.

    Returns:
        np.ndarray: The stitched image.
    """
    # Get target image dimensions
    height, width, _ = target_image.shape

    # Create a grid of pixel coordinates in the target image
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones_like(x)
    pixels_target = np.stack([x, y, ones], axis=-1).reshape(-1, 3).T  # Shape (3, num_pixels)

    # Convert pixel coordinates to normalized image coordinates in the target camera
    K_target_inv = np.linalg.inv(K_target)
    normalized_coords_target = K_target_inv @ pixels_target  # Shape (3, num_pixels)

    # Convert to homogeneous coordinates for 4x4 RT multiplication
    normalized_coords_target_h = np.vstack([normalized_coords_target, np.ones((1, normalized_coords_target.shape[1]))])  # Shape (4, num_pixels)

    # Apply the inverse transformation to map to the source camera
    transformed_coords_h = np.linalg.inv(RT) @ normalized_coords_target_h  # Shape (4, num_pixels)
    transformed_coords = transformed_coords_h[:3]  # Convert back to non-homogeneous coordinates
    transformed_coords /= transformed_coords[2]  # Normalize by z

    # Project back to pixel coordinates in the source image
    pixel_coords_source = K_source @ transformed_coords  # Shape (3, num_pixels)
    pixel_coords_source = pixel_coords_source[:2].T.reshape(height, width, 2)  # Shape (height, width, 2)

    # Interpolate source image using remap
    warped_source = cv2.remap(
        source_image,
        pixel_coords_source[..., 0].astype(np.float32),
        pixel_coords_source[..., 1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    # Combine target and warped source image into a stitched canvas
    stitched = np.maximum(target_image, warped_source)

    return stitched


# def draw_epipolar_lines(img1, img2, points1, points2, F):
#     """Draws corresponding epipolar lines on img1 and img2 using points1 and points2 with the fundamental matrix F.

#     Args:
#         img1 (ndarray): The first image.
#         img2 (ndarray): The second image.
#         points1 (ndarray): Points in the first image.
#         points2 (ndarray): Points in the second image.
#         F (ndarray): The fundamental matrix.
#     """
#     # Convert images to color for drawing lines
#     img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1
#     img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2

#     # Compute the epipolar lines in both images
#     lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
#     lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

#     # Generate a set of colors for each corresponding pair
#     colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(points1))]

#     # Draw lines and points on img1
#     for r, pt1, color in zip(lines1, points1.squeeze(), colors):
#         # breakpoint()
#         x0, y0 = map(int, [0, -r[2] / r[1]])
#         x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1]])
#         img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 5)
#         img1_color = cv2.circle(img1_color, (int(pt1[0]), int(pt1[1])), 6, tuple(color), 5)

#     # Draw lines and points on img2
#     for r, pt2, color in zip(lines2, points2.squeeze(), colors):
#         x0, y0 = map(int, [0, -r[2] / r[1]])
#         x1, y1 = map(int, [img2.shape[1], -(r[2] + r[0] * img2.shape[1]) / r[1]])
#         img2_color = cv2.line(img2_color, (x0, y0), (x1, y1), color, 5)
#         img2_color = cv2.circle(img2_color, (int(pt2[0]), int(pt2[1])), 6, tuple(color), 5)

#     # Plot both images on the same figure
#     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#     ax[0].imshow(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB))
#     ax[0].set_title('Image 1 with Epipolar Lines')
#     ax[0].axis('off')  # Hide axes

#     ax[1].imshow(cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))
#     ax[1].set_title('Image 2 with Epipolar Lines')
#     ax[1].axis('off')  # Hide axes

#     plt.tight_layout()
#     plt.savefig("epipole_lines.png")



def build_point_clouds(cam_list, root_path, output_path="/scratch/projects/fouheylab/dma9300/recon3d/masked_pcs/"):
    for cam_id in cam_list:
        info = read_data(root_path, cam_id, need_depth = True, mask = True, near_clip = 0.5, far_clip = 3.0)
        if info["mask"] is not  None:
            build_point_cloud_from_depth_downsampled(info, far_clip=2, output_path=output_path, downsample_step=1)
if __name__ == "__main__":
    root_path = "/scratch/projects/fouheylab/dma9300/recon3d/data_old/billy/"
    save_pose = "camera_poses"
    os.makedirs(save_pose, exist_ok=True)

    cam_series = valid_camera_series("/scratch/projects/fouheylab/dma9300/recon3d/data_old/billy/meta_data/")
    cam_list  = sorted(list(cam_series))
    # build_point_clouds(cam_list, root_path)

    source_cam_info = read_data(root_path, cam_list[0], need_depth = True, mask = True, near_clip = 0.5, far_clip = 3.0)
    target_cam_list = cam_list[1:]

    for cam_id  in target_cam_list:
        target_cam_info = read_data(root_path, cam_id,  need_depth = True, mask =True , near_clip=0.5, far_clip=3.0)
        if target_cam_info["mask"] is  not None:
            target_image  = apply_mask(target_cam_info["rgb_img"], target_cam_info["mask"])
            source_image =  apply_mask(source_cam_info["rgb_img"], source_cam_info["mask"])

            # plot_image_and_mask(source_cam_info["rgb_img"], source_image, save_name = f"source_cam_{cam_id}")
            # plot_image_and_mask(target_cam_info["rgb_img"], target_image, save_name = f"target_cam_{cam_id}")

            K_target  = target_cam_info["intrinsics"]
            K_source = source_cam_info["intrinsics"]
            R, t = compute_camera_pose(source_image, target_image, K_source, K_target)
            H_mat  = transformation_matrix(R, t)

            # stich_ouput = stitch_images(source_image, target_image, K_source, K_target, H_mat)
            # plt.imshow(stich_ouput)
            # plt.savefig(f'sttiched_image{cam_list[0]}_{cam_id}.png')
            # breakpoint()

            trans_fn = f'{cam_list[0]}_to_{cam_id}_H_fine.txt'
            with open(os.path.join(save_pose, trans_fn), 'w') as wb_f:
                np.savetxt(wb_f, H_mat)






