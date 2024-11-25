import os
import glob
import cv2
import torch
from pathlib import Path
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

def read_color_image(color_path, width=1280, height=800,
                     verbose=False):
    '''Read color image from raw file and convert to RGB.
    Args:
        color_path: str, path to the raw color image.
        width: int, width of the color image.
        height: int, height of the color image.'''
    raw_img = np.fromfile(color_path, dtype=np.uint8).reshape(height, width, 2)
    bgr_img = cv2.cvtColor(raw_img, cv2.COLOR_YUV2BGR_YUY2)
    if verbose:
        print('[Verbose] Save color image to color_img.png.')
        cv2.imwrite('color_img.png', bgr_img)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img

def read_depth_image(depth_path, width=1280, height=720,
                     verbose=False):
    '''Read depth image from raw file.
    Args:
        depth_path: str, path to the raw depth image.
        width: int, width of the depth image.
        height: int, height of the depth image.'''
    depth = np.fromfile(depth_path, dtype=np.uint16).reshape(height, width)

    # Visualize depth image if verbose
    if verbose:
        print('[Verbose] Save depth image to depth_img.png.')
        depth_img = depth.astype(np.float32) / np.max(depth) * 255
        depth_img = depth_img.astype(np.uint8)
        depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
        cv2.imwrite('depth_img.png', depth_img)
    return depth



# def read_rgb_images_and_depth_from_raw_rgbd_files(path):


def load_images_from_subfolders(directory_path):
    """
    Load all images from subfolders.
    """
    image_paths = glob.glob(os.path.join(directory_path, "**", "*.*"), recursive=True)
    supported_formats = (".png")
    return [img_path for img_path in image_paths if img_path.lower().endswith(supported_formats)]

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

def generate_human_masks(image_path, predictor):
    """
    Generate human masks using Mask R-CNN for the given image.
    """
    image = cv2.imread(image_path)
    outputs = predictor(image)

    # Extract human masks (class 0 in COCO)
    human_masks = []
    instances = outputs["instances"]
    human_indices = (instances.pred_classes == 0).nonzero(as_tuple=True)[0]
    
    for idx in human_indices:
        mask = instances.pred_masks[idx].cpu().numpy()
        human_masks.append(mask)

    return human_masks

def save_masks_with_structure(input_path, masks, input_dir, output_dir):
    """
    Save the masks with the same folder structure as the input directory.
    """
    # Create a corresponding path in the output directory
    relative_path = os.path.relpath(input_path, input_dir)
    output_folder = os.path.join(output_dir, os.path.dirname(relative_path))
    os.makedirs(output_folder, exist_ok=True)

    # Save masks
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    for i, mask in enumerate(masks):
        mask_filename = os.path.join(output_folder, f"{base_name}_mask.png")
        # breakpoint()
        cv2.imwrite(mask_filename, mask.astype("uint8") * 255)

def main():
    input_directory = "/scratch/projects/fouheylab/dma9300/recon3d/data_old/cali_1/img_data/"
    output_directory = "/scratch/projects/fouheylab/dma9300/recon3d/masks/"
    os.makedirs(output_directory, exist_ok=True)

    # Load images
    image_paths = load_images_from_subfolders(input_directory)

    # Setup Mask R-CNN
    predictor = setup_maskrcnn()

    # Process each image
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        masks = generate_human_masks(image_path, predictor)
        if masks:
            save_masks_with_structure(image_path, masks, input_directory, output_directory)

if __name__ == "__main__":
    main()
