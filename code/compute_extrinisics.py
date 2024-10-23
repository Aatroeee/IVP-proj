import numpy as np
import cv2
import open3d as o3d
import os
import sys
import json
import argparse
import torch
import glob
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from baseline import *
from PIL import Image
import numpy as np
from r2d2.tools import common
from r2d2.tools.dataloader import norm_RGB
from r2d2.nets.patchnet import *
from r2d2.extract import NonMaxSuppression,extract_multiscale
from segment_anything import sam_model_registry, SamPredictor
from skimage.metrics import structural_similarity as ssim

# from mast3r.mast3r.model import AsymmetricMASt3R
# from mast3r.mast3r.fast_nn import fast_reciprocal_NNs

# import mast3r.mast3r.utils.path_to_dust3r
# from mast3r.dust3r.inference import inference
# from mast3r.dust3r.utils.image import load_images

SEGMENT_ANYTHING_WEIGHTS ="/scratch/projects/fouheylab/dma9300/recon3d/weights/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
class FeatureExtractor:
    def __init__(self, model_type, weights_path, model_name ="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", topk = 5000):
        self.model_type  = model_type
        self.weights_path = weights_path
        self.device = "cuda"
        self.model = None 
        self.topk = topk
        self.model_name = model_name
        self.load_model()


    def load_model(self):
        if self.model_type == "r2d2":
            self.model = self.load_r2d2().eval()
        else: 
            pass
        #     # self.model = AsymmetricMASt3R.from_pretrained(self.model_name).to(self.device)

    def extract_features(self, image):
        if self.model_type == "r2d2":
            # create the non-maxima detector
            detector = NonMaxSuppression(
                rel_thr = 0.70, 
                rep_thr = 0.70)

            img = Image.fromarray(image)
            W, H = img.size
            img = norm_RGB(img)[None].cuda()

            # extract keypoints/descriptors for a single image
            xys, desc, scores = extract_multiscale(self.model, img, detector,
                scale_f   = 2**0.25, 
                min_scale = 0, 
                max_scale = 1,
                min_size  = 256, 
                max_size  = 1024, 
                verbose = False)

            xys = xys.cpu().numpy()
            desc = desc.cpu().numpy()
            scores = scores.cpu().numpy()
            idxs = scores.argsort()[self.topk or None:]

            return xys[idxs], desc[idxs], scores[idxs]
        else:
            pass

    def load_r2d2(self):
        checkpoint = torch.load(self.weights_path)
        print("\n>> Creating net = " + checkpoint['net']) 
        net = eval(checkpoint['net'])
        nb_of_weights = common.model_size(net)
        print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

        # initialization
        weights = checkpoint['state_dict']
        net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
        return net.cuda()
    


class calibrator():
    def __init__(self,args):
        self.frame_id = None
        self.root_path = None 
        self.feature_extractor = args.feature_extractor
        if self.feature_extractor == "orb":
            self.feature_detector = cv2.ORB_create()
        elif self.feature_extractor == "sift":
           self.feature_detector = cv2.SIFT_create()
        elif self.feature_extractor =="r2d2":
            self.feature_detector = FeatureExtractor("r2d2",args.weights_path, topk = 2000)

        self.source_mat = {}
        self.correspondence_matcher  = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
 
        # self.sam_model = sam_model_registry[MODEL_TYPE](checkpoint=SEGMENT_ANYTHING_WEIGHTS)


    def save_visualization(self, target_image, warped_image, save_path):
        """
        Save the visualization of the warped image against the target image.

        Parameters:
        - target_image: The target image (numpy array).
        - warped_image: The warped image (numpy array).
        - save_path: Path where the visualization will be saved.
        """
        # Create a figure and axis
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot target image
        axes[0].imshow(target_image)
        axes[0].set_title('Target Image')
        axes[0].axis('off')  # Hide axes ticks

        # Plot warped image
        axes[1].imshow(warped_image)
        axes[1].set_title('Warped Image')
        axes[1].axis('off')  # Hide axes ticks

        # Save the visualization
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        return



    def visualize_difference(self, original_image, warped_image):
        """
        Visualize the difference between the original and warped images.

        Parameters:
        - original_image: The original image (numpy array).
        - warped_image: The warped image (numpy array).
        """

        difference_image = np.abs(original_image.astype(float) - warped_image.astype(float))

        # Create a figure
        plt.figure(figsize=(8, 6))

        # Plot the difference image
        im = plt.imshow(difference_image, cmap='bwr', vmin=0, vmax=255)
        plt.title('Difference Image')
        
        # Add a colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Difference Value')

        # Show the plot
        plt.axis('off')  # Hide axes ticks
        plt.tight_layout()
        plt.show()
        plt.savefig("difference_plt_warp.png")
        plt.close()


    def warp_image(self, img1, H, img2):
        """Warp img1 to align with img2 using homography H."""
        # Get the dimensions of img2
        height, width = img2.shape[:2]
        
        # Warp img1 to img2's perspective
        warped_image = cv2.warpPerspective(img1, H, (width, height))
        
        return warped_image


    def get_3d_points_from_matches(self, image_info, pts_2d):
        # breakpoint()
        tree = KDTree(image_info['depth_warp_pixels'])
        obj_points = []
        for new_point in pts_2d:
            distances, indices = tree.query(new_point, k=4)
            coor_3d = image_info['depth_warp_xyz'][indices]
            obj_points.append(self.weighted_sum(distances, coor_3d))
        obj_points = np.array(obj_points)
        return obj_points

    def weighted_sum(self, distances, coor_3d):
        dist = distances.reshape((-1,1))
        weights = 1 / (dist + 1e-8) 
        weighted_coor = np.sum(weights * coor_3d, axis=0)
        average_coor = weighted_coor / np.sum(weights)
        return average_coor

    def solve_essential_matrix_pnp(self, pts_source, pts_target, K):
        """
        Solve for the Essential matrix using the PnP solver in OpenCV.
        
        Parameters:
            pts_source (np.ndarray): 3D points in the source view (Nx3).
            pts_target (np.ndarray): 2D points in the target view (Nx2).
            K (np.ndarray): Camera intrinsic matrix (3x3).
        
        Returns:
            E (np.ndarray): Essential matrix (3x3).
            R (np.ndarray): Rotation matrix (3x3).
            t (np.ndarray): Translation vector (3x1).
        """
        # Convert points to appropriate format
        pts_source = np.array(pts_source, dtype=np.float32)
        pts_target = np.array(pts_target, dtype=np.float32)

        # Solve PnP to find the rotation (R) and translation (t)
        # success, rvec, tvec = cv2.solvePnP(pts_source, pts_target, K, None)

        success, rvec, tvec, _ = cv2.solvePnPRansac(pts_source, pts_target, K, None, flags=cv2.SOLVEPNP_SQPNP,
                                                            iterationsCount=10_000,
                                                            reprojectionError=5.0,
                                                            confidence=0.999)
        
        if not success:
            raise ValueError("PnP solution was not found")

        # Convert the rotation vector to a rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Compute the Essential matrix: E = t_x * R
        # where t_x is the skew-symmetric matrix of the translation vector t
        t = tvec.flatten()
        t_x = np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ])

        # Essential matrix
        E = t_x @ R

        return E, R, t

    def compute_metrics(self, original_image, warped_image):
    
        # Convert images to grayscale
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        warped_gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((original_gray - warped_gray) ** 2)

        # Calculate Structural Similarity Index (SSIM)
        ssim_value = ssim(original_gray, warped_gray)

        return mse, ssim_value

    def get_masked_image(self, image_info):
        mask = image_info['mask']
        rgb_image  = image_info['rgb_img']
        rgb_image[~mask] = 0 
        return rgb_image

    def detect_features(self,image):
        keypoints, descriptors = self.feature_detector.detectAndCompute(image, None)
        return  keypoints, descriptors

    def save_trans_dict(self,td, wb_path):
        if not os.path.exists(wb_path):
            os.mkdir(wb_path)
        target_id = td['target']
        trans_dict = td['trans_mat']
        for sid in trans_dict.keys():
            trans_mat = trans_dict[sid]
            trans_fn = f'{sid}_to_{target_id}_H_fine.txt'
            with open(os.path.join(wb_path, trans_fn), 'w') as wb_f:
                np.savetxt(wb_f, trans_mat)

   
    def filter_matches(self, matches, ratio_thresh=0.70):
        # Function to filter matches using Lowe's ratio test
        good_matches = []
        for m in matches:
            print(m.distance)
            if m.distance < ratio_thresh * m.distance:
                good_matches.append(m)
        return good_matches

    def filter_matches_ratio(self, matches, descriptors1, descriptors2, ratio_thresh=0.20):
        good_matches = []
        for m in matches:
            # Get the distances of the matched descriptors
            d1 = descriptors1[m.queryIdx]
            d2 = descriptors2[m.trainIdx]

            # Calculate the distances to all matches for the current descriptor
            distances = np.linalg.norm(d1 - descriptors1, axis=1)

            # Find the nearest and second nearest distances
            sorted_indices = np.argsort(distances)
            closest_distance = distances[sorted_indices[0]]
            second_closest_distance = distances[sorted_indices[1]]

            # Apply the ratio test
            if closest_distance < ratio_thresh * second_closest_distance:
                good_matches.append(m)
        return good_matches

    def convert_to_cv2_keypoints(self, keypoints):
        return [cv2.KeyPoint(x=kp[0], y=kp[1], size=kp[2]) for kp in keypoints]

    def draw_matches_lines(self, img1, keypoints1, img2, keypoints2, matches):
        # Create a blank canvas to combine the two images horizontally
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        combined_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype='uint8')
        combined_image[:h1, :w1, :] = img1
        combined_image[:h2, w1:w1 + w2, :] = img2

        # Draw lines between matching keypoints
        for match in matches:
            # Get keypoint coordinates
            pt1 = tuple(np.round(keypoints1[match.queryIdx].pt).astype(int))
            pt2 = tuple(np.round(keypoints2[match.trainIdx].pt).astype(int))

            # Adjust the second point's x-coordinate to match the horizontal stacking
            pt2 = (pt2[0] + w1, pt2[1])

            # Draw a red line (BGR color for red is (0, 0, 255))
            cv2.line(combined_image, pt1, pt2, (0, 0, 255), 2)

        return combined_image

    
    def compute_camera_extrinisics(self, root_path, target_cam, source_cams,frame_id, args):
        self.root_path = root_path 
        self.frame_id = str(frame_id).zfill(7)
        self.target_cam = target_cam
        self.target_info = read_data(self.root_path, self.target_cam, self.frame_id, True, mask_path="/scratch/projects/fouheylab/dma9300/recon3d/data/masked_imgs/masks")
        self.target_image  = self.target_info["rgb_img"]
        self.source_cams = source_cams
        self.vis_flag = args.vis_flag
        self.args = args

        print(f"compute extriniscs for frame index ::{self.frame_id}") 
        if self.feature_extractor == "orb" or self.feature_extractor ==  "sift":
            target_img = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)
            keypoints_target, descriptors_target = self.feature_detector.detectAndCompute(target_img, None)
        elif self.feature_extractor == "r2d2":
            keypoints_target, descriptors_target, scores  = self.feature_detector.extract_features(self.target_image)
     
        for s_cam in self.source_cams:
            # Load image
            source_info = read_data(self.root_path, s_cam, self.frame_id, True, mask_path="/scratch/projects/fouheylab/dma9300/recon3d/data/masked_imgs/masks")
            # Detect keypoints and descriptors in the source image
            if self.feature_extractor == "orb" or self.feature_extractor == "sift":
                source_img = cv2.cvtColor(source_info['rgb_img'], cv2.COLOR_BGR2GRAY)
                keypoints_source, descriptors_source = self.feature_detector.detectAndCompute(source_img, None)

            elif self.feature_extractor == "r2d2":
                keypoints_source, descriptors_source, scores  = self.feature_detector.extract_features(source_info['rgb_img'])

            # Match descriptors between the source and target images
            matches = self.correspondence_matcher.match(descriptors_source, descriptors_target)
        
            # Sort matches by distance (best matches first)
            matches = sorted(matches, key=lambda x: x.distance)

            numGoodMatches = int(len(matches) * 0.75)
            
            matches = matches[:numGoodMatches]
        

            # Extract the matched points in both images
            if self.feature_extractor == "orb" or self.feature_extractor == "sift":

                pts_source = np.float32([keypoints_source[m.queryIdx].pt for m in matches]).reshape(-1, 2)
                pts_target = np.float32([keypoints_target[m.trainIdx].pt for m in matches]).reshape(-1, 2)
                if self.vis_flag:
                    matched_image = cv2.drawMatches(source_info['rgb_img'], keypoints_source, self.target_image, keypoints_target, matches, None)
                    plt.imshow(matched_image)
                    plt.axis("off")
                    plt.savefig("matched_sample_orb.png")
            elif self.feature_extractor == "r2d2":
            
                keypoints_source_t = self.convert_to_cv2_keypoints(keypoints_source)
                keypoints_target_t = self.convert_to_cv2_keypoints(keypoints_target)
                pts_source = np.float32([keypoints_source_t[m.queryIdx].pt for m in matches]).reshape(-1, 2)
                pts_target = np.float32([keypoints_target_t[m.trainIdx].pt for m in matches]).reshape(-1, 2)
                if self.vis_flag:
                    matched_image = cv2.drawMatches(source_info['rgb_img'], keypoints_source_t, self.target_image, keypoints_target_t, matches, None)
                    plt.imshow(matched_image)
                    plt.axis("off")
                    plt.savefig("matched_sample_r2d2.png")

            

            H_matrix , _ = cv2.findHomography(pts_source, pts_target, cv2.RANSAC)
            warped_image  = self.warp_image(source_info["rgb_img"],H_matrix, self.target_info["rgb_img"])
            mse, ssim = self.compute_metrics(self.target_image, warped_image)
            if self.vis_flag:
                print(f"MSE and SSIM scores are {mse:.3f} and {ssim:.3f}")
                self.save_visualization(self.target_info["rgb_img"],warped_image, "warped_vis.png")
                self.visualize_difference(self.target_info["rgb_img"], source_info["rgb_img"])

 	        # Calculates an essential matrix from the corresponding points in two images from potentially two different cameras.
            # breakpoint()
            #3D points
            if self.args.pnp:
                pts_source =  self.get_3d_points_from_matches(source_info, pts_source)
                E, R, t= self.solve_essential_matrix_pnp(pts_source, pts_target, self.target_info['intrinsics'])
                homo_mat = np.eye(4)
                homo_mat[:3, :3] = R
                homo_mat[:3,3] = t
                self.source_mat[s_cam[-4:]] = np.linalg.inv(homo_mat)

            else:

                E, mask = cv2.findEssentialMat(pts_source, pts_target,  source_info['intrinsics'],source_info['distortion'],self.target_info['intrinsics'],self.target_info['distortion'])

                # Recover the relative rotation (R) and translation (t)
                _, R, t, mask = cv2.recoverPose(E, pts_source, pts_target, source_info['intrinsics'])

                # Alternatively, manually decompose the essential matrix into possible R, t
                U, _, Vt = np.linalg.svd(E)
                W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

                # Two possible rotations
                R1 = np.dot(U, np.dot(W, Vt))
                R2 = np.dot(U, np.dot(W.T, Vt))

                # Two possible translations (up to scale)
                t1 = U[:, 2]
                t2 = -U[:, 2]

                # Now check all four configurations:
                # (R1, t1), (R1, t2), (R2, t1), (R2, t2)

                # Triangulate points for each case and check depths

                max_valid_depths = 0  # Initialize to keep track of the maximum number of valid depths
                best_R = None
                best_T = None

                P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera projection matrix
                for R, t in [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]:
                    P2 = np.hstack((R, t.reshape(-1, 1)))  # Second camera projection matrix
                    # breakpoint()
                    points4D = cv2.triangulatePoints(P1, P2, pts_source.T, pts_target.T)
                    
                    # Convert to Euclidean coordinates
                    points3D = points4D[:3] / points4D[3]
                    
                    # Check if points are in front of both cameras (positive Z)
                    valid_depths = np.sum(points3D[2, :] > 0)
                    print(f"Valid depths: {valid_depths}")

                    if valid_depths > max_valid_depths:
        
                        max_valid_depths = valid_depths
                        best_R = R  # Update best rotation matrix
                        best_T = t

                # After determining the best rotation matrix, save it as a .npy file
                if best_R is not None :
                
                    homo_mat = np.eye(4)
                    # note: camera to world(1246)
                    homo_mat[:3, :3] = best_R
                    homo_mat[:3,3] = best_T
                
                    self.source_mat[s_cam[-4:]] = np.linalg.inv(homo_mat)

                else:
                    print('No best rotation matrix!')
        trans_dict = dict(
                    target = self.target_cam[-4:],
                    trans_mat = self.source_mat
                )
        return trans_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Estimate camera extrinsics")
    parser.add_argument("--feature_extractor", type=str, required=True, help='type of kpts and desc extractor')
    parser.add_argument("--vis_flag", type=bool, default =False, help='log visualizations')
    parser.add_argument("--pnp", type=bool, default =False, help='pnp solver')
    parser.add_argument("--weights_path", type=str, default='/scratch/projects/fouheylab/dma9300/recon3d/code/r2d2/models/faster2d2_WASF_N8_big.pt', help='weights of pre-trained model')
    
    args = parser.parse_args()

    calibrate = calibrator(args = args)

    cali_sequence = []
    for i in range(8):
        cali_sequence.append(
            dict(
                target = camera_set[i][-2],
                frame = keyframe_list[i],
                source = camera_set[i] + camera_set[(i+1) % 8] + camera_set[(i-1) % 8]
            )
        )
    root_path = 'data'
    output_path = f'data/output_data_calib_{args.feature_extractor}'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for cali_set in cali_sequence:
        source_series = [cam_series[i] for i in cali_set['source']]
        target_serie = cam_series[cali_set['target']]
        trans_dict =  calibrate.compute_camera_extrinisics(root_path, target_serie, source_series, frame_id=cali_set['frame'], args=args)
        calibrate.save_trans_dict(trans_dict, output_path)



    








