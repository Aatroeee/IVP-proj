from einops import rearrange,repeat
from structure import *
import torch
import random
from PIL import Image
from io import BytesIO

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def convert_colmap_pose(intrinsics, extrinsics):
    # intrinsics & extrinsics
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    intrs = [fx, fy, cx, cy]
    qvec = rotmat2qvec(extrinsics[:3, :3])
    tvec = extrinsics[:3, 3]
    extrins = np.concatenate([qvec, tvec], axis=0)
    return intrs, extrins

def convert_re10k_pose(intrinsics, extrinsics):
    pose = np.zeros(18)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    pose[0:4] = [fx, fy, cx, cy]
    pose[6:] = rearrange(np.linalg.inv(extrinsics)[:3, :], 'h w -> (h w)', h=3, w=4)
    return pose.reshape(1, -1)

def rotate_90(intrinsics, extrinsics, H):
    theta = np.pi / 2
    R_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    intr = intrinsics
    new_intr = np.eye(3)
    new_intr[0, 0] = intr[1, 1]
    new_intr[1, 1] = intr[0, 0]
    new_intr[0, 2] = H - intr[1, 2]
    new_intr[1, 2] = intr[0, 2]
    new_extr = R_mat @ extrinsics[:3, :]
    return new_intr, new_extr

# ------------- Re10k -----------------
def export_re10k_poses_from_scene(scene: Scene):
    poses = []
    for cam in scene.cameras:
        poses.append(cam.export_re10k_pose())
    poses = np.concatenate(poses, axis=0)
    return torch.from_numpy(poses)

def export_re10k_images_from_scene(scene: Scene):
    img_tensor_list = []
    for img in scene.images:
        img = Image.fromarray(img)
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        raw_bytes = img_bytes.getvalue()
        byte_tensor = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.uint8))
        img_tensor_list.append(byte_tensor)
    return img_tensor_list

def export_re10k_data_from_scene(scene: Scene):
    data_dict = {}
    data_dict['cameras'] = export_re10k_poses_from_scene(scene)
    data_dict['images'] = export_re10k_images_from_scene(scene)
    data_dict['url'] = ''
    data_dict['timestampes'] = None
    data_dict['key'] = scene.frame_id
    return data_dict

def save_re10k_data(scene_list:List[Scene], scene_cam_list, save_folder,  sample_num = 3):
    key_list = []
    chunk_id = 0
    chunk_list = []
    chunk_index = {}
    fn = f"{chunk_id:0>6}.torch"
    
    sample_index = {}
    for scene in scene_list:
        if scene.frame_id not in chunk_index.keys():
            chunk_list.append(scene.export_re10k_data())
            chunk_index[scene.frame_id] = fn
        if len(chunk_list) >= TARGET_CHUNK_SIZE:
            torch.save(chunk_list, os.path.join(save_folder, fn))
            chunk_id += 1
            fn = f"{chunk_id:0>6}.torch"
            chunk_list = []
        sample_index[scene.frame_id] = get_ds_sampler(scene_cam_list,target_size=sample_num)
    if len(chunk_list) > 0:
        torch.save(chunk_list, os.path.join(save_folder, fn))
    with open(os.path.join(save_folder, 'index.json'), 'w') as f:
        json.dump(chunk_index, f)
    
    with open(os.path.join(save_folder, 'evaluation_index.json'), 'w') as f:
        json.dump(sample_index, f)


# ------------- Instantsplat -----------------
def save_instantsplat_data(img_list, save_folder, dataset_name = 'TT', scene_name = 'Family', img_name_list = None):
    if img_name_list is None:
        img_name_list = [i for i in range(len(img_list))]
    view_num = len(img_list)
    img_folder = os.path.join(save_folder, dataset_name, scene_name, f'{view_num}_views', 'images')
    os.makedirs(img_folder, exist_ok=True)
    for i, img in enumerate(img_list):
        img_name = f'{img_name_list[i]:0>6}.jpg'
        img_path = os.path.join(img_folder, img_name)
        img_wb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img_wb)
    print(f'save {view_num} images to {img_folder}')
    

# ------------- colmap -----------------
def save_colmap_data(scene:Scene, context_cam_list, save_folder):
    img_path = os.path.join(save_folder, "images")
    os.makedirs(img_path, exist_ok=True)
    intrinsic_list = []
    extrinsic_list = []
    pcd_list = []
    h, w = scene.get_HW()
    for i, cam in enumerate(context_cam_list):
        img = scene.get_img(cam)
        img_name = f'{cam:0>6}.jpg'
        current_img_path = os.path.join(img_path, img_name)
        img_wb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(current_img_path, img_wb)
        
        intrinsic, extrinsic = scene.get_cam(cam).export_colmap_pose()
        intrinsic_str = " ".join(map(str, intrinsic))
        extrinsic_str = " ".join(map(str, extrinsic))
        intrinsic_list.append(f"{i+1} PINHOLE {w} {h} {intrinsic_str}")
        extrinsic_list.append(f"{i+1} {extrinsic_str} {i+1} {img_name}")
        
        cam_idx = scene.get_cam_idx(cam)
        pcd = scene.view_infos[cam_idx].get_pcd()
        pcd_list.append(pcd)
    
    combined_pcd = merge_pcd(scene.trans_collection, pcd_list, context_cam_list, '1246')
        
    pose_path = os.path.join(save_folder, "sparse/0")
    os.makedirs(pose_path, exist_ok=True)
    with open(os.path.join(pose_path, "cameras.txt"), 'w') as f:
        f.write("\n".join(intrinsic_list))
    with open(os.path.join(pose_path, "images.txt"), 'w') as f:
        f.write("\n".join(extrinsic_list))
    pcd_path = os.path.join(pose_path, "points3D.txt")
    o3d.io.write_point_cloud(pcd_path, combined_pcd)