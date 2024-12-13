from einops import rearrange,repeat
from utils import *
import torch
import random
from PIL import Image
from io import BytesIO

TARGET_CHUNK_SIZE = int(4)
# cam_list = [cam for cam in cam_series.keys() if cam in camera_set[8]]

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
class SceneCamera:
    def __init__(self, cam_id, intrinsics, extrinsics):
        self.cam_id = cam_id    
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
    def get_extrinsics(self):
        return self.extrinsics
    def get_intrinsics(self):
        return self.intrinsics
    
    def export_re10k_pose(self):
        intrinsic_matrix = self.intrinsics
        pose = np.zeros(18)
        fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        pose[0:4] = [fx, fy, cx, cy]
        pose[6:] = rearrange(np.linalg.inv(self.extrinsics)[:3, :], 'h w -> (h w)', h=3, w=4)
        return pose.reshape(1, -1)
    
    def export_colmap_pose(self):
        # intrinsics & extrinsics
        fx, fy, cx, cy = self.intrinsics[0, 0], self.intrinsics[1, 1], self.intrinsics[0, 2], self.intrinsics[1, 2]
        intrs = [fx, fy, cx, cy]
        qvec = rotmat2qvec(self.extrinsics[:3, :3])
        tvec = self.extrinsics[:3, 3]
        extrins = np.concatenate([qvec, tvec], axis=0)
        return intrs, extrins
        
    
    def rotate_90(self, H):
        theta = np.pi / 2
        R_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        intr = self.intrinsics
        new_intr = np.eye(3)
        new_intr[0, 0] = intr[1, 1]
        new_intr[1, 1] = intr[0, 0]
        new_intr[0, 2] = H - intr[1, 2]
        new_intr[1, 2] = intr[0, 2]
        new_extr = R_mat @ self.extrinsics[:3, :]
        self.intrinsics = new_intr
        self.extrinsics[:3, :] = new_extr
    
    @staticmethod
    def from_raw_info(meta_path, trans_path):
        cam_info = CameraInfo(meta_path)
        intrinsics = cam_info.get_intrinsics()
        extrinsics = np.loadtxt(trans_path)
        cam_id = cam_info.get_cam_id()
        return SceneCamera(cam_id, intrinsics, extrinsics)

def convert_poses(poses):
    b, _ = poses.shape

    # Convert the intrinsics to a 3x3 normalized K matrix.
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=b)
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
    w2c = repeat(np.eye(4, dtype=np.float32), "h w -> b h w", b=b)
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return np.linalg.inv(w2c), intrinsics

class Scene:
    # a single frame scene containing multiple cameras
    def __init__(self, view_infos: List[FrameInfo], transform_collection: TransformCollection, target_camera = '1246',frame_id = '000000'):
        self.cameras = []
        self.images = []
        self.frame_id = str(frame_id).zfill(7)
        extrinsics_dict = transform_collection.get_transform_to_target(target_camera)
        for view in view_infos:
            cam_id = view.cam_info.get_cam_id()
            intrinsics = view.cam_info.get_intrinsics()
            extrinsics = extrinsics_dict[cam_id]
            self.cameras.append(SceneCamera(cam_id, intrinsics, extrinsics))
            self.images.append(view.color_img)
        self.rotated = False
        self.view_infos = view_infos
        self.trans_collection = transform_collection
    
    def get_HW(self):
        return self.images[0].shape[0], self.images[0].shape[1]
            
    def get_cam_list(self):
        return [cam.cam_id for cam in self.cameras]
    
    def get_cam(self, cam_id):
        cam_list = self.get_cam_list()
        return self.cameras[cam_list.index(cam_id)]
    
    def get_img(self, cam_id):
        cam_list = self.get_cam_list()
        return self.images[cam_list.index(cam_id)]
    
    def get_cam_idx(self, cam_id):
        cam_list = self.get_cam_list()
        return cam_list.index(cam_id)
    
    def rotate_90(self):
        if self.rotated:
            return
        self.images = [np.rot90(img, k=-1) for img in self.images]
        h = self.images[0].shape[0]
        for cam in self.cameras:
            cam.rotate_90(h)
        self.rotated = True
        
    def export_colmap_pose(self, cam_id_list):
        poses = []
        for cam in self.cameras:
            if cam.cam_id in cam_id_list:
                poses.append(cam.export_colmap_pose())
        return poses
        
            
    def export_re10k_poses(self):
        poses = []
        for cam in self.cameras:
            poses.append(cam.export_re10k_pose())
        poses = np.concatenate(poses, axis=0)
        return torch.from_numpy(poses)
    
    
    def export_re10k_images(self):
        img_tensor_list = []
        for img in self.images:
            img = Image.fromarray(img)
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            raw_bytes = img_bytes.getvalue()
            byte_tensor = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.uint8))
            img_tensor_list.append(byte_tensor)
        return img_tensor_list
    
    def export_re10k_data(self):
        data_dict = {}
        data_dict['cameras'] = self.export_re10k_poses()
        data_dict['images'] = self.export_re10k_images()
        data_dict['url'] = ''
        data_dict['timestampes'] = None
        data_dict['key'] = self.frame_id
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
    


def get_ds_sampler(all_cam_list, context_cam_list=['1246','0879','1040'], target_size = 3):
    target_cam_list = [cam for cam in all_cam_list if cam not in context_cam_list]
    random.shuffle(target_cam_list)
    target_cam = target_cam_list[0:target_size]
    
    context_cam_id = [all_cam_list.index(cam) for cam in context_cam_list]
    target_cam_id = [all_cam_list.index(cam) for cam in target_cam]
    
    context_cam_id.sort()
    target_cam_id.sort()
    
    sampler = {}
    sampler['context'] = context_cam_id
    sampler['target'] = target_cam_id
    return sampler

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

        
if __name__ == '__main__':
    trans_collection = TransformCollection()
    trans_collection.load_collection('data/cali_data/output_data')
    # frame_list = keyframe_list
    # scene_cam_list = [cam for cam in cam_series.keys() if cam in camera_set[7][:2] + camera_set[0][:2] + camera_set[1][:2]]
    # context_cam_list=['1246','0028','1318','1265']
    # --- get instantsplat data
    context_cam_list = []
    for i in range(8):
        context_cam_list.append(camera_set[i][-2])
    # context_cam_list = ['1246','0879','1040']
    view_list = FrameInfo.get_view_infos('data/billy_data', 1, context_cam_list)
    scene = Scene(view_list, trans_collection, '1246', frame_id = 1)
    scene.rotate_90()
    img_list = []
    for cam in context_cam_list:    
        img_list.append(scene.get_img(cam))
    save_colmap_data(scene, context_cam_list, 'data/colmap')
    # save_instantsplat_data(img_list, 'data/instantsplat', dataset_name = 'TT', scene_name = 'Barn',img_name_list=context_cam_list)
    # --end get instantsplat data
    
    # # --- load masked instantsplat data
    # masked_path = 'data/cali_data/masked_imgs/masked_images'
    # masked_list = os.listdir(masked_path)
    # masked_list.sort()
    # img_list = []
    # for masked_img in masked_list[:12]:
    #     masked_img_path = os.path.join(masked_path, masked_img)
    #     img = cv2.imread(masked_img_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img_list.append(img)
    # save_instantsplat_data(img_list, 'data/instantsplat', dataset_name = 'TT', scene_name = 'Barn')
    
    #---- end load masked instantsplat data
    #---- get re10k data
    # scene_list = []
    # for frame in frame_list:
    #     view_list = FrameInfo.get_view_infos('data/billy_data', frame, scene_cam_list)
    #     scene = Scene(view_list, trans_collection, '1246', frame_id = frame)
    #     # scene.rotate_90()
    #     scene_list.append(scene)
    # save_re10k_data(scene_list, scene_cam_list, 'data/re10k', sample_num=5)
