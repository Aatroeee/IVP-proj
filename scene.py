from einops import rearrange,repeat
from utils import *
import torch
import random
from PIL import Image
from io import BytesIO

TARGET_CHUNK_SIZE = int(4)
# cam_list = [cam for cam in cam_series.keys() if cam in camera_set[8]]
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
            
    def get_cam_list(self):
        return [cam.cam_id for cam in self.cameras]
    
    def rotate_90(self):
        if self.rotated:
            return
        self.images = [np.rot90(img, k=-1) for img in self.images]
        h = self.images[0].shape[0]
        for cam in self.cameras:
            cam.rotate_90(h)
        self.rotated = True
        
            
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
    
    
        
if __name__ == '__main__':
    trans_collection = TransformCollection()
    trans_collection.load_collection('/scratch/projects/fouheylab/dma9300/recon3d/data/output_data')
    # frame_list = keyframe_list
    scene_cam_list = [cam for cam in cam_series.keys() if cam in camera_set[7] + camera_set[0] + camera_set[1]]
    # context_cam_list=['1246','0028','1318','1265']
    context_cam_list = ['1246','0879','1040']
    frame_list = [1]
    scene_list = []
    for frame in frame_list:
        view_list = FrameInfo.get_view_infos('/scratch/projects/fouheylab/dma9300/recon3d/data_old/billy', frame, scene_cam_list)
        scene = Scene(view_list, trans_collection, '1246', frame_id = frame)
        # scene.rotate_90()
        scene_list.append(scene)
    save_re10k_data(scene_list, scene_cam_list, '/scratch/projects/fouheylab/dma9300/recon3d/data/re10k', sample_num=5)
