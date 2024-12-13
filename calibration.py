from utils.structure import *

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='data/cali_data')
    parser.add_argument('--task', type=str, default='cali')
    parser.add_argument('--trans_path', type=str, default='data/cali_data/trans_icp_data4')
    parser.add_argument('--pcd_path', type=str, default='billy.ply')
    parser.add_argument('--mask_path', type=str, default='billy_data/masks_data/0000001/masks')
    parser.add_argument('--frame_id', type=str, default='50')
    parser.add_argument('--near_clip', type=float, default=0.1)
    parser.add_argument('--far_clip', type=float, default=2.0)
    parser.add_argument('--downsample_step', type=int, default=None)
    parser.add_argument('--refine', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    root_path = args.root_path
    task = args.task
    trans_path = args.trans_path
    pcd_path = os.path.join(root_path, args.pcd_path)
    cam_id_list = [cam_id for cam_id in list(cam_series.keys()) if cam_id not in camera_set[8]]
    cam_series_list = [cam_series[cam_id] for cam_id in cam_id_list]
    cam_info_dict = {cam_id: CameraInfo(get_meta_path(root_path, cam_series[cam_id])) for cam_id in cam_id_list}
    
    target_id = '1246'
    
    if task == 'cali':
        frame_info_dict = {} # (cam_id, frame_id) -> Frame
        for i in range(len(keyframe_list)):
            frame_id = str(keyframe_list[i]).zfill(7)
            for cam_id in cam_id_list:
                frame_info = Frame(get_color_raw_path(root_path, frame_id, cam_series[cam_id]), cam_info_dict[cam_id])
                frame_info_dict[(cam_id, frame_id)] = frame_info
        cali_sequence = []  
        for i in range(8):
            cali_sequence.append(
                dict(
                    target = camera_set[i][-2],
                    frame = keyframe_list[i],
                    source = camera_set[i] + camera_set[(i+1) % 8] + camera_set[(i-1) % 8]
                )
            )
        
        refine_diff = []
        trans_collection = TransformCollection()
        for cali_set in tqdm(cali_sequence):
            current_frame = str(cali_set['frame']).zfill(7)
            set_target_cam_id = cali_set['target']
            target_info = frame_info_dict[(set_target_cam_id, current_frame)]
            for s_cam_id in cali_set['source']:
                source_info = frame_info_dict[(s_cam_id, current_frame)]
                trans_mat = camera_calibration_pnp(source_info, target_info, k=4)
                if trans_mat is not None:
                    if args.refine: 
                        trans_mat_adjust = calibration_icp_adjust(source_info, target_info, trans_mat)
                        refine_diff.append(np.linalg.norm(trans_mat - trans_mat_adjust))
                        trans_mat = trans_mat_adjust
                    trans_collection.add_transform(s_cam_id, set_target_cam_id, trans_mat)
                
        if len(refine_diff) > 0:
            print(f'Refine diff: {np.max(refine_diff)}')
        
        if not os.path.exists(trans_path):
            os.mkdir(trans_path)
        
        trans_collection.merge_to_target(target_id)
        trans_collection.save_collection(trans_path)
        
    elif task == 'merge':
        trans_collection = TransformCollection()
        trans_collection.load_collection(trans_path)
        cali_set = trans_collection.merge_to_target(target_id)
        frame_id = args.frame_id
            
        frame_info_dict = {}
        pcd_list = []
        for cam_id in cali_set:
            cam_series_id = cam_series[cam_id]
            frame_info_dict[cam_id] = Frame(get_color_raw_path(root_path, frame_id, cam_series[cam_id]), cam_info_dict[cam_id])
            pcd_list.append(frame_info_dict[cam_id].get_pcd(args.near_clip, args.far_clip, args.downsample_step))
        combined_pcd = merge_pcd(trans_collection, pcd_list, cali_set, target_id)
        o3d.io.write_point_cloud(pcd_path, combined_pcd)    
    
    elif task == 'test':
        frame_id = 1
        cam_id = '1246'
        mask_patzh = get_mask_path(root_path, frame_id, cam_id)
        frame_info = Frame(get_color_raw_path(root_path, frame_id, cam_id), cam_info_dict[cam_id], mask_path)
        pcd = frame_info.get_pcd(args.near_clip, args.far_clip, args.downsample_step)
        o3d.io.write_point_cloud(pcd_path, pcd)