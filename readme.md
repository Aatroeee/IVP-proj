# Camera Calibration

## 1. Run Calibration


Start with Data:
- `data/`
    - `raw_data/`: contains raw data for each frame
    - `meta_data/`: contains meta data for each camera

Run the `calibration.ipynb` notebook to get calibration results

Basic functions:
- `generate_png_from_raw`: generate png from raw
- `read_data`: load color and depth data from raw and meta
- `build_point_cloud_from_depth`: map depth to 3D points in color space, build point cloud on different camera space
- `camera_calibration`: extract chessboard corners from color image, get 3D world points in target camera space, 
use PnP for source camera to estimate extrinsic matrix
- `save_registration_result_color_multi`: save pointcloud registration result

## 2. Save Camera Json
To generate the camera JSON files for 3DGS training:

Run the `generate_camera_json.py` script

   ```
   python generate_camera_json.py --input *root_path* --output *json_output_path* --pcd *combined_pcd_path* --frame *frame_id*
   ```

## 3. Other Tools
- `others/colmap_ds.py`: modify COLMAP database to insert camera poses
- `others/mask_generator.ipynb`: generate mask for human figure using segment anything model
- `registration_icp.py`: register point cloud using ICP algorithm
- `sort_json.ipynb`: sort camera from distance when saving json

## Update
- 2024/12/05: 
    - rewrite calibration pipeline to `utils.py` camera pose estimation and point cloud registration
    - add `scene.py` for generating data structure for 3DGS training
