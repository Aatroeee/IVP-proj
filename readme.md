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
- `utils/colmap_ds.py`: modify COLMAP database to insert camera poses
- `utils/mask_generator.ipynb`: generate mask for human figure using segment anything model
- `registration_icp.py`: register point cloud using ICP algorithm
- `sort_json.ipynb`: sort camera from distance when saving json



## How to Run the Code - Denis

### Step 1: Run Calibration
## Requirements

- Python 3.x
- OpenCV
- NumPy
- Pytorch

You need GPU to run  with r2d2 

To perform camera calibration, use the following command:

```bash
python code/compute_extrinsics.py --feature_extractor {options: r2d2, orb, sift} --pnp {True or False, default: False}
```

**Example:**

```bash
python code/compute_extrinsics.py --feature_extractor r2d2 --pnp True
```

This will compute the extrinsic parameters for your images.

### Step 2: Transform Cameras

After generating the extrinsics, proceed with transforming the cameras using:

```bash
python code/transform_cameras.py
```

**Important:** Edit `trans_path` in `transform_cameras.py` (line 34) to point to the calibration output file. This file is usually inside your data folder and follows the naming convention:

```
data/output_data_calib_{feature_extractor name}
```

For example:

```python
trans_path = 'data/output_data_calib_r2d2'
```

### Step 3: Register Point Clouds

1. Open `code/registration.py` and update the `trans_path` variable (line 45) with the same path used earlier:

   ```python
   trans_path = 'data/output_data_calib_r2d2'
   ```

2. Run the registration script:

   ```bash
   python code/registration.py
   ```

### Step 4: Visualize the Point Cloud

Use a 3D visualization tool like **MeshLab** or **CloudCompare** to view the generated point cloud.










