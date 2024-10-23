import open3d as o3d
import numpy as np
import argparse
import os
import sys
import copy

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    combined_point_cloud = copy.deepcopy(target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])
    combined_point_cloud += source_temp
    o3d.io.write_point_cloud(args.save, combined_point_cloud)

def save_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    combined_point_cloud = copy.deepcopy(target)
    source_temp.transform(transformation)
    # o3d.visualization.draw_geometries([source_temp, target])
    combined_point_cloud += source_temp
    o3d.io.write_point_cloud(args.save, combined_point_cloud)

def draw_registration_result_color_multi(sources, target, transformations):
    geometries = [copy.deepcopy(target)]
    combined_point_cloud = copy.deepcopy(target)

    for i in range(len(sources)):
        source_temp = copy.deepcopy(sources[i])
        source_temp.transform(transformations[i])
        geometries.append(copy.deepcopy(source_temp))
        combined_point_cloud += source_temp

    o3d.visualization.draw_geometries(geometries)
    o3d.io.write_point_cloud(args.save, combined_point_cloud)

def save_registration_result_color_multi(sources, target, transformations):
    geometries = [copy.deepcopy(target)]
    combined_point_cloud = copy.deepcopy(target)
    transformations = np.array(transformations)
    if transformations.ndim == 2:
        transformations = np.expand_dims(transformations, axis=0)

    for i in range(len(sources)):
        source_temp = copy.deepcopy(sources[i])
        source_temp.transform(transformations[i])
        geometries.append(copy.deepcopy(source_temp))
        combined_point_cloud += source_temp

    o3d.io.write_point_cloud(args.save, combined_point_cloud)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ICP Registration')
    parser.add_argument('--source', type=str, nargs='+', help='source point cloud')
    parser.add_argument('--target', type=str, help='target point cloud')
    parser.add_argument('--load_H', type=str, nargs='+', help='Loading path for initial transformation matrix', default='')
    parser.add_argument('--save_H', type=str, help='Saving path for final transformation matrix', default='transform.txt')
    parser.add_argument('--save_global_H', type=str, help='Saving path for global transformation matrix',
                        default='global_transform.txt')
    parser.add_argument('--save', type=str, help='File to save the final generated point cloud', default='final.ply')
    parser.add_argument('--verbose', action='store_true', help='dump debug information')
    args = parser.parse_args()

    sources_paths = args.source

    sources = [o3d.io.read_point_cloud(sources_paths) for sources_paths in sources_paths]

    # Save the filtered point cloud
    target = o3d.io.read_point_cloud(args.target)

    print("1. Load two point clouds and show initial pose")
    current_transformation = [np.identity(4) for _ in sources]
    if args.verbose:
        draw_registration_result_color_multi(sources, target, current_transformation)

    if not args.load_H:
        print("No initial transformation matrix is provided. Run global registration.")
        print("2. Global Registration for a coarse start")

        def preprocess_point_cloud(pcd, voxel_size):
            print(":: Downsample with a voxel size %.3f." % voxel_size)
            pcd_down = pcd.voxel_down_sample(voxel_size)

            radius_normal = voxel_size * 2
            print(":: Estimate normal with search radius %.3f." % radius_normal)
            pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

            radius_feature = voxel_size * 5
            print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
            pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            return pcd_down, pcd_fpfh

        voxel_size = 0.02  # means 5cm for the dataset

        sources_down = []
        sources_fpfh = []

        for source in sources:
            source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
            sources_down.append(source_down)
            sources_fpfh.append(source_fpfh)

        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

        def execute_global_registration(source_down, target_down, source_fpfh,
                                        target_fpfh, voxel_size):
            distance_threshold = voxel_size * 1.5
            print(":: RANSAC registration on downsampled point clouds.")
            print("   Since the downsampling voxel size is %.3f," % voxel_size)
            print("   we use a liberal distance threshold %.3f." % distance_threshold)
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, True,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold)
                ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.99))
            return result

        result_ransac = execute_global_registration(sources_down[0], target_down,
                                                    sources_fpfh[0], target_fpfh,
                                                    voxel_size)
        print(result_ransac)

        current_transformation = result_ransac.transformation

        # Save global transformation
        np.savetxt(args.save_global_H, result_ransac.transformation)
    else:
        print("Load initial transformation matrix")
        current_transformation = [np.loadtxt(file) for file in args.load_H]

    # Visualize the result of global registration
    if args.verbose:
        if len(sources) == 1:
            if not args.load_H:
                draw_registration_result_original_color(sources[0], target, current_transformation)
            else:
                draw_registration_result_original_color(sources[0], target, current_transformation[0])
        else:
            draw_registration_result_color_multi(sources, target, current_transformation)
    else:
        save_registration_result_color_multi(sources, target, current_transformation)

    if not args.load_H:
        # colored pointcloud registration
        # This is implementation of following paper
        # J. Park, Q.-Y. Zhou, V. Koltun,
        # Colored Point Cloud Registration Revisited, ICCV 2017
        voxel_radius = [0.04, 0.02, 0.01]
        max_iter = [50, 30, 14]

        def colored_registration(source, target, voxel_radius, max_iter, initial_transformation):
            current_transformation = initial_transformation
            print("3. Colored point cloud registration")
            for scale in range(3):
                iter = max_iter[scale]
                radius = voxel_radius[scale]
                print([iter, radius, scale])

                print("3-1. Downsample with a voxel size %.2f" % radius)
                source_down = source  # .voxel_down_sample(radius)
                target_down = target  # .voxel_down_sample(radius)

                print("3-2. Estimate normal.")
                source_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
                target_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

                print("3-3. Applying colored point cloud registration")
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down, target_down, radius, current_transformation,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                    relative_rmse=1e-6,
                                                                    max_iteration=iter))
                current_transformation = result_icp.transformation
                print(result_icp)
            return current_transformation

        print(sources)
        if len(sources) == 1:
            current_transformation = colored_registration(sources[0], target, voxel_radius, max_iter, current_transformation)
            # remove normals in point cloud
            sources[0].normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
            target.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
            if args.verbose:
                draw_registration_result_original_color(sources[0], target, current_transformation)
            # Dump transformation matrix
            np.savetxt(args.save_H, current_transformation)
        else:
            final_transformations = []
            for i, source in enumerate(sources):
                transformation = current_transformation[i]
                final_transformation = colored_registration(source, target, voxel_radius, max_iter, transformation)
                final_transformations.append(final_transformation)

            # Remove normals from point clouds
            for source in sources:
                source.normals = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            target.normals = o3d.utility.Vector3dVector(np.zeros((0, 3)))

            if args.verbose:
                draw_registration_result_color_multi(sources, target, final_transformations)

            np.savetxt(args.save_H, final_transformations[-1])
    else:
        print("Skipping global and colored registration as transformation matrix is provided.")


if __name__ == "__main__":
    print("running")
    # build masked pcd
    root_path = 'data'
    output_path = os.path.join('data', 'pcd_data')
    frame_id = str(50).zfill(7)
    cam_list = list(cam_series.keys())
    cam_list = [cam for cam in cam_list if cam not in camera_set[8]]
    for cam in cam_list:
        print(cam)
        test_serie = cam_series[cam]
        info = read_data(root_path, cam_series[cam], frame_id, need_depth=True, mask_path=os.path.join('data', 'masked_imgs','masks'))
        build_point_cloud_from_depth_downsampled(info, far_clip=2, downsample_step=2, output_path=os.path.join('data', 'masked_pcd_data',))