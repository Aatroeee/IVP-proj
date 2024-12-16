import open3d as o3d
import numpy as np
import trimesh

def load_smpl_mesh(smpl_mesh_path):
    """Load SMPL template mesh from file."""
    mesh = trimesh.load_mesh(smpl_mesh_path)
    return mesh

def apply_icp(source_points, target_points, max_iter=1000, tolerance=1e-6):
    """Align the source points (noisy point cloud) to the target points (SMPL template mesh) using ICP."""
    # Convert numpy arrays to Open3D PointCloud objects
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_points)

    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_points)

    # Apply ICP
    reg_icp = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, max_correspondence_distance=0.05,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter, relative_fitness=tolerance)
    )
    
    # Return the transformation matrix
    return reg_icp.transformation

def main(smpl_mesh_path, noisy_point_cloud_path, output_ply_path):
    # Load SMPL template mesh
    smpl_mesh = load_smpl_mesh(smpl_mesh_path)

    # Extract the vertices of the mesh (SMPL template points)
    smpl_vertices = smpl_mesh.vertices

    # Load noisy point cloud
    noisy_point_cloud = o3d.io.read_point_cloud(noisy_point_cloud_path)
    noisy_points = np.asarray(noisy_point_cloud.points)

    # Apply ICP to align noisy point cloud to SMPL mesh
    transformation_matrix = apply_icp(noisy_points, smpl_vertices)

    # Apply the transformation to the noisy point cloud
    noisy_point_cloud.transform(transformation_matrix)

    # Save the aligned point cloud to a .ply file
    o3d.io.write_point_cloud(output_ply_path, noisy_point_cloud)

    # Visualize the result
    # o3d.visualization.draw_geometries([noisy_point_cloud, smpl_mesh])

# Example usage
smpl_mesh_path = "/scratch/projects/fouheylab/dma9300/recon3d/template.obj"  # Path to the SMPL template mesh
noisy_point_cloud_path = "/scratch/projects/fouheylab/dma9300/recon3d/billy_pcs/point_cloud.ply"  # Path to the noisy point cloud
output_ply_path = "aligned_point_cloud.ply"  # Path where the aligned point cloud will be saved

main(smpl_mesh_path, noisy_point_cloud_path, output_ply_path)
