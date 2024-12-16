import open3d as o3d
import numpy as np

pcd_path = '/scratch/projects/fouheylab/dma9300/recon3d/billy_pcs/point_cloud.ply'
mesh_path = 'mesh_0.ply'
# Load the point cloud
pcd = o3d.io.read_point_cloud(pcd_path)

# Perform statistical outlier removal
# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
# pcd = pcd.select_by_index(ind)
o3d.visualization.draw_geometries([pcd])

# Estimate normals if not already present
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Perform Poisson surface reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Optionally, you can remove low-density vertices to clean up the mesh
vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)

# Save the mesh
o3d.io.write_triangle_mesh(mesh_path, mesh)

# Visualize the mesh
# o3d.visualization.draw_geometries([mesh])