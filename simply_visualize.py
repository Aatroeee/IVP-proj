import open3d as o3d
import numpy as np
import sys

def loadCloudFromBinary(file, cols=3):
    f = open(file, "rb")
    binary_data = f.read()
    f.close()
    temp = np.frombuffer(
        binary_data, dtype='float32', count=-1)
    data = np.reshape(temp, (cols, int(temp.size/cols)))
    return data.T[:,:3]

def loadCloudFromNpy(file):
    pcd = o3d.geometry.PointCloud()
    pcd0 = o3d.geometry.PointCloud()
    d = np.load(file)
    pcd0.points = o3d.utility.Vector3dVector(d)
    # o3d.visualization.draw_geometries([pcd0])
    mins = np.min(d, 0)
    d -= mins
    center = np.mean(d, 0)
    d += center
    max_value = np.max(d)
    d = np.round(d/max_value*500)
    pcd.points = o3d.utility.Vector3dVector(d.astype(float))
    return pcd, d

fn = sys.argv[1]
tg = sys.argv[2]
if fn[-3:] == 'ply':
    pcd = o3d.io.read_point_cloud(fn)
elif fn[-3:] == 'npz':
    d = np.load(fn)
    pts = d['point_cloud']
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
elif fn[-3:] == 'bin':
    d = loadCloudFromBinary(fn, 11)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(d) 
elif fn[-3:] == 'npy':
    pcd, d = loadCloudFromNpy(fn)


# rescale = True
rescale = False
if rescale:
    total_size = 1003.0
    center = [281, 510.5, 170]
    d = np.asarray(pcd.points)
    d *= total_size
    d += np.array(center)
    pcd.points = o3d.utility.Vector3dVector(d)

# filtered = True
filtered = False
d = np.asarray(pcd.points)

if filtered:
    valid_indices = np.where(d[:,1] > 0)
    d = d[valid_indices]

    valid_indices = np.where(d[:,2] < 400)
    d = d[valid_indices]

    pcd.points = o3d.utility.Vector3dVector(d)

quantize = False
# quantize = True
if quantize:
    mins = np.min(d, 0)
    maxs = np.max(d, 0)
    ranges = maxs - mins
    max_range = np.max(ranges)
    q = 2**10 / max_range
    nd = np.round(d*q)
    npcd = o3d.geometry.PointCloud()
    npcd.points = o3d.utility.Vector3dVector(nd)
    pcd = npcd
    print(np.max(nd, 0))
    print(np.min(nd, 0))
    mse = np.mean(np.square(d*q - nd))
    print("MSE: %.4e" % (mse))

print(np.max(d, 0))
print(np.min(d, 0))

print(np.asarray(pcd.points).shape)
o3d.visualization.draw_geometries([pcd])
# o3d.io.write_point_cloud(tg, pcd)
import IPython
#IPython.embed()
quit()

# Coordinates transformation

# First normalize
pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())

norm_pts = np.asarray(pcd.points)
# q2 = np.random.rand(128,3)
maxs = np.max(norm_pts, 0)
mins = np.min(norm_pts, 0)
ranges = maxs - mins
norm_pts -= mins

axis_x = 0
axis_y = 1
axis_z = 2

# pts = np.asarray(pcd.points)
pts = norm_pts
zs = pts[:, axis_z]
xs = pts[:, axis_x]
ys = pts[:, axis_y]
r = np.sqrt(xs**2 + ys**2)
theta = np.arcsin(ys/r)
new_pts = np.stack([r, theta, zs], -1)

new_pcd = o3d.geometry.PointCloud()
new_pcd.points = o3d.utility.Vector3dVector(new_pts)
o3d.visualization.draw_geometries([pcd])

# import IPython
# IPython.embed()
