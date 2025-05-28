import os
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import matplotlib.pyplot as plt

msgpack_numpy.patch()

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
lmdb_path_1 = os.path.join(project_root, 'data/gembench/train_dataset/keysteps_bbox_pcd/seed0/voxel1cm', 'close_fridge+0')
lmdb_path_2 = os.path.join(project_root, 'data/gembench/train_dataset/keysteps_bbox/seed0/', 'close_fridge+0')
lmdb_path_3 = os.path.join(project_root, 'data/gembench/train_dataset/motion_keysteps_bbox_pcd/seed0/voxel1cm', 'close_fridge+0')

# ---------- Load point cloud from lmdb_path_1 ----------
env1 = lmdb.open(lmdb_path_1, readonly=True, lock=False)
txn1 = env1.begin()
cursor1 = txn1.cursor()
print (cursor1)
for key, value in cursor1:
    data1 = msgpack.unpackb(value)
    break  # just one sample

xyz = data1['xyz'][0]  # shape (N, 3)
rgb = data1['rgb'][0] / 255.0  # normalize to [0, 1]

# ---------- Load point cloud from lmdb_path_3 ----------
env3 = lmdb.open(lmdb_path_3, readonly=True, lock=False)
txn3 = env3.begin()
cursor3 = txn3.cursor()

for key, value in cursor3:
    data3 = msgpack.unpackb(value)
    break  # just one sample

xyz3 = data3['xyz'][0]  # shape (N, 3)
rgb3 = data3['rgb'][0] / 255.0  # normalize to [0, 1]

# ---------- Load trajectory from lmdb_path_2 ----------
env2 = lmdb.open(lmdb_path_2, readonly=True, lock=False)
txn2 = env2.begin()
cursor2 = txn2.cursor()

for key, value in cursor2:
    data2 = msgpack.unpackb(value)
    break

actions = data2['action']  # shape (T, 8)
trajectory = np.array([a[:3] for a in actions])  # just xyz positions

# ---------- Visualization ----------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot point cloud
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1, alpha=0.5, label='Point Cloud')

# ax.scatter(xyz3[:, 0], xyz3[:, 1], xyz3[:, 2], c='blue', s=1, alpha=0.5, label='Motion Point Cloud')

# Plot trajectory
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        color='red', marker='o', linestyle='-', linewidth=2, markersize=4, label='Gripper Trajectory')

# Labels and view
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud with Robot Trajectory')
ax.legend()
plt.tight_layout()
plt.show()
