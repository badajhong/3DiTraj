import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import os
import pprint
import numpy as np

# Change this to a valid path in your dataset
lmdb_root = 'data/gembench/train_dataset/keysteps_bbox/seed0/'

# Pick a subdirectory (task variation)
taskvar = os.listdir(lmdb_root)[0]  # just take the first one
lmdb_path = os.path.join(lmdb_root, taskvar)

# Open LMDB
env = lmdb.open(lmdb_path, readonly=True, lock=False)
txn = env.begin()

cursor = txn.cursor()

for key, val in cursor:
    print(f"🔑 Key: {key.decode('utf-8')}")
    
    # Decode the value
    data = msgpack.unpackb(val)
    
    print("\n📦 Top-level keys:")
    for k in data:
        print(f"  - {k}: type={type(data[k])}")
        if isinstance(data[k], (list, tuple)) and len(data[k]) > 0:
            print(f"    ↳ First item type: {type(data[k][0])}")
            if hasattr(data[k][0], 'shape'):
                print(f"    ↳ First item shape: {data[k][0].shape}")
        elif hasattr(data[k], 'shape'):
            print(f"    ↳ Shape: {data[k].shape}")

    # print("\n📦 Keys in data:")
    # print(list(data.keys()))

    # print("\n🔍 Inspecting pose_info:")
    # pose_info = data.get('pose_info', None)
    # if pose_info is None:
    #     print("❌ pose_info not found in this entry.")
    # else:
    #     for k, v in pose_info.items():
    #         print(f"\n🔹 Key: '{k}'")
    #         print(f"   Type: {type(v)}")
    #         if isinstance(v, list):
    #             print(f"   Length: {len(v)}")
    #             if len(v) > 0:
    #                 print(f"   Sample[0]: {v[0]}")
    #                 if hasattr(v[0], 'shape'):
    #                     print(f"   Shape: {v[0].shape}")
    #         elif isinstance(v, dict):
    #             pprint.pprint(v)
    #         elif isinstance(v, np.ndarray):
    #             print(f"   Shape: {v.shape}")
    #             print(f"   Sample[0]: {v[0]}")
    #         else:
    #             print(f"   Value: {v}")

    # print("\n🔍 Inspecting action:")
    # action = data.get('action', None)

    # if action is None:
    #     print("❌ action not found in this entry.")
    # else:
    #     print(f"✅ Found action: type={type(action)}, shape={action.shape}")
        
    #     # Print first few entries
    #     for i in range(min(3, len(action))):
    #         a = action[i]
    #         print(f"\n🔹 Step {i}: {a}")
    #         if isinstance(a, (list, np.ndarray)) and len(a) == 8:
    #             print(f"   ↳ Pos:     {a[:3]}")
    #             print(f"   ↳ Rot:     {a[3:7]} (quaternion)")
    #             print(f"   ↳ Gripper: {a[7]}")

    print("\n🔍 Inspecting gripper_pose:")
    gripper_pose = data.get('gripper_pose', None)

    if gripper_pose is None:
        print("❌ gripper_pose not found.")
    elif not isinstance(gripper_pose, list):
        print(f"⚠️ Expected list, but got: {type(gripper_pose)}")
    else:
        print(f"✅ Found gripper_pose: list of length {len(gripper_pose)}")
        
        # Print first few items
        for i, pose_dict in enumerate(gripper_pose[:3]):  # first 3 timesteps
            print(f"\n🔹 Step {i}:")
            if isinstance(pose_dict, dict):
                for k, v in pose_dict.items():
                    print(f"   - {k}: {type(v)}", end="")
                    if isinstance(v, (list, tuple, np.ndarray)):
                        print(f", len={len(v)}", end="")
                        if hasattr(v, 'shape'):
                            print(f", shape={v.shape}", end="")
                        print(f", sample={v[:5]}")
                    else:
                        print(f", value={v}")
            else:
                print(f"   ⚠️ Not a dict: {pose_dict}")

    break  # Stop after inspecting one entry