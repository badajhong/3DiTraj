import os
import numpy as np
import json
import copy
import random
from scipy.special import softmax

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

import torch
from torch.utils.data import Dataset

# import open3d as o3d
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.transform import Rotation as R

from genrobo3d.train.datasets.common import (
    pad_tensors, gen_seq_masks, random_rotate_z
)
from genrobo3d.configs.rlbench.constants import (
    get_rlbench_labels, get_robot_workspace
)
from genrobo3d.utils.rotation_transform import (
    RotationMatrixTransform, quaternion_to_discrete_euler
)
from genrobo3d.utils.robot_box import RobotBox
from genrobo3d.utils.action_position_utils import get_disc_gt_pos_prob


class IK_MLP_Dataset(Dataset):
    def __init__(self, data_dir, taskvar_file=None, **kwargs):
        self.taskvars = json.load(open(taskvar_file))

        self.gripper_data = {}
        self.joint_data = {}
        self.data_ids = []

        for taskvar in self.taskvars:
            # Read Joint angle json data
            gripper_json_path = os.path.join(data_dir, f"{taskvar}_gripper.json")
            joint_json_path = os.path.join(data_dir, f"{taskvar}_joints.json")

            # Skip if files do not exist
            if not (os.path.exists(gripper_json_path) and os.path.exists(joint_json_path)):
                print(f"[WARNING] Skipping taskvar {taskvar} because JSON files are missing.")
                continue

            with open(gripper_json_path, 'r') as f:
                self.gripper_data[taskvar] = json.load(f)

            with open(joint_json_path, 'r') as f:
                self.joint_data[taskvar] = json.load(f)

            gripper_episodes = self.gripper_data[taskvar][taskvar]
            joint_episodes = self.joint_data[taskvar][taskvar]

            # Only use common episodes that exist in both files
            common_episode_keys = set(gripper_episodes.keys()).intersection(joint_episodes.keys())

            for episode_key in common_episode_keys:
                gripper_steps = gripper_episodes[episode_key]
                joint_steps = joint_episodes[episode_key]
                num_steps = min(len(gripper_steps), len(joint_steps))

                for t in range(num_steps):
                    self.data_ids.append((taskvar, episode_key, t))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        taskvar, episode_key, frame_idx = self.data_ids[idx]

        gripper_step = self.gripper_data[taskvar][taskvar][episode_key][frame_idx]
        joint_step = self.joint_data[taskvar][taskvar][episode_key][frame_idx]

        gt_gripper = np.array(gripper_step["gripper_pose"])
        gt_joint = np.array(joint_step["joint_positions"])

        return {
            'data_ids': taskvar,
            'episode_ids': episode_key,
            'frame_ids': torch.tensor(float(frame_idx)),
            'gt_gripper': torch.from_numpy(gt_gripper).float(),
            'gt_joint': torch.from_numpy(gt_joint).float(),
        }
    
def base_collate_fn(data):
    batch = {}
    for key in data[0].keys():
        if isinstance(data[0][key], torch.Tensor):
            batch[key] = torch.stack([x[key] for x in data], 0)
        elif isinstance(data[0][key], (int, float)):
            batch[key] = torch.tensor([x[key] for x in data])
        elif isinstance(data[0][key], str):
            batch[key] = [x[key] for x in data]  # keep as list of strings
        else:
            batch[key] = [x[key] for x in data]  # fallback for other types

    return batch

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = IK_MLP_Dataset(
        'data/gembench/GemBench/train/',
        taskvar_file='assets/taskvars_train_ik_mlp.json', 
        )
    
    print('#data', len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, 
        collate_fn=base_collate_fn
    )

    print('#steps', len(dataloader))

    for batch in dataloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.size())

        print(batch['gt_gripper'].min(dim=0)[0])
        print(batch['gt_gripper'].max(dim=0)[0])
        print(batch['gt_joint'].min(dim=0)[0])
        print(batch['gt_joint'].max(dim=0)[0])
        print(batch['data_ids'])
        print(batch['episode_ids'])
        print(batch['frame_ids'])
        print(batch['gt_gripper'])
        print(batch['gt_joint'])
        break
