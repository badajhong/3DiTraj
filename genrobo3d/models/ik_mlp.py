from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from genrobo3d.utils.rotation_transform import discrete_euler_to_quaternion
from genrobo3d.models.base import BaseModel, RobotPoseEmbedding
from genrobo3d.utils.rotation_transform import RotationMatrixTransform

from .DiT.preprocess import rescale_joint

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.LayerNorm(size),
        )

    def forward(self, x):
        return x + self.block(x)

class IKMLP(BaseModel):

    '''
    Inverse Kinematics MLP for Franka Panda 7-DoF robot arm.
    This model predicts joint angles given the end-effector position and orientation.
    The input is a 7D vector representing the end-effector pose (position + orientation).
    The output is a 7D vector representing the joint angles.
    '''

    def __init__(self, config):
        super().__init__()
        
        self.config = config

        # Franka Panda 7-DoF
        self.ik_mlp = nn.Sequential(
            nn.Linear(7, 512),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(512),
            nn.Dropout(0.5),
            
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(512),
            nn.Dropout(0.5),
            nn.Linear(512, 7),
            # nn.Tanh()  # optional, depending on joint scaling
        )

    def forward(self, batch, compute_loss=False, **kwargs):
        
        ''' 
        batch data:
        gt_gripper: (batch, 7)
        gt_joint: (batch, 7)
        '''

        batch = self.prepare_batch(batch)

        pred_joints = self.ik_mlp(batch['gt_gripper'])
        pred_joints = rescale_joint(pred_joints, mode='real')

        if compute_loss:
            losses = self.compute_loss(
                    pred_joints, batch['gt_joint']
                )
            return pred_joints, losses
        else:
            return pred_joints

    def compute_loss(self, pred_joints, tgt_joints):

        joint_loss = F.mse_loss(pred_joints, tgt_joints, reduction='mean')
        # joint_loss = torch.mean(torch.abs(((pred_joints - tgt_joints + np.pi) % np.pi*2)-np.pi))
        total_loss = joint_loss
        
        # Mean over batch for each joint
        joint_loss_per_joint = F.mse_loss(pred_joints, tgt_joints, reduction='none').mean(dim=0)
        # joint_loss_per_joint = torch.mean(torch.abs(((pred_joints - tgt_joints + np.pi) % np.pi*2)-np.pi), dim=0)

        # Create a dictionary with named joints
        named_joint_losses = {
            f'j{i+1}': joint_loss_per_joint[i] for i in range(7)
        }

        total_loss = joint_loss_per_joint.mean()

        # Combine everything into the returned dict
        return {
            'total': total_loss,
            **named_joint_losses
        }

if __name__ == '__main__':
    from genrobo3d.configs.default import get_config

    config = get_config('genrobo3d/configs/rlbench/ik_mlp.yaml')
    model = IKMLP(config.MODEL).cuda()

    fake_batch = {
        'gt_gripper': torch.rand(4, 7),
        'gt_joint': torch.rand(4, 7),
    }

    outs = model(fake_batch, compute_loss=True)
    print(outs[1])