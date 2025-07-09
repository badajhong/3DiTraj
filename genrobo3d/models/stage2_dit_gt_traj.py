from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from genrobo3d.utils.rotation_transform import discrete_euler_to_quaternion
from genrobo3d.models.base import BaseModel, RobotPoseEmbedding
from genrobo3d.utils.rotation_transform import RotationMatrixTransform
from genrobo3d.models.PointTransformerV3.model import (
    PointTransformerV3, offset2bincount, offset2batch
)
from genrobo3d.models.PointTransformerV3.model_ca import PointTransformerV3CA
from genrobo3d.utils.action_position_utils import get_best_pos_from_disc_pos
from .dit import DiT


class DiTraj(BaseModel):
    """Adaptive batch/layer normalization conditioned on text/pose/stepid
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.ptv3_model = PointTransformerV3(**config.ptv3_config)

        self.dit = DiT(
            in_channels=8,
            out_channels=8,
            hidden_size=64,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            action_dropout_prob=0.1,
            learn_sigma=False,
            max_seq_len=30
        )

    def prepare_ptv3_batch(self, batch):
        outs = {
            'coord': batch['pc_fts'][:, :3],
            'grid_size': self.config.action_config.voxel_size,
            'offset': batch['offset'],
            'batch': offset2batch(batch['offset']),
            'feat': batch['pc_fts'],
        }
        # encode context for each point cloud
        ctx_embeds = self.txt_fc(batch['txt_embeds'])
        if self.config.action_config.txt_reduce == 'attn':
            txt_weights = torch.split(self.txt_attn_fc(batch['txt_embeds']), batch['txt_lens'])
            txt_embeds = torch.split(ctx_embeds, batch['txt_lens'])
            ctx_embeds = []
            for txt_weight, txt_embed in zip(txt_weights, txt_embeds):
                txt_weight = torch.softmax(txt_weight, 0)
                ctx_embeds.append(torch.sum(txt_weight * txt_embed, 0))
            ctx_embeds = torch.stack(ctx_embeds, 0)
        
        if self.config.action_config.use_ee_pose:
            pose_embeds = self.pose_embedding(batch['ee_poses'])
            ctx_embeds += pose_embeds

        if self.config.action_config.use_step_id:
            step_embeds = self.stepid_embedding(batch['step_ids'])
            ctx_embeds += step_embeds

        outs['context'] = ctx_embeds

        return outs

    def forward(self, batch, pc_tokens, final_pred_joints_actions, max_steps, batch_size, robot_type, compute_loss=False, global_step=None, **kwargs):
        '''batch data:
            pc_fts: (batch, npoints, dim)
            txt_embeds: (batch, txt_dim)
        '''
        batch = self.prepare_batch(batch)
        device = batch['pc_fts'].device

        pc_tokens = torch.stack([torch.mean(x, 0) for x in pc_tokens], 0)
        pc_tokens = pc_tokens.unsqueeze(1).repeat(1, max_steps, 1)         # shape: [8, 30, 64]
        

        # Define padding of gt_actions, mask, step_ids
        pad_gt_traj = torch.zeros((batch_size, max_steps, robot_type), device=device)
        pad_mask = torch.zeros((batch_size, max_steps,), dtype=torch.bool, device=device)

        gt_joints = batch.get('gt_joints', None)
        gt_actions = batch.get('gt_actions', None)
        
        if gt_actions is not None and gt_joints is not None:
        

            step_ids = batch['step_ids'] 

            start_indices = (batch['step_ids'] == 0).nonzero(as_tuple=True)[0]
            end_indices = start_indices[1:] - 1
            last_index = torch.tensor([len(batch['step_ids']) - 1], device=batch['step_ids'].device)
            end_indices = torch.cat([end_indices, last_index]) 
            gt_trajs = torch.hstack([gt_joints, gt_actions[:, -1:]])
            pad_gt_traj = gt_trajs[end_indices].unsqueeze(1).expand(-1, max_steps, -1).clone()

            # ==== Teacher Forcing Schedule ====
            # Define decay strategy
            initial_ratio = self.config.initial_ratio
            decay_steps = self.config.num_train_steps
            teach_prob = max(0.0, initial_ratio - (global_step / decay_steps))
            use_teacher = np.random.rand() < teach_prob

            # Select input for joint MLP
            if use_teacher:
                final_pred_joints_actions = pad_gt_traj[:, -1, :]
            else:
                pass

            task_ids = torch.cumsum(torch.cat([
                        torch.tensor([1], device=step_ids.device),
                        (step_ids[1:] == 0).int()
                    ]), dim=0) - 1
            
            for i in range(gt_actions.shape[0]):  # N = total number of steps (task*steps)
                task = int(task_ids[i].item())
                step = int(step_ids[i].item())
                pad_gt_traj[task, step] = torch.cat([gt_joints[i], gt_actions[i][-1].unsqueeze(0)], dim=0)
                pad_mask[task, step] = True
            
            init_seq = torch.randn_like(pad_gt_traj)                            # (batch, max_steps, 8)
        else:
            # For Prediction
            init_seq = torch.randn(1, max_steps, robot_type, device=device)

        timesteps = torch.randint(0, 1000, (batch_size,), device=device)          # (batch)

        # pred_actions: (batch*max_steps, 8)
        pred_traj = self.dit(
            x=init_seq,
            t=timesteps,
            y=final_pred_joints_actions,                           # (batch, 8)
            pc_token=pc_tokens,                                    # (batch, N(num_points), 64)          
        )

        pad_gt_traj = pad_gt_traj.contiguous().view(-1, 8)     # (batch * max_steps, 8)
        pad_mask = pad_mask.reshape(-1)   # (batch * max_steps, )
        
        if compute_loss:
            losses = self.compute_loss(
                pred_traj, pad_gt_traj,
                mask = pad_mask,
            )
            return pred_traj, losses
        else:
            return pred_traj

    def compute_loss(self, pred_traj, tgt_joint, mask=None):
        device = tgt_joint.device
        pred_joint, pred_open  = pred_traj[..., :-1], pred_traj[..., -1]
        tgt_joint, tgt_open = tgt_joint[..., :-1], tgt_joint[..., -1]

        if mask is None:
            joint_loss = F.mse_loss(pred_joint, tgt_joint, reduction='mean')  
            # openness state loss
            open_loss = F.binary_cross_entropy_with_logits(pred_open, tgt_open, reduction='mean')
            
        else: 
            weights = torch.where(mask == 1, torch.tensor(3, device=mask.device), torch.tensor(1, device=mask.device))

            joint_loss = F.mse_loss(pred_joint, tgt_joint, reduction='none').mean(-1)  
            joint_loss = (weights * joint_loss).sum()

            open_loss = F.binary_cross_entropy_with_logits(pred_open, tgt_open, reduction='none')  # (N,)
            open_loss = (weights * open_loss).sum()

        total_loss = self.config.loss_config.joint_weight * joint_loss + self.config.loss_config.open_weight * open_loss
        
        return {
            'joint': joint_loss, 'open': open_loss, 
            'total': total_loss
        }


if __name__ == '__main__':
    from genrobo3d.configs.default import get_config

    config = get_config('genrobo3d/configs/rlbench/simple_policy_ptv3.yaml')
    model = DiTraj(config.MODEL).cuda()
    # model = SimplePolicyPTV3CA(config.MODEL).cuda()

    fake_batch = {
        'pc_fts': torch.rand(100, 6),
        'npoints_in_batch': [30, 70],
        'offset': torch.LongTensor([30, 100]),
        'txt_embeds': torch.rand(2, 1280),
        'txt_lens': [1, 1],
        'ee_poses': torch.rand(2, 8),
        'step_ids': torch.LongTensor([0, 1]),
        'gt_actions': torch.rand(2, 8),
    }

    outs = model(fake_batch, compute_loss=True)
    print(outs[1])