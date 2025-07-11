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
from genrobo3d.utils.action_position_utils import get_best_pos_from_disc_pos
from .dit import DiT
from tqdm.auto import tqdm
from genrobo3d.models.DiT.diffusion import create_diffusion
from genrobo3d.models.DiT.preprocess import padding, rescale



class DiTraj(BaseModel):
    """Adaptive batch/layer normalization conditioned on text/pose/stepid
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.diffusion = create_diffusion(timestep_respacing="", diffusion_steps=self.config.dit_config.num_timesteps)
        self.dit = DiT(
            in_channels= 8,
            out_channels= 8,
            hidden_size=64,
            depth=self.config.dit_config.depth,
            num_heads=16,
            mlp_ratio=4.0,
            action_dropout_prob=self.config.dit_config.action_dropout_prob,
            learn_sigma=False,
            max_seq_len= self.config.dit_config.max_steps
        )

    # TODO: Change config file to remove forward input '''max_steps, batch_size, robot_type'''
    def forward(self, batch, pc_tokens, final_pred_joints_actions, max_steps, batch_size, robot_type, mode='train', **kwargs):
        '''batch data:
            pc_fts: (batch, npoints, dim)
            txt_embeds: (batch, txt_dim)
        '''
        batch = self.prepare_batch(batch)
        device = batch['pc_fts'].device

        pc_tokens = torch.stack([torch.mean(x, 0) for x in pc_tokens], 0)
        pc_tokens = pc_tokens.unsqueeze(1).repeat(1, max_steps, 1)         # shape: [8, 30, 64]
        
        
        
        pad_gt_traj, pad_mask = padding(max_steps, batch, batch_size, robot_type, device)

        timesteps = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=device)
        pad_gt_traj = rescale(pad_gt_traj, mode= 'norm')
        model_kwargs = {
            'pc_token': pc_tokens,                                    # (batch, N(num_points), 64)
            'y': final_pred_joints_actions,                          # (batch, 8)
        }
        if mode == 'train':
            loss_dict = self.diffusion.training_losses(self.dit, pad_gt_traj, timesteps, model_kwargs)
            loss = loss_dict["loss"].mean()
            return loss
        
        else: 
            # Forward pass with classifier-free guidance
            x_t = torch.randn(batch_size, max_steps, robot_type, device=device) 
            model_output = self.diffusion.p_sample_loop(self.dit, x_t.shape, x_t, clip_denoised=True, model_kwargs=model_kwargs, progress=True, device=device)
            pred_traj = model_output.view(-1, 8)
            
            pad_gt_traj = pad_gt_traj.view(-1, 8)
            pad_mask = pad_mask.view(-1)
            
            if mode == 'val':
                losses = self.validation_loss(
                    pred_traj, pad_gt_traj,
                    mask=pad_mask,
                )
                return losses
            else: 
                pred_traj = rescale(pred_traj, mode='real')
                return pred_traj    
            

    def validation_loss(self, pred_traj, tgt_traj, mask=None):
        rpred_traj = rescale(pred_traj, mode='real')
        rgt_traj = rescale(tgt_traj, mode= 'real')

        rpred_joint, rpred_open  = rpred_traj[..., :-1], rpred_traj[..., -1]
        npred_joint, npred_open = pred_traj[..., :-1], pred_traj[..., -1]

        rgt_joint, rgt_open = rgt_traj[..., :-1], rgt_traj[..., -1]
        ngt_joint, ngt_open = tgt_traj[..., :-1], tgt_traj[..., -1]

        ngt_open = (ngt_open + 1) / 2

        if mask is None:
            joint_loss = F.mse_loss(npred_joint, ngt_joint, reduction='mean')

            # open_loss = F.binary_cross_entropy_with_logits(npred_open, ngt_open, reduction='mean')
            open_loss = F.mse_loss(rpred_open, rgt_open, reduction='mean')

            real_joint_loss = F.mse_loss(rpred_joint, rgt_joint, reduction='mean') 
            
        else: 
            weights = torch.where(mask == 1, torch.tensor(3, device=mask.device), torch.tensor(1, device=mask.device))

            joint_loss = F.mse_loss(npred_joint, ngt_joint, reduction='none').mean(-1)  
            joint_loss = (weights * joint_loss).sum() / weights.sum() 

            # open_loss = F.binary_cross_entropy_with_logits(npred_open, ngt_open, reduction='none')  # (N,)
            open_loss = F.mse_loss(rpred_open, rgt_open, reduction='none')
            open_loss = (weights * open_loss).sum() / weights.sum() 

            real_joint_loss = F.mse_loss(rpred_joint, rgt_joint, reduction='none').mean(-1)   
            real_joint_loss = (weights * real_joint_loss).sum() / weights.sum() 

        total_loss = self.config.loss_config.joint_weight * joint_loss + self.config.loss_config.open_weight * open_loss
        
        return {
            'joint': joint_loss, 'open': open_loss, 'real_joint': real_joint_loss,
            'total': total_loss, 'pred_traj':rpred_traj
        }
        
    