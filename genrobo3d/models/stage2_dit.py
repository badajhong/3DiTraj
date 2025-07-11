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
from .DiT.models import DiT
from .DiT.preprocess import rescale, padding, extract
from tqdm.auto import tqdm

class DiTraj(BaseModel):
    """Adaptive batch/layer normalization conditioned on text/pose/stepid
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        # Diffusion noise schedule
        self.num_timesteps = config.dit_config.num_timesteps
        self.sampling_timesteps = config.dit_config.sampling_timesteps
        self.eta = config.dit_config.eta

        # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
        #TODO beta start end timestep에 맞게 수정시키기
        def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
            scale = 1000 / T
            beta_start = beta_start * scale
            beta_end = beta_end * scale
            return torch.linspace(beta_start, beta_end, T)

        # Register noise schedule buffers
        betas = get_beta_schedule(self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

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
    def forward(self, batch, pc_tokens, final_pred_joints_actions, max_steps, batch_size, robot_type, compute_loss=False, **kwargs):
        '''batch data:
            pc_fts: (batch, npoints, dim)
            txt_embeds: (batch, txt_dim)
        '''
        batch = self.prepare_batch(batch)
        device = batch['pc_fts'].device

        pc_tokens = torch.stack([torch.mean(x, 0) for x in pc_tokens], 0)
        pc_tokens = pc_tokens.unsqueeze(1).repeat(1, max_steps, 1)         # shape: [8, 30, 64]

        pad_gt_traj, pad_mask = padding(max_steps, batch, batch_size, robot_type, device)

        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[timesteps]  # shape: [batch_size]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps]  # shape: [batch_size]

        pad_gt_traj = rescale(pad_gt_traj, mode= 'norm')
        
        noise = torch.randn_like(pad_gt_traj)
        noisy_input = sqrt_alpha_cumprod_t[:, None, None] * pad_gt_traj + \
                      sqrt_one_minus_alpha_cumprod_t[:, None, None] * noise
        
        # pred_actions: (batch*max_steps, 8)
        norm_pred_traj = self.dit(
            x=noisy_input,
            t=timesteps,
            y=final_pred_joints_actions,                           # (batch, 8)
            pc_token=pc_tokens,                                    # (batch, N(num_points), 64)          
        )

        if compute_loss:

            noise = noise.contiguous().view(-1, 8)                 # (batch * max_steps, 8)
            pad_mask = pad_mask.reshape(-1)                        # (batch * max_steps, )

            losses = self.compute_loss(
                norm_pred_traj, noise,
                mask = pad_mask,
            )
            return norm_pred_traj, losses
        else:
            return norm_pred_traj
        
        
      
    def compute_loss(self, pred_traj, tgt_traj, mask=None):
        pred_joint, pred_open  = pred_traj[..., :-1], pred_traj[..., -1]
        tgt_joint, tgt_open = tgt_traj[..., :-1], tgt_traj[..., -1]

        if mask is None:
            joint_loss = F.mse_loss(pred_joint, tgt_joint, reduction='mean')  
            open_loss = F.mse_loss(pred_open, tgt_open, reduction='mean')
            
        else: 
            weights = torch.where(mask == 1, torch.tensor(3, device=mask.device), torch.tensor(1, device=mask.device))

            joint_loss = F.mse_loss(pred_joint, tgt_joint, reduction='none').mean(-1)  
            joint_loss = (weights * joint_loss).sum() / weights.sum() 

            open_loss = F.mse_loss(pred_open, tgt_open, reduction='none')
            open_loss = (weights * open_loss).sum() / weights.sum() 

        total_loss = self.config.loss_config.joint_weight * joint_loss + self.config.loss_config.open_weight * open_loss
        
        return {
            'joint': joint_loss, 'open': open_loss, 
            'total': total_loss
        }
        
    @torch.no_grad()
    def ddpm_sample(self, batch, pc_tokens, final_pred_joints_actions, max_steps, batch_size, robot_type, compute_loss=False, **kwargs):
        """
        Diffusion Sampling: Starts from pure noise and iteratively denoises.
        If compute_loss=True, calculates loss against ground truth.
        """
        device = batch['pc_fts'].device
        x_t = torch.randn(batch_size, max_steps, robot_type, device=device)  # Start from pure noise

        # Prepare pc_tokens
        pc_tokens = torch.stack([torch.mean(x, 0) for x in pc_tokens], 0)
        pc_tokens = pc_tokens.unsqueeze(1).repeat(1, max_steps, 1)

        # Iterative denoising
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device)

            predicted_noise = self.dit(
                x=x_t,
                t=t_batch,
                y=final_pred_joints_actions,
                pc_token=pc_tokens,
            )

            predicted_noise = predicted_noise.view(batch_size, max_steps, -1)

            beta_t = self.betas[t]
            alpha_t = 1.0 - beta_t
            alpha_cumprod_t = self.alphas_cumprod[t]

            sqrt_recip_alpha_t = 1 / torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)

            model_mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)

            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)

            x_t = model_mean + torch.sqrt(beta_t) * noise

        # Change to real Value
        x_t = torch.clamp(x_t, -1.0, 1.0)
        pred_traj = x_t.view(-1, 8)

        if compute_loss:

            pad_gt_traj, pad_mask = padding(max_steps, batch, batch_size, robot_type, device)
            pad_gt_traj = pad_gt_traj.view(-1, 8)
            pad_mask = pad_mask.view(-1)

            losses = self.sample_compute_loss(
                pred_traj, pad_gt_traj,
                mask = pad_mask,
            )
            return pred_traj, losses
        else:
            pred_traj = rescale(pred_traj, mode='real')
            return pred_traj

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
        
    def model_predictions(self, x_t, time_cond, pc_tokens, final_pred_joints_actions):
        """
        Get model predictions for the current timestep.
        """
        model_output = self.dit(x=x_t,
                                t=time_cond,
                                y=final_pred_joints_actions,
                                pc_token=pc_tokens,
                            )
        
        pred_noise = model_output.view(x_t.shape[0], x_t.shape[1], -1)  # (batch, max_steps, 8)
        x_start = self.predict_start_from_noise(x_t, time_cond, pred_noise)
        # x_start = torch.clamp(x_start, -1.0, 1.0)
        # pred_noise = self.predict_noise_from_start(x_t, time_cond, x_start)
        return pred_noise, x_start   

    @torch.no_grad()
    def ddim_sample(self, batch, pc_tokens, final_pred_joints_actions, max_steps, batch_size, robot_type, compute_loss=False, **kwargs):
        """
        DDIM Sampling: Deterministic or semi-stochastic sampling depending on eta.
        """
        device = batch['pc_fts'].device

        pc_tokens = torch.stack([torch.mean(x, 0) for x in pc_tokens], 0)
        pc_tokens = pc_tokens.unsqueeze(1).repeat(1, max_steps, 1)

        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps, device=device)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x_t = torch.randn(batch_size, max_steps, robot_type, device=device)  # Start from pure noise
        trajs = [x_t]

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(x_t, time_cond, pc_tokens, final_pred_joints_actions)
            
            if time_next < 0:
                x_t = x_start
                trajs.append(x_t)
                continue
            
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            
            sigma = self.config.dit_config.eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()
            
            noise = torch.randn_like(x_t)
            
            x_t = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            trajs.append(x_t)
        
        x_t = torch.clamp(x_t, -1.0, 1.0)
        pred_traj = x_t.view(-1, 8)

        if compute_loss:
            pad_gt_traj, pad_mask = padding(max_steps, batch, batch_size, robot_type, device)
            pad_gt_traj = pad_gt_traj.view(-1, 8)
            pad_mask = pad_mask.view(-1)

            losses = self.sample_compute_loss(
                pred_traj, pad_gt_traj,
                mask=pad_mask,
            )
            pred_traj = rescale(pred_traj, mode='real')
            return pred_traj, losses
        else:
            pred_traj = rescale(pred_traj, mode='real')
            return pred_traj
        
    def sample_compute_loss(self, pred_traj, gt_traj, mask=None):
        rpred_traj = rescale(pred_traj, mode='real')
        ngt_traj = rescale(gt_traj, mode= 'norm')

        rpred_joint, rpred_open  = rpred_traj[..., :-1], rpred_traj[..., -1]
        npred_joint, npred_open = pred_traj[..., :-1], pred_traj[..., -1]

        rgt_joint, rgt_open = gt_traj[..., :-1], gt_traj[..., -1]
        ngt_joint, ngt_open = ngt_traj[..., :-1], ngt_traj[..., -1]

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
            'total': total_loss
        }










