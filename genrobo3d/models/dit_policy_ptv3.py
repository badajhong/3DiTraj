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

from torch.nn.utils.rnn import pad_sequence
from .dit import DiT

class ActionHead(nn.Module):
    def __init__(
        self, reduce, pos_pred_type, rot_pred_type, hidden_size, dim_actions, 
        dropout=0, voxel_size=0.01, euler_resolution=5, ptv3_config=None, pos_bins=50,
    ) -> None:
        super().__init__()
        assert reduce in ['max', 'mean', 'attn', 'multiscale_max', 'multiscale_max_large']
        assert pos_pred_type in ['heatmap_mlp', 'heatmap_mlp3', 'heatmap_mlp_topk', 'heatmap_mlp_clf', 'heatmap_normmax', 'heatmap_disc']
        assert rot_pred_type in ['quat', 'rot6d', 'euler', 'euler_delta', 'euler_disc']

        self.reduce = reduce
        self.pos_pred_type = pos_pred_type
        self.rot_pred_type = rot_pred_type
        self.hidden_size = hidden_size
        self.dim_actions = dim_actions
        self.voxel_size = voxel_size
        self.euler_resolution = euler_resolution
        self.euler_bins = 360 // euler_resolution
        self.pos_bins = pos_bins

        if self.pos_pred_type == 'heatmap_disc':
            self.heatmap_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.02),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 3 * self.pos_bins * 2)
            )
        else:
            output_size = 1 + 3
            self.heatmap_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.02),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size)
            )

        if self.rot_pred_type == 'euler_disc':
            output_size = self.euler_bins * 3 + 1
        else:
            output_size = dim_actions - 3

        if self.reduce == 'attn':
            output_size += 1
                    
        self.action_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.02),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(
        self, point_embeds, npoints_in_batch, coords=None, temp=1, 
        gt_pos=None, dec_layers_embed=None,
    ):
        '''
        Args:
            point_embeds: (# all points, dim)
            npoints_in_batch: (batch_size, )
            coords: (# all points, 3)
        Return:
            pred_actions: (batch, num_steps, dim_actions)
        ''' 

        if self.pos_pred_type.startswith('heatmap_mlp'):
            heatmap_embeds = self.heatmap_mlp(point_embeds)
            if self.pos_pred_type == 'heatmap_mlp3':
                heatmaps = torch.split(heatmap_embeds[:, :3], npoints_in_batch)
                new_coords = coords + heatmap_embeds[:, 3:]
            else:
                heatmaps = torch.split(heatmap_embeds[:, :1], npoints_in_batch)
                new_coords = coords + heatmap_embeds[:, 1:]
            # temp = 0.01
            heatmaps = [torch.softmax(x / temp, dim=0)for x in heatmaps]
            # print([x.sum() for x in heatmaps], [x.size() for x in heatmaps])
            # print(npoints_in_batch, temp, [x.max() for x in heatmaps], [x.min() for x in heatmaps])
            new_coords = torch.split(new_coords, npoints_in_batch)
            if self.pos_pred_type == 'heatmap_mlp3':
                xt = torch.stack([
                    torch.einsum('pc,pc->c', h, p) for h, p in zip(heatmaps, new_coords)
                ], dim=0)
            else:
                xt = torch.stack([
                    torch.einsum('p,pc->c', h.squeeze(1), p) for h, p in zip(heatmaps, new_coords)
                ], dim=0)
            if self.pos_pred_type == 'heatmap_mlp_topk':
                topk = 20 #min(npoints_in_batch)
                topk_idxs = [torch.topk(x[:, 0], topk)[1] for x in heatmaps]
                topk_xt = torch.stack([x[i] for x, i in zip(new_coords, topk_idxs)], 0)
                # topk_xt = new_coords

            # import numpy as np
            # np.save('debug1.npy', {'coords': coords.data.cpu().numpy(), 'new_coords': new_coords[0].data.cpu().numpy(), 'heatmaps': heatmaps[0].data.cpu().numpy()})

        elif self.pos_pred_type == 'heatmap_disc':
            xt = self.heatmap_mlp(point_embeds) # (npoints, 3*pos_bins)
            xt = einops.rearrange(xt, 'n (c b) -> c n b', c=3) # (3, #npoints, pos_bins)

        if self.reduce == 'max':
            split_point_embeds = torch.split(point_embeds, npoints_in_batch)
            pc_embeds = torch.stack([torch.max(x, 0)[0] for x in split_point_embeds], 0)
            action_embeds = self.action_mlp(pc_embeds)
        elif self.reduce.startswith('multiscale_max'):
            pc_embeds = []
            for dec_layer_embed in dec_layers_embed:
                split_dec_embeds = torch.split(dec_layer_embed.feat, offset2bincount(dec_layer_embed.offset).data.cpu().numpy().tolist())
                pc_embeds.append(
                    F.normalize(torch.stack([torch.max(x, 0)[0] for x in split_dec_embeds], 0), p=2, dim=1)
                )
                # print(torch.stack([torch.max(x, 0)[0] for x in split_dec_embeds], 0).max(), torch.stack([torch.max(x, 0)[0] for x in split_dec_embeds], 0).min())
            pc_embeds = torch.cat(pc_embeds, dim=1)
            action_embeds = self.action_mlp(pc_embeds)
        elif self.reduce == 'mean':
            split_point_embeds = torch.split(point_embeds, npoints_in_batch)
            pc_embeds = torch.stack([torch.mean(x, 0) for x in split_point_embeds], 0)
            action_embeds = self.action_mlp(pc_embeds)
        else: # attn
            action_embeds = self.action_mlp(point_embeds)
            action_heatmaps = torch.split(action_embeds[:, :1], npoints_in_batch)
            action_heatmaps = [torch.softmax(x / temp, dim=0)for x in action_heatmaps]
            split_action_embeds = torch.split(action_embeds[:, 1:], npoints_in_batch)
            action_embeds = torch.stack([(h*v).sum(dim=0) for h, v in zip(action_heatmaps, split_action_embeds)], 0)
            
        if self.rot_pred_type == 'quat':
            xr = action_embeds[..., :4]
            xr = xr / xr.square().sum(dim=-1, keepdim=True).sqrt()
        elif self.rot_pred_type == 'rot6d':
            xr = action_embeds[..., :6]
        elif self.rot_pred_type in ['euler', 'euler_delta']:
            xr = action_embeds[..., :3]
        elif self.rot_pred_type == 'euler_disc':
            xr = action_embeds[..., :self.euler_bins*3].view(-1, self.euler_bins, 3)
        
        xo = action_embeds[..., -1]
        
        if self.pos_pred_type == 'heatmap_mlp_topk':
            return (xt, topk_xt), xr, xo
        else:
            return xt, xr, xo


class SimplePolicyPTV3AdaNorm(BaseModel):
    """Adaptive batch/layer normalization conditioned on text/pose/stepid
    """
    def __init__(self, config):
        super().__init__()

        config.defrost()
        config.ptv3_config.pdnorm_only_decoder = config.ptv3_config.get('pdnorm_only_decoder', False)
        config.freeze()
        
        self.config = config

        self.ptv3_model = PointTransformerV3(**config.ptv3_config)

        act_cfg = config.action_config
        self.txt_fc = nn.Linear(act_cfg.txt_ft_size, act_cfg.context_channels)
        if act_cfg.txt_reduce == 'attn':
            self.txt_attn_fc = nn.Linear(act_cfg.txt_ft_size, 1)
        if act_cfg.use_ee_pose:
            self.pose_embedding = RobotPoseEmbedding(act_cfg.context_channels)
        if act_cfg.use_step_id:
            self.stepid_embedding = nn.Embedding(act_cfg.max_steps, act_cfg.context_channels)

        self.act_proj_head = ActionHead(
            act_cfg.reduce, act_cfg.pos_pred_type, act_cfg.rot_pred_type, 
            config.ptv3_config.dec_channels[0], act_cfg.dim_actions, 
            dropout=act_cfg.dropout, voxel_size=act_cfg.voxel_size,
            ptv3_config=config.ptv3_config, pos_bins=config.action_config.pos_bins,
        )
            
        self.apply(self._init_weights)

        self.rot_transform = RotationMatrixTransform()

        self.dit = DiT(
            in_channels=8,
            out_channels=8,
            hidden_size=128,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            action_dropout_prob=0.1,
            learn_sigma=False,
            max_seq_len=25
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

    def forward(self, batch, compute_loss=False, **kwargs):
        '''batch data:
            pc_fts: (batch, npoints, dim)
            txt_embeds: (batch, txt_dim)
        '''
        batch = self.prepare_batch(batch)

        device = batch['pc_fts'].device

        ptv3_batch = self.prepare_ptv3_batch(batch)

        point_outs = self.ptv3_model(ptv3_batch, return_dec_layers=True) # point_outs[-1].feat: Extract last decoder layer feature from ptv3_model

        # TODO: I think we also have to consider the pc_token + language_token for input to DiT
        # input of DiT & act_proj_head
        pc_token = point_outs[-1].feat 

        # predefine
        npoints_in_batch = batch['npoints_in_batch']
        gt_seq = batch.get('gt_actions', None)
        if gt_seq is not None:
            dim = int(batch['gt_actions'].shape[1])
        else:
            dim = 8
        step_ids = batch['step_ids'] 

        final_step_mask = torch.cat([
            (step_ids[1:] == 0),
            torch.tensor([True], device=step_ids.device)
        ]) # Masking the last steps of task

        max_steps = 25 
        assert max_steps > max(step_ids) -1     # You have to increase number of max_steps

        batch_size = int(final_step_mask.sum().item()) 
        point_dim = int(pc_token.shape[1])

        pc_mask = step_ids == 0

        pc_token_chunks = torch.split(pc_token, npoints_in_batch)
        selected_chunks = [chunk for chunk, use in zip(pc_token_chunks, pc_mask) if use] # List[Tensor of shape [npoints_in_batch, 128]]

        # Max-Pooling
        max_pooling_chunks = [torch.max(chunk, dim=0)[0] for chunk in selected_chunks]
        mask_pc_token = torch.stack(max_pooling_chunks, dim=0)
        
        mask_pc_token = torch.zeros((batch_size, max_steps, point_dim), device=device)
        for i in range(batch_size):
            mask_pc_token[i, :, :] = max_pooling_chunks[i].unsqueeze(0).expand(max_steps, -1) # (batch, max_steps, 128)
          
        mask_npoints_in_batch = torch.tensor(
            [chunk.shape[0] for chunk in selected_chunks],
            device=pc_token.device
        ) 

        padding_mask_npoints_in_batch = torch.tensor(
            [chunk.shape[0] for chunk in selected_chunks for i in range(25)],
            device=pc_token.device
        ) 

        # TODO to predict only the initial task space pc to last trajectory output (could be heatmap)
        pred_actions = self.act_proj_head(
            point_outs[-1].feat, batch['npoints_in_batch'], coords=point_outs[-1].coord,
            temp=self.config.action_config.get('pos_heatmap_temp', 1),
            gt_pos=batch['gt_actions'][..., :3] if 'gt_actions' in batch else None,
            #dec_layers_embed=[point_outs[k] for k in [0, 2, 4, 6, 8]] # TODO
            dec_layers_embed=[point_outs[k] for k in [0, 1, 2, 3, 4]] if self.config.ptv3_config.dec_depths[0] == 1 else [point_outs[k] for k in [0, 2, 4, 6, 8]] # TODO
        )
        
        pred_pos, pred_rot, pred_open = pred_actions
        if self.config.action_config.pos_pred_type == 'heatmap_disc':
            # TODO
            # if not compute_loss:
            if kwargs.get('compute_final_action', True):
                # import time
                # st = time.time()
                cont_pred_pos = []
                npoints_in_batch = offset2bincount(point_outs[-1].offset).data.cpu().numpy().tolist()
                # [(3, npoints, pos_bins)]
                split_pred_pos = torch.split(pred_pos, npoints_in_batch, dim=1)
                split_coords = torch.split(point_outs[-1].coord, npoints_in_batch)
                for i in range(len(npoints_in_batch)):
                    disc_pos_prob = torch.softmax(
                        split_pred_pos[i].reshape(3, -1), dim=-1
                    )
                    cont_pred_pos.append(
                        get_best_pos_from_disc_pos(
                            disc_pos_prob.data.cpu().numpy(), 
                            split_coords[i].data.cpu().numpy(), 
                            best=self.config.action_config.get('best_disc_pos', 'max'),
                            topk=split_coords[i].size(1) * 10,
                            pos_bin_size=self.config.action_config.pos_bin_size, 
                            pos_bins=self.config.action_config.pos_bins, 
                            # best='ens' , topk=1
                        )
                    )
                cont_pred_pos = torch.from_numpy(np.array(cont_pred_pos)).float().to(device)
                # print('time', time.time() - st)
                pred_pos = cont_pred_pos
            else:
                pred_pos = batch['gt_actions'][..., :3]

        if self.config.action_config.rot_pred_type == 'rot6d':
            # no grad
            pred_rot = self.rot_transform.matrix_to_quaternion(
                self.rot_transform.compute_rotation_matrix_from_ortho6d(pred_rot.data.cpu())
            ).float().to(device)
        elif self.config.action_config.rot_pred_type == 'euler':
            pred_rot = pred_rot * 180
            pred_rot = self.rot_transform.euler_to_quaternion(pred_rot.data.cpu()).float().to(device)
        elif self.config.action_config.rot_pred_type == 'euler_delta':
            pred_rot = pred_rot * 180
            cur_euler_angles = R.from_quat(batch['ee_poses'][..., 3:7].data.cpu()).as_euler('xyz', degrees=True)
            pred_rot = pred_rot.data.cpu() + cur_euler_angles
            pred_rot = self.rot_transform.euler_to_quaternion(pred_rot).float().to(device)
        elif self.config.action_config.rot_pred_type == 'euler_disc':
            pred_rot = torch.argmax(pred_rot, 1).data.cpu().numpy()
            pred_rot = np.stack([discrete_euler_to_quaternion(x, self.act_proj_head.euler_resolution) for x in pred_rot], 0)
            pred_rot = torch.from_numpy(pred_rot).to(device)
 
        final_pred_actions = torch.cat([pred_pos, pred_rot, pred_open.unsqueeze(-1)], dim=-1)

        final_pred_actions = final_pred_actions[final_step_mask] # (batch, 8)
        final_gt_actions = batch['gt_actions'][final_step_mask]

        # Define padding of gt_seq, mask, step_ids
        padding_gt_seq = torch.zeros((batch_size, max_steps, dim), device=device)  
        padding_mask = torch.zeros((batch_size, max_steps,), dtype=torch.bool, device=device)
        padding_step_ids = torch.full((batch_size, max_steps,), -1, dtype=torch.long, device=device)
        
        task_ids = torch.cumsum(torch.cat([
                    torch.tensor([1], device=step_ids.device),
                    (step_ids[1:] == 0).int()
                ]), dim=0) - 1
        
        if gt_seq is not None:
            for i in range(gt_seq.shape[0]):  # N = total number of steps
                task = int(task_ids[i].item())
                step = int(step_ids[i].item())
                padding_gt_seq[task, step] = gt_seq[i]
                padding_mask[task, step] = True
                padding_step_ids[task, step] = step
            
            init_seq = torch.randn_like(padding_gt_seq)                            # (batch, max_steps, 8)
        else:
            action_dim= 8 # (x, y, z, qx, qy, qz, qw , open or close)
            init_seq = torch.randn(1, max_steps, action_dim)  

        timesteps = torch.randint(0, 1000, (batch_size,), device=device)       # (batch)

        # pred_actions: (batch*max_steps, 8)
        pred_actions = self.dit(
            x=init_seq,
            t=timesteps,
            y=final_pred_actions,                             # (batch, 8)
            m=padding_mask,                                   # (batch, max_steps, 8)
            s=padding_step_ids,                               # (batch, max_steps, )
            pc_token=mask_pc_token,                           # (batch, max_steps, 128)          
            # npoints_in_batch=mask_npoints_in_batch          # (npoints_in_batch, )    tensor([4096, 1252,  690,  960, 1232, 4096,  687,  678], device='cuda:0')
        )

        padding_gt_seq = padding_gt_seq.contiguous().view(-1, 8)   # (batch * max_steps, 8)
        padding_mask = padding_mask.reshape(-1)   # (batch * max_steps, )
        
        if compute_loss:
            loss_final = self.compute_loss(
                final_pred_actions, final_gt_actions, 
                disc_pos_probs=batch.get('disc_pos_probs', None), 
                npoints_in_batch=mask_npoints_in_batch, 
            )
            loss_seq = self.compute_loss(
                pred_actions, padding_gt_seq, 
                disc_pos_probs=batch.get('disc_pos_probs', None), 
                npoints_in_batch=padding_mask_npoints_in_batch,
                mask = padding_mask
            )
            total_loss = {
                            k: loss_final[k] + loss_seq[k]
                            for k in loss_final
                        }
            return pred_actions, total_loss
        else:
            return pred_actions
        

    def compute_loss(self, pred_actions, tgt_actions, disc_pos_probs=None, npoints_in_batch=None, mask=None):
        """
        Args:
            pred_actions: (batch_size, max_action_len, dim_action)
            tgt_actions: (all_valid_actions, dim_action) / (batch_size, max_action_len, dim_action)
            masks: (batch_size, max_action_len)
        """
        # loss_cfg = self.config.loss_config
        device = tgt_actions.device
        pred_pos, pred_rot, pred_open  = pred_actions[..., :3], pred_actions[..., 3:-1], pred_actions[..., -1]
        tgt_pos, tgt_rot, tgt_open = tgt_actions[..., :3], tgt_actions[..., 3:-1], tgt_actions[..., -1]

        if mask is None:
            # position loss
            if self.config.action_config.pos_pred_type == 'heatmap_disc':
                # pos_loss = F.cross_entropy(
                #     pred_pos.view(-1, 100), disc_pos_probs.view(-1, 100), reduction='mean'
                # )

                # TODO check how it compute heatmap_disc
                # split_pred_pos = torch.split(pred_pos, npoints_in_batch.tolist(), dim=1)
                # pos_loss = 0
                # for i in range(len(npoints_in_batch)):
                #     pos_loss += F.cross_entropy(
                #         split_pred_pos[i].reshape(3, -1), disc_pos_probs[i].to(device), reduction='mean'
                #     )
                # pos_loss /= len(npoints_in_batch)

                pos_loss = F.mse_loss(pred_pos, tgt_pos, reduction='mean')
            else:
                pos_loss = F.mse_loss(pred_pos, tgt_pos, reduction='mean')

            # rotation loss
            if self.config.action_config.rot_pred_type == 'quat':
                # Automatically matching the closest quaternions (symmetrical solution)
                tgt_rot_ = -tgt_rot.clone()
                rot_loss = F.mse_loss(pred_rot, tgt_rot, reduction='none').mean(-1)
                rot_loss_ = F.mse_loss(pred_rot, tgt_rot_, reduction='none').mean(-1)
                select_mask = (rot_loss < rot_loss_).float()
                rot_loss = (select_mask * rot_loss + (1 - select_mask) * rot_loss_).mean()
            elif self.config.action_config.rot_pred_type == 'rot6d':
                tgt_rot6d = self.rot_transform.get_ortho6d_from_rotation_matrix(
                    self.rot_transform.quaternion_to_matrix(tgt_rot.data.cpu())
                ).float().to(device)
                rot_loss = F.mse_loss(pred_rot, tgt_rot6d)
            elif self.config.action_config.rot_pred_type == 'euler':
                # Automatically matching the closest angles
                tgt_rot_ = tgt_rot.clone()
                tgt_rot_[tgt_rot < 0] += 2
                tgt_rot_[tgt_rot > 0] -= 2
                rot_loss = F.mse_loss(pred_rot, tgt_rot, reduction='none')
                rot_loss_ = F.mse_loss(pred_rot, tgt_rot_, reduction='none')
                select_mask = (rot_loss < rot_loss_).float()
                rot_loss = (select_mask * rot_loss + (1 - select_mask) * rot_loss_).mean()
            elif self.config.action_config.rot_pred_type == 'euler_disc':
                tgt_rot = tgt_rot.long()    # (batch_size, 3)
                rot_loss = F.cross_entropy(pred_rot, tgt_rot, reduction='mean')
            else: # euler_delta
                rot_loss = F.mse_loss(pred_rot, tgt_rot)
                
            # openness state loss
            open_loss = F.binary_cross_entropy_with_logits(pred_open, tgt_open, reduction='mean')
            
        else: 

            weights = torch.where(mask == 1, torch.tensor(3, device=mask.device), torch.tensor(1, device=mask.device))

            # position loss
            if self.config.action_config.pos_pred_type == 'heatmap_disc':
                pos_loss = F.mse_loss(pred_pos, tgt_pos, reduction='none').mean(-1)  # (N,)
            else:
                pos_loss = F.mse_loss(pred_pos, tgt_pos, reduction='none').mean(-1)  # (N,)
            # pos_loss = (weights * pos_loss).sum() / (weights.sum() + 1e-6) #too small loss
            pos_loss = (weights * pos_loss).sum()

            # rotation loss
            if self.config.action_config.rot_pred_type == 'quat':
                tgt_rot_ = -tgt_rot.clone()
                rot_loss = F.mse_loss(pred_rot, tgt_rot, reduction='none').mean(-1)
                rot_loss_ = F.mse_loss(pred_rot, tgt_rot_, reduction='none').mean(-1)
                select_mask = (rot_loss < rot_loss_).float()
                rot_loss = (select_mask * rot_loss + (1 - select_mask) * rot_loss_)
            elif self.config.action_config.rot_pred_type == 'rot6d':
                tgt_rot6d = self.rot_transform.get_ortho6d_from_rotation_matrix(
                    self.rot_transform.quaternion_to_matrix(tgt_rot.data.cpu())
                ).float().to(pred_rot.device)
                rot_loss = F.mse_loss(pred_rot, tgt_rot6d, reduction='none').mean(-1)
            elif self.config.action_config.rot_pred_type == 'euler':
                tgt_rot_ = tgt_rot.clone()
                tgt_rot_[tgt_rot < 0] += 2
                tgt_rot_[tgt_rot > 0] -= 2
                rot_loss = F.mse_loss(pred_rot, tgt_rot, reduction='none').mean(-1)
                rot_loss_ = F.mse_loss(pred_rot, tgt_rot_, reduction='none').mean(-1)
                select_mask = (rot_loss < rot_loss_).float()
                rot_loss = (select_mask * rot_loss + (1 - select_mask) * rot_loss_)
            elif self.config.action_config.rot_pred_type == 'euler_disc':
                tgt_rot = tgt_rot.long()
                rot_loss = F.cross_entropy(pred_rot, tgt_rot, reduction='none')  # (N,)
            else:  # euler_delta
                rot_loss = F.mse_loss(pred_rot, tgt_rot, reduction='none').mean(-1)
            # rot_loss = (weights * rot_loss).sum() / (weights.sum() + 1e-6)
            rot_loss = (weights * rot_loss).sum()
            # openness loss
            open_loss = F.binary_cross_entropy_with_logits(pred_open, tgt_open, reduction='none')  # (N,)
            # open_loss = (weights * open_loss).sum() / (weights.sum() + 1e-6)
            open_loss = (weights * open_loss).sum()

        total_loss = self.config.loss_config.pos_weight * pos_loss + \
                    self.config.loss_config.rot_weight * rot_loss + open_loss
        
        return {
            'pos': pos_loss, 'rot': rot_loss, 'open': open_loss, 
            'total': total_loss
        }


class SimplePolicyPTV3CA(SimplePolicyPTV3AdaNorm):
    """Cross attention conditioned on text/pose/stepid
    """
    def __init__(self, config):
        super().__init__(config) # TODO check the change of # BaseModel.__init__(self)

        self.config = config

        self.ptv3_model = PointTransformerV3CA(**config.ptv3_config)

        act_cfg = config.action_config
        self.txt_fc = nn.Linear(act_cfg.txt_ft_size, act_cfg.context_channels)
        if act_cfg.use_ee_pose:
            self.pose_embedding = RobotPoseEmbedding(act_cfg.context_channels)
        if act_cfg.use_step_id:
            self.stepid_embedding = nn.Embedding(act_cfg.max_steps, act_cfg.context_channels)
        self.act_proj_head = ActionHead(
            act_cfg.reduce, act_cfg.pos_pred_type, act_cfg.rot_pred_type, 
            config.ptv3_config.dec_channels[0], act_cfg.dim_actions, 
            dropout=act_cfg.dropout, voxel_size=act_cfg.voxel_size,
            ptv3_config=config.ptv3_config, pos_bins=config.action_config.pos_bins,
        )

        self.apply(self._init_weights)

        self.rot_transform = RotationMatrixTransform()

    def prepare_ptv3_batch(self, batch):
        outs = {
            'coord': batch['pc_fts'][:, :3],
            'grid_size': self.config.action_config.voxel_size,
            'offset': batch['offset'],
            'batch': offset2batch(batch['offset']),
            'feat': batch['pc_fts'],
        }
        device = batch['pc_fts'].device

        # encode context for each point cloud
        txt_embeds = self.txt_fc(batch['txt_embeds'])
        ctx_embeds = torch.split(txt_embeds, batch['txt_lens'])
        ctx_lens = torch.LongTensor(batch['txt_lens'])

        if self.config.action_config.use_ee_pose:
            pose_embeds = self.pose_embedding(batch['ee_poses'])
            ctx_embeds = [torch.cat([c, e.unsqueeze(0)], dim=0) for c, e in zip(ctx_embeds, pose_embeds)]
            ctx_lens += 1

        if self.config.action_config.use_step_id:
            step_embeds = self.stepid_embedding(batch['step_ids'])
            ctx_embeds = [torch.cat([c, e.unsqueeze(0)], dim=0) for c, e in zip(ctx_embeds, step_embeds)]
            ctx_lens += 1

        outs['context'] = torch.cat(ctx_embeds, 0)
        outs['context_offset'] = torch.cumsum(ctx_lens, dim=0).to(device)

        return outs
    
if __name__ == '__main__':
    from genrobo3d.configs.default import get_config

    config = get_config('genrobo3d/configs/rlbench/simple_policy_ptv3.yaml')
    model = SimplePolicyPTV3AdaNorm(config.MODEL).cuda()
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
