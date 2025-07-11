import os
import sys
import json
import argparse
import time
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
from functools import partial

import warnings
warnings.filterwarnings("ignore")

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader

from genrobo3d.train.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from genrobo3d.train.utils.save import ModelSaver, save_training_meta
from genrobo3d.train.utils.misc import NoOp, set_dropout, set_random_seed
from genrobo3d.train.utils.distributed import set_cuda, wrap_model, all_gather
from genrobo3d.train.utils.utils import requires_grad, update_ema

from genrobo3d.train.optim import get_lr_sched, get_lr_sched_decay_rate
from genrobo3d.train.optim.misc import build_optimizer

from genrobo3d.configs.default import get_config

from genrobo3d.train.datasets.loader import build_dataloader
from genrobo3d.train.datasets.ptv3_dit_policy_dataset import (
    ptv3_dit_PolicyDataset, base_collate_fn, ptv3_collate_fn
)

from genrobo3d.models.chidit import DiTraj
from genrobo3d.models.DiT.diffusion import create_diffusion

from genrobo3d.models.stage2_ptv3 import SimplePolicyPTV3AdaNorm, SimplePolicyPTV3CA  # or AdaNorm

DATASET_FACTORY = {
    'DiTraj': (ptv3_dit_PolicyDataset, ptv3_collate_fn),
    # 'mmDiTraj': (ptv3_dit_PolicyDataset, ptv3_collate_fn),
}

MODEL_FACTORY = {
    'SimplePolicyPTV3AdaNorm': SimplePolicyPTV3AdaNorm,
    'SimplePolicyPTV3CA': SimplePolicyPTV3CA,
    'DiTraj': DiTraj,
    # 'mmDiTraj': (ptv3_dit_PolicyDataset, ptv3_collate_fn),
}

def main(config):
    config.defrost()
    default_gpu, n_gpu, device = set_cuda(config)

    # wandb
    if default_gpu:
        wandb.init(
            project=config.wandb_project,  
            name=config.wandb_run_name,    
            config=config,                 
        )

    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}'.format(
                device, n_gpu, bool(config.local_rank != -1)
            )
        )

    seed = config.SEED
    if config.local_rank != -1:
        seed += config.rank
    set_random_seed(seed)

    # load data training set
    dataset_class, dataset_collate_fn = DATASET_FACTORY[config.MODEL.model_class]
    trn_dataset = dataset_class(**config.TRAIN_DATASET)
    LOGGER.info(f'#num_train: {len(trn_dataset)}')
    trn_dataloader, pre_epoch = build_dataloader(
        trn_dataset, dataset_collate_fn, True, config
    )

    if config.VAL_DATASET.use_val:
        val_dataset = dataset_class(**config.VAL_DATASET)
        LOGGER.info(f"#num_val: {len(val_dataset)}")
        val_dataloader = DataLoader(
            val_dataset, batch_size=config.TRAIN.val_batch_size, shuffle=False,
            num_workers=config.TRAIN.n_workers, pin_memory=True, collate_fn=dataset_collate_fn, drop_last=True
        )
    else:
        val_dataloader = None

    LOGGER.info(f'#num_steps_per_epoch: {len(trn_dataloader)}')
    if config.TRAIN.num_train_steps is None:
        config.TRAIN.num_train_steps = len(trn_dataloader) * config.TRAIN.num_epochs
    else:
        # assert config.TRAIN.num_epochs is None, 'cannot set num_train_steps and num_epochs at the same time.'
        config.TRAIN.num_epochs = int(np.ceil(config.TRAIN.num_train_steps / len(trn_dataloader)))
        
    if config.TRAIN.gradient_accumulation_steps > 1:
        config.TRAIN.num_train_steps *= config.TRAIN.gradient_accumulation_steps
        config.TRAIN.num_epochs *= config.TRAIN.gradient_accumulation_steps

    # setup loggers
    if default_gpu:
        save_training_meta(config)
        # TB_LOGGER.create(os.path.join(config.output_dir, 'logs'))
        if config.tfboard_log_dir is None:
            output_dir_tokens = config.output_dir.split('/')
            config.tfboard_log_dir = os.path.join(output_dir_tokens[0], 'TFBoard', *output_dir_tokens[1:])
        TB_LOGGER.create(config.tfboard_log_dir)
        model_saver = ModelSaver(os.path.join(config.output_dir, 'ckpts'))
        add_log_to_file(os.path.join(config.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        model_saver = NoOp()

    # Prepare model
    model_class = MODEL_FACTORY[config.MODEL.model_class]
    model = model_class(config.MODEL)
    model.to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    # Create and load PTV3 model
    ptv3_class = MODEL_FACTORY[config.MODEL.ptv3_model_class]  # NOTE: not under config.MODEL
    ptv3_model = ptv3_class(config.MODEL)

    # Load the PTV3 checkpoint
    ptv3_ckpt_path = os.path.join(config.ptv3_output_dir)
    ptv3_ckpt = torch.load(ptv3_ckpt_path, map_location='cpu')

    # Filter and load matching keys from the checkpoint
    ptv3_state_dict = ptv3_model.state_dict()
    filtered_ckpt = {k: v for k, v in ptv3_ckpt.items() if k in ptv3_state_dict and v.size() == ptv3_state_dict[k].size()}
    ptv3_model.load_state_dict(filtered_ckpt, strict=False)

    # Freeze the PTV3 model
    ptv3_model.eval()
    for param in ptv3_model.parameters():
        param.requires_grad = False

    # Move to device
    ptv3_model = ptv3_model.to(device)

    # DDP: SyncBN
    if config.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Fix parameters
    if config.TRAIN.freeze_params.encoder:
        for param_name, param in model.named_parameters():
            if param_name.startswith('mae_encoder') and 'decoder_block' not in param_name:
                    param.requires_grad = False

    LOGGER.info("Model: nweights %d nparams %d" % (model.num_parameters))
    LOGGER.info("Model: trainable nweights %d nparams %d" % (model.num_trainable_parameters))
    
    config.freeze()

    # Load from checkpoint
    model_checkpoint_file = config.checkpoint
    optimizer_checkpoint_file = os.path.join(
        config.output_dir, 'ckpts', 'train_state_latest.pt'
    )
    if os.path.exists(optimizer_checkpoint_file) and config.TRAIN.resume_training:
        LOGGER.info('Load the optimizer checkpoint from %s' % optimizer_checkpoint_file)
        optimizer_checkpoint = torch.load(
            optimizer_checkpoint_file, map_location=lambda storage, loc: storage
        )
        lastest_model_checkpoint_file = os.path.join(
            config.output_dir, 'ckpts', 'model_step_%d.pt' % optimizer_checkpoint['step']
        )
        if os.path.exists(lastest_model_checkpoint_file):
            LOGGER.info('Load the model checkpoint from %s' % lastest_model_checkpoint_file)
            model_checkpoint_file = lastest_model_checkpoint_file
        global_step = optimizer_checkpoint['step']
        restart_epoch = global_step // len(trn_dataloader)
    else:
        optimizer_checkpoint = None
        # to compute training statistics
        restart_epoch = 0
        global_step = restart_epoch * len(trn_dataloader) 

    if model_checkpoint_file is not None:
        checkpoint = torch.load(
            model_checkpoint_file, map_location=lambda storage, loc: storage)
        LOGGER.info('Load the model checkpoint (%d params)' % len(checkpoint))
        new_checkpoint = {}
        state_dict = model.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
                # TODO: mae_encoder.encoder.first_conv.0.weight
                if k == 'mae_encoder.encoder.first_conv.0.weight':
                    if v.size(1) != state_dict[k].size(1):
                        new_checkpoint[k] = torch.zeros_like(state_dict[k])
                        min_v_size = min(v.size(1), state_dict[k].size(1))
                        new_checkpoint[k][:, :min_v_size] = v[:, :min_v_size]
                if v.size() == state_dict[k].size():
                    if config.TRAIN.resume_encoder_only and (k.startswith('mae_decoder') or 'decoder_block' in k):
                        continue
                    new_checkpoint[k] = v
        LOGGER.info('Resumed the model checkpoint (%d params)' % len(new_checkpoint))
        model.load_state_dict(new_checkpoint, strict=config.checkpoint_strict_load)

    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()
    # set_dropout(model, config.TRAIN.dropout)
    model = wrap_model(model, device, config.local_rank, find_unused_parameters=True)

    # Prepare optimizer
    optimizer, init_lrs = build_optimizer(model, config.TRAIN)
    if optimizer_checkpoint is not None:
        optimizer.load_state_dict(optimizer_checkpoint['optimizer'])

    if default_gpu:
        pbar = tqdm(initial=global_step, total=config.TRAIN.num_train_steps)
    else:
        pbar = NoOp()

    LOGGER.info(f"***** Running training with {config.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", config.TRAIN.train_batch_size if config.local_rank == -1 
                else config.TRAIN.train_batch_size * config.world_size)
    LOGGER.info("  Accumulate steps = %d", config.TRAIN.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", config.TRAIN.num_train_steps)

    optimizer.zero_grad()
    optimizer.step()

    running_metrics = {}

    best_val_step, best_val_metric = None, np.inf

    epoch_id = 0
    for epoch_id in range(restart_epoch, config.TRAIN.num_epochs):
        if global_step >= config.TRAIN.num_train_steps:
            break

        # In distributed mode, calling the set_epoch() method at the beginning of each epoch
        pre_epoch(epoch_id)
        lr_this_step = 0.0
        
        for step, batch in enumerate(trn_dataloader):
            _, final_pred_joints_actions, pc_tokens, _ = ptv3_model(batch, compute_loss=False, global_step=None)

            # forward pass
            losses = model(batch, 
                              pc_tokens,
                              final_pred_joints_actions,
                              max_steps=config.MODEL.action_config.max_steps, 
                              batch_size=config.TRAIN.train_batch_size, 
                              robot_type = config.MODEL.action_config.robot_type, 
                              mode='train', 
                              compute_final_action=False)

            # backward pass
            if config.TRAIN.gradient_accumulation_steps > 1:  # average loss
                losses = losses / config.TRAIN.gradient_accumulation_steps
            losses.backward()

            loss_value = losses.item()
            loss_key_name = 'total'

            # TensorBoard에 기록
            TB_LOGGER.add_scalar(f'step/loss_{loss_key_name}', loss_value, global_step)

            # W&B에 기록
            if default_gpu:
                wandb.log({f"loss/{loss_key_name}": loss_value, "step": global_step})

            # Running Metrics 업데이트
            running_metrics.setdefault(f'loss_{loss_key_name}', RunningMeter(f'loss_{loss_key_name}'))
            running_metrics[f'loss_{loss_key_name}'](loss_value)

            # optimizer update and logging
            if (step + 1) % config.TRAIN.gradient_accumulation_steps == 0:
                global_step += 1
                # learning rate scheduling
                lr_decay_rate = get_lr_sched_decay_rate(global_step, config.TRAIN)
                for kp, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = lr_this_step = max(init_lrs[kp] * lr_decay_rate, 1e-8)
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.step()

                # update model params
                if config.TRAIN.grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.grad_norm
                    )
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                update_ema(ema, model)
                optimizer.zero_grad()
                pbar.update(1)

            if global_step % config.TRAIN.log_steps == 0:
                # monitor training throughput
                LOGGER.info(
                    f'==============Epoch {epoch_id} Step {global_step}===============')
                LOGGER.info(', '.join(['%s:%.4f' % (lk, lv.val) for lk, lv in running_metrics.items()]))
                LOGGER.info('===============================================')                

            if global_step % config.TRAIN.save_steps == 0:
                model_saver.save(model, global_step, optimizer=optimizer, ema_model=ema, rewrite_optimizer=True)

            if (val_dataloader is not None) and (global_step % config.TRAIN.val_steps == 0):
                # val_metrics = validate(ema, val_dataloader) 
                val_metrics = validate(model, val_dataloader) 

                if default_gpu:
                    val_log_data = {
                        f"val/{lk}": lv for lk, lv in val_metrics.items()
                    }
                    val_log_data["epoch"] = epoch_id
                    wandb.log(val_log_data, step=global_step)

                LOGGER.info(f'=================Validation=================')
                metric_str = ', '.join([f'{lk}: {lv:.4f}' for lk, lv in val_metrics.items()])
                LOGGER.info(metric_str)
                LOGGER.info('===============================================')
                
                if val_metrics['joint_loss'] < best_val_metric:
                    best_val_metric = val_metrics['joint_loss']
                    best_val_step = global_step
                model.train()
            

            if global_step >= config.TRAIN.num_train_steps:
                break
            
     
        
        # Reset running metrics for the next epoch
      


    if global_step % config.TRAIN.save_steps != 0:
        LOGGER.info(f'==============Final Save and Validation at Step {global_step}===============')
        model_saver.save(model, global_step, optimizer=optimizer, ema_model=ema, rewrite_optimizer=True)

        val_metrics = validate(ema, val_dataloader, config, device)
        LOGGER.info(f'=================Final Validation=================')
        metric_str = ', '.join([f'{lk}: {lv:.4f}' for lk, lv in val_metrics.items()])
        LOGGER.info(metric_str)
        LOGGER.info('===============================================')
        
        if val_metrics['joint_loss'] < best_val_metric:
            best_val_metric = val_metrics['joint_loss']
            best_val_step = global_step
        
        if default_gpu:
            final_val_log = {f"final_val/{lk}": lv for lk, lv in val_metrics.items()}
            final_val_log["epoch"] = epoch_id
            wandb.log(final_val_log, step=global_step)

    LOGGER.info(f'Validation: Best loss: {best_val_metric:.4f} at step {best_val_step}')
    if default_gpu:
        wandb.finish()

@torch.no_grad()
def validate(model, val_dataloader):
    model.eval()
    model.diffusion = create_diffusion(str(config.MODEL.dit_config.sampling_timesteps), diffusion_steps=config.MODEL.dit_config.num_timesteps)
    open_loss, joint_loss, real_joint_loss, total_loss, num_examples, num_batches = 0, 0, 0, 0, 0, 0

    # Create and load PTV3 model
    ptv3_class = MODEL_FACTORY[config.MODEL.ptv3_model_class]
    ptv3_model = ptv3_class(config.MODEL)

    # Load the PTV3 checkpoint
    ptv3_ckpt_path = os.path.join(config.ptv3_output_dir)
    ptv3_ckpt = torch.load(ptv3_ckpt_path, map_location='cpu')

    # Filter and load matching keys from the checkpoint
    ptv3_state_dict = ptv3_model.state_dict()
    filtered_ckpt = {k: v for k, v in ptv3_ckpt.items() if k in ptv3_state_dict and v.size() == ptv3_state_dict[k].size()}
    ptv3_model.load_state_dict(filtered_ckpt, strict=False)

    # Freeze the PTV3 model
    ptv3_model.eval()
    for param in ptv3_model.parameters():
        param.requires_grad = False

    # Move to device
    ptv3_model = ptv3_model.to(next(model.parameters()).device)

    for batch in val_dataloader:

        _, final_pred_joints_actions, pc_tokens, _ = ptv3_model(batch, compute_loss=False, global_step=None)
        
        
        val_loss = model(batch, 
                        pc_tokens,
                        final_pred_joints_actions,
                        max_steps=config.MODEL.action_config.max_steps, 
                        batch_size=config.TRAIN.train_batch_size, 
                        robot_type = config.MODEL.action_config.robot_type, 
                        mode='val', 
                        compute_final_action=False)
        
        joint_loss += val_loss['joint'].item()
        real_joint_loss += val_loss['real_joint'].item()
        open_loss += val_loss['open'].item()
        total_loss += val_loss['total'].item() 

        num_batches += 1
        
    return {
        'total_loss': total_loss / num_batches, 
        'joint_loss': joint_loss / num_batches,
        'real_joint_loss': real_joint_loss / num_batches,
        'open_loss': open_loss / num_batches,
        # 'open_acc': open_acc / num_examples,
    }


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line (use , to separate values in a list)",
    )
    args = parser.parse_args()

    config = get_config(args.exp_config, args.opts)

    if os.path.exists(config.output_dir) and os.listdir(config.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                config.output_dir
            )
        )

    return config

if __name__ == '__main__':
    # ======= VSCode Debug: Hardcoded Args ==========
    exp_config_path = "genrobo3d/configs/rlbench/stage2_dit.yaml"
    opts = [
    ]
    # ===============================================

    from genrobo3d.configs.default import get_config
    config = get_config(exp_config_path, opts)

    main(config)