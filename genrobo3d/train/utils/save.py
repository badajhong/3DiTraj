"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

saving utilities
"""
import json
import os
import torch


def save_training_meta(args):
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ckpts'), exist_ok=True)

    with open(os.path.join(args.output_dir, 'logs', 'training_config.yaml'), 'w') as writer:
        args_str = args.dump()
        print(args_str, file=writer)

class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_step', suffix='pt'):
        self.output_dir = output_dir
        self.prefix = prefix
        self.suffix = suffix

    def save(self, model, step, optimizer=None, ema_model=None, rewrite_optimizer=False):
        # --- Save the standard model (no changes here) ---
        output_model_file = os.path.join(self.output_dir,
                                 f"{self.prefix}_{step}.{self.suffix}")
        state_dict = {}
        for k, v in model.state_dict().items():
            if k.startswith('module.'):
                k = k[7:]
            state_dict[k] = v.cpu() if isinstance(v, torch.Tensor) else v
        torch.save(state_dict, output_model_file)

        # --- ADD THIS BLOCK TO SAVE THE EMA MODEL ---
        if ema_model is not None:
            output_ema_file = os.path.join(self.output_dir,
                                     f"ema_{self.prefix}_{step}.{self.suffix}")
            ema_state_dict = {}
            for k, v in ema_model.state_dict().items():
                if k.startswith('module.'):
                    k = k[7:]
                ema_state_dict[k] = v.cpu() if isinstance(v, torch.Tensor) else v
            torch.save(ema_state_dict, output_ema_file)
            
        # --- Save the optimizer state (no changes here) ---
        if optimizer is not None:
            dump = {'step': step, 'optimizer': optimizer.state_dict()}
            if hasattr(optimizer, '_amp_stash'):
                pass  # TODO fp16 optimizer
            if rewrite_optimizer:
                torch.save(dump, f'{self.output_dir}/train_state_latest.pt')
            else:
                torch.save(dump, f'{self.output_dir}/train_state_{step}.pt')