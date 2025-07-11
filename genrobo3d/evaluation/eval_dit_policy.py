from typing import Tuple, Dict, List

import os
import ctypes

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/uhcc/Desktop/robot-3dlotus/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
os.environ['LD_LIBRARY_PATH'] = "/home/uhcc/Desktop/robot-3dlotus/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04:" + os.environ.get('LD_LIBRARY_PATH', '')
ctypes.CDLL("/home/uhcc/Desktop/robot-3dlotus/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/libcoppeliaSim.so.1")

import json
import jsonlines
import tap
import copy
from pathlib import Path
from filelock import FileLock

import torch
import numpy as np
from scipy.special import softmax

try:
    import open3d as o3d
except ImportError:
    print("Open3D could not be imported.")
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.transform import Rotation as R

from genrobo3d.train.utils.misc import set_random_seed
from genrobo3d.configs.default import get_config

try:
    from genrobo3d.rlbench.environments import RLBenchEnv
except:
    print('No RLBench')

from genrobo3d.train.train_dit_policy import MODEL_FACTORY
from genrobo3d.configs.rlbench.constants import get_robot_workspace, get_rlbench_labels
from genrobo3d.utils.robot_box import RobotBox
from genrobo3d.train.datasets.common import gen_seq_masks
from genrobo3d.evaluation.common import write_to_file
from genrobo3d.vlm_models.clip_encoder import ClipEncoder

from genrobo3d.models.dit_policy_ptv3 import SimplePolicyPTV3CA

class Arguments(tap.Tap):
    exp_config: str
    device: str = 'cuda'
    microstep_data_dir: str = ''
    seed: int = 100
    num_demos: int = 20
    taskvar: str = 'push_button+0'
    checkpoint: str = None
    headless: bool = False
    max_tries: int = 10
    max_steps: int = 25
    cam_rand_factor: float = 0.0
    image_size: List[int] = [256, 256]
    save_image: bool = False
    save_obs_outs_dir: str = None
    record_video: bool = False
    not_include_robot_cameras: bool = False
    video_rotate_cam: bool = False
    video_resolution: int = 480
    num_ensembles: int = 1
    best_disc_pos: str = 'max'
    real_robot: bool = False

class Actioner(object):
    def __init__(self, args) -> None:
        self.args = args
        self.WORKSPACE = get_robot_workspace(real_robot=args.real_robot)
        self.device = torch.device(args.device)

        config = get_config(args.exp_config, args.remained_args)
        self.config = config
        self.config.defrost()
        self.config.MODEL.action_config.best_disc_pos = args.best_disc_pos
        
        self.config.freeze()

        self.model = SimplePolicyPTV3CA(config.MODEL).to(self.device)
        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            self.model.load_state_dict(checkpoint, strict=True)

        self.model.eval()
        self.clip_model = ClipEncoder()

        if os.path.exists(config.TRAIN_DATASET.instr_embed_file):
            self.instr_embeds = np.load(config.TRAIN_DATASET.instr_embed_file, allow_pickle=True).item()
        else:
            self.instr_embeds = {}
        if config.TRAIN_DATASET.instr_embed_type == 'last':
            self.instr_embeds = {k: v[-1:] for k, v in self.instr_embeds.items()}

        self.taskvar_instrs = json.load(open(config.TRAIN_DATASET.taskvar_instr_file))
        self.TABLE_HEIGHT = self.WORKSPACE['TABLE_HEIGHT']

    def preprocess_obs(self, taskvar, step_id, obs, instructions):
        instr = instructions[0]
        if instr not in self.instr_embeds:
            embed = self.clip_model('text', instr, use_prompt=False, output_hidden_states=True)[0].cpu().numpy()
            if self.config.TRAIN_DATASET.instr_embed_type == 'last':
                embed = embed[-1:]
            self.instr_embeds[instr] = embed

        instr_embed = self.instr_embeds[instr]

        xyz = np.stack(obs['pc'], 0).reshape(-1, 3)
        rgb = np.stack(obs['rgb'], 0).reshape(-1, 3)
        ee_pose = obs['gripper']
        pc_ft = np.concatenate([(xyz - xyz.mean(0)) / np.linalg.norm(xyz - xyz.mean(0)), rgb / 255.], axis=1)

        batch = {
            'pc_fts': torch.from_numpy(pc_ft).float(),
            'ee_poses': torch.from_numpy(np.array(ee_pose)).float().unsqueeze(0),
            'step_ids': torch.LongTensor([step_id]),
            'txt_embeds': torch.from_numpy(instr_embed).float(),
            'txt_lens': [instr_embed.shape[0]],
            'npoints_in_batch': [pc_ft.shape[0]],
            'offset': torch.LongTensor([pc_ft.shape[0]]),
            'pc_centroids': xyz.mean(0),
            'pc_radius': np.linalg.norm(xyz - xyz.mean(0)),
        }
        return batch

    def predict(self, task_str, variation, step_id, obs_state_dict, episode_id, instructions):
        taskvar = f'{task_str}+{variation}'
        batch = self.preprocess_obs(taskvar, step_id, obs_state_dict, instructions)
        device = self.device

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        with torch.no_grad():
            output = self.model(batch=batch)[0].data.cpu().numpy()

        output[-1] = 1.0 if torch.sigmoid(torch.tensor(output[-1])) > 0.5 else 0.0
        output[:3] = output[:3] * batch['pc_radius'] + batch['pc_centroids']
        output[2] = max(output[2], self.TABLE_HEIGHT + 0.005)

        return {'action': output}


def evaluate_actioner(args):
    set_random_seed(args.seed)
    actioner = Actioner(args)

    pred_dir = os.path.join(actioner.config.output_dir, 'preds', f'seed{args.seed}')
    if args.cam_rand_factor > 0:
        pred_dir = f"{pred_dir}-cam_rand_factor{args.cam_rand_factor:.1f}"
    os.makedirs(pred_dir, exist_ok=True)

    outfile = os.path.join(pred_dir, 'results.jsonl')
    existed_data = set()
    if os.path.exists(outfile):
        with jsonlines.open(outfile, 'r') as f:
            for item in f:
                existed_data.add((item['checkpoint'], f"{item['task']}+{item['variation']}"))

    if (args.checkpoint, args.taskvar) in existed_data:
        print(">> Skipping evaluation: already in results file")
        return

    env = RLBenchEnv(
        data_path=args.microstep_data_dir,
        apply_rgb=True,
        apply_pc=True,
        apply_mask=True,
        headless=args.headless,
        image_size=args.image_size,
        cam_rand_factor=args.cam_rand_factor,
    )

    task_str, variation = args.taskvar.split('+')
    variation = int(variation)

    success_rate = env.evaluate(
        task_str, variation,
        actioner=actioner,
        max_episodes=args.max_steps,
        num_demos=args.num_demos,
        log_dir=Path(pred_dir),
        max_tries=args.max_tries,
        save_image=args.save_image,
        record_video=args.record_video,
        include_robot_cameras=(not args.not_include_robot_cameras),
        video_rotate_cam=args.video_rotate_cam,
        video_resolution=args.video_resolution,
    )

    print(f"Testing Success Rate {task_str}: {success_rate:.04f}")
    write_to_file(
        outfile,
        {
            'checkpoint': args.checkpoint,
            'task': task_str,
            'variation': variation,
            'num_demos': args.num_demos,
            'sr': success_rate
        }
    )


if __name__ == '__main__':
    args = Arguments().parse_args(known_only=True)
    args.remained_args = args.extra_args

    print(">> About to create RLBenchEnv")
    env = RLBenchEnv(
        data_path=args.microstep_data_dir,
        apply_rgb=True,
        apply_pc=True,
        apply_mask=True,
        headless=args.headless,
        image_size=args.image_size,
        cam_rand_factor=args.cam_rand_factor,
    )
    print(">> RLBenchEnv created successfully")

    evaluate_actioner(args)
