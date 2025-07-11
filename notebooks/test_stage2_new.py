import os
import ctypes
import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict

from genrobo3d.rlbench.environments_joint import RLBenchEnv, Mover
from genrobo3d.train.utils.misc import set_random_seed
from genrobo3d.evaluation.eval_stage2_test import Actioner
from rlbench.backend.utils import task_file_to_task_class

from genrobo3d.models.stage2_ptv3 import SimplePolicyPTV3CA
from genrobo3d.models.stage2_dit import DiTraj
from genrobo3d.configs.default import get_config

# PTV3 model is required to generate tokens
ptv3_ckpt_path = 'data/experiments/gembench/stage2_ptv3_0617/ckpts/model_step_29000.pt'
ptv3_cfg_path = "data/experiments/gembench/stage2_ptv3_0617/logs/training_config.yaml"

dit_ckpt_path = 'data/experiments/gembench/stage2_dit_0703/ckpts/model_step_100000.pt'
dit_cfg_path = "data/experiments/gembench/stage2_dit_0703/logs/training_config.yaml"


# Load the PTV3 config
ptv3_cfg = get_config(ptv3_cfg_path, [])
dit_cfg = get_config(dit_cfg_path, [])


# Preload CoppeliaSim lib
coppelia_path = "/home/uhcc/Desktop/robot-3dlotus/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = coppelia_path
os.environ['LD_LIBRARY_PATH'] = f"{coppelia_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
ctypes.CDLL(f"{coppelia_path}/libcoppeliaSim.so.1")

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ptv3_model = SimplePolicyPTV3CA(ptv3_cfg.MODEL)
ptv3_ckpt = torch.load(ptv3_ckpt_path, map_location='cuda')
ptv3_model.load_state_dict(ptv3_ckpt, strict=False)
ptv3_model.to(device).eval()

dit_model = DiTraj(dit_cfg.MODEL)
dit_ckpt = torch.load(dit_ckpt_path, map_location='cuda')
dit_model.load_state_dict(dit_ckpt, strict=False)
dit_model.to(device).eval()

args = EasyDict({
    'exp_config': ptv3_cfg_path,
    'checkpoint': ptv3_ckpt_path,
    'device': device,
    'real_robot': False,
    'save_obs_outs_dir': None,
    'best_disc_pos': 'max',
    'num_ensembles': 1,
    'remained_args': [],
})

full_actioner = Actioner(args)

def move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch

class SimpleActioner:
    def __init__(self, policy_model, ptv3_model, full_actioner, device='cuda'):
        self.policy = policy_model.ddim_sample
        self.ptv3 = ptv3_model.eval().to(device)
        self.device = device
        self.full_actioner = full_actioner

    @torch.no_grad()
    def predict(self, **kwargs):
        # Use full Actioner preprocessing
        taskvar = f"{kwargs['task_str']}+{kwargs['variation']}"
        step_id = kwargs['step_id']
        obs_state_dict = kwargs['obs_state_dict']
        instructions = kwargs['instructions']

        batch = full_actioner.preprocess_obs(taskvar, step_id, obs_state_dict, instructions)
 
        batch = move_to_device(batch, self.device)

        _, final_pred_joints_actions, pc_tokens, _ = self.ptv3(batch, compute_loss=False, global_step=None)
        output = self.policy(batch, 
                              pc_tokens,
                              final_pred_joints_actions,
                              max_steps=30, 
                              batch_size=1, 
                              robot_type =8, 
                              compute_loss=False, 
                              compute_final_action=False)
        return output.cpu().numpy()
    
# Seed for reproducibility
set_random_seed(10)

# taskvar = 'open_door+0'
taskvar = 'close_microwave+0'
# taskvar = 'pick_and_lift+0'
# taskvar = 'close_jar_peract+16'
taskvar = 'pick_up_cup+8'

taskvar = 'push_button+0'
# taskvar = 'close_fridge+0'
# taskvar = 'close_laptop_lid+0'
# taskvar = 'open_box+0'

task_str, variation_id = taskvar.split('+')
variation_id = int(variation_id)
image_size = [256, 256]

# Launch environment
env = RLBenchEnv(
    data_path='',
    apply_rgb=True,
    apply_pc=True,
    apply_mask=True,
    headless=False,
    image_size=image_size,
    cam_rand_factor=0,
)
env.env.launch()
task_type = task_file_to_task_class(task_str)
task = env.env.get_task(task_type)
task.set_variation(variation_id)

mover = Mover(task, max_tries=10)
instructions, obs = task.reset()
print('Instructions:', instructions)

# Show initial observation
rgb_combined = np.concatenate([
    obs.left_shoulder_rgb,
    obs.right_shoulder_rgb,
    obs.wrist_rgb,
    obs.front_rgb
], axis=1)
# plt.imshow(rgb_combined.astype(np.uint8))
# plt.axis('off')
# plt.show()

# Extract obs and reset mover
obs_state_dict = env.get_observation(obs)
mover.reset(obs_state_dict['gripper'])

actioner = SimpleActioner(dit_model, ptv3_model, full_actioner, device=device)

# Inference loop
reward = 0
demo_id = 0

batch = {
    'task_str': task_str,
    'variation': variation_id,
    'step_id': 0,
    'obs_state_dict': obs_state_dict,
    'episode_id': demo_id,
    'instructions': instructions,
}

output = actioner.predict(**batch)



for step_id in range(30):

    action = output[step_id]
    action[-1] = 1 if action[-1] > 0 else 0

    print (action)


    if action is None:
        print("No action predicted, breaking.")
        break

    try:
        obs, reward, terminate, _ = mover(action, verbose=True)
        print(f'Step id: {step_id + 1}')
        rgb_combined = np.concatenate([
            obs.left_shoulder_rgb,
            obs.right_shoulder_rgb,
            obs.wrist_rgb,
            obs.front_rgb
        ], axis=1)
        # plt.imshow(rgb_combined.astype(np.uint8))
        # plt.axis('off')
        # plt.show()

        obs_state_dict = env.get_observation(obs)

        if reward == 1:
            print("Task completed successfully.")
            break
        if terminate:
            print("Episode terminated.")
            break

    except (IKError, ConfigurationPathError, InvalidActionError) as e:
        print(f"{taskvar}, demo {demo_id}, step {step_id}: Error - {e}")
        reward = 0
        break

print('Final Reward:', reward)
env.env.shutdown()
