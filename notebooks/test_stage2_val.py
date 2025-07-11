import os
import ctypes
import matplotlib.pyplot as plt
import numpy as np
import torch
import pprint
from easydict import EasyDict

from genrobo3d.rlbench.environments_joint import RLBenchEnv, Mover
from genrobo3d.train.utils.misc import set_random_seed
from genrobo3d.evaluation.eval_stage2 import Actioner
from rlbench.backend.utils import task_file_to_task_class

from genrobo3d.models.stage2_ptv3 import SimplePolicyPTV3CA
from genrobo3d.models.stage2_dit import DiTraj
from genrobo3d.configs.default import get_config

# PTV3 model is required to generate tokens
ptv3_ckpt_path = 'data/experiments/gembench/stage2_ptv3/ckpts/model_step_5000.pt'
dit_ckpt_path = 'data/experiments/gembench/stage2_dit/ckpts/model_step_5000.pt'
# Load the PTV3 config
ptv3_cfg = get_config("data/experiments/gembench/stage2_ptv3/logs/training_config.yaml", [])
dit_cfg = get_config("data/experiments/gembench/stage2_dit/logs/training_config.yaml", [])


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
    'exp_config': 'data/experiments/gembench/stage2_dit/logs/training_config.yaml',
    'checkpoint': 'data/experiments/gembench/stage2_dit/ckpts/model_step_5000.pt',
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
        self.policy = policy_model.eval().to(device)
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
        output = self.policy(batch, pc_tokens, final_pred_joints_actions, max_steps=30, batch_size=1, robot_type=8, compute_loss=False)
        return output.cpu().numpy()
    
# Seed for reproducibility
set_random_seed(90)

taskvar = 'close_jar_peract+15'
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

output = np.array([
    [4.2951123759849e-06, 0.17528465390205383, 5.993514605506789e-06, -0.8731035590171814, 8.395782060688362e-07, 1.2215666770935059, 0.7853997945785522, 1],
    [0.23206400871276855, 0.6638941764831543, 0.10354617238044739, -1.5426342487335205, -0.07551609724760056, 2.132195234298706, 1.3959020376205444, 1],
    [0.2465958595275879, 0.9527158737182617, 0.08250811696052551, -1.5060172080993652, -0.09790884703397751, 2.384737491607666, 1.407901406288147, 1],
    [0.2317514419555664, 0.6674516201019287, 0.10340431332588196, -1.5462713241577148, -0.07565533369779587, 2.137094497680664, 1.3960152864456177, 1],
    [-0.08753848075866699, 0.016955137252807617, -0.024588316679000854, -2.5314533710479736, 0.0008658245205879211, 2.477821111679077, 0.9242910146713257, 1],
    [-0.09073710441589355, 0.1565866470336914, -0.025102823972702026, -2.552497386932373, 0.008303053677082062, 2.638749361038208, 0.9139708280563354, 1],
    [0.07953476905822754, 0.16550564765930176, -0.1932961642742157, -2.5462305545806885, 0.06534851342439651, 2.6380748748779297, 2.435086250305176, 1]
])

output = actioner.predict(**batch)

plt.imshow(rgb_combined.astype(np.uint8))
plt.axis('off')
plt.show()

for step_id in range(30):

    action = output[step_id]

    print(f"Step {step_id} - Pred Joint:", action[:7])

    if action is None:
        print("No action predicted, breaking.")
        break

    try:
        obs, reward, terminate, _ = mover(action, verbose=False)
        print(f'Step id: {step_id + 1}')
        joint_positions = task._robot.arm.get_joint_positions()
        print(f"Step {step_id} - Real Joint:", joint_positions)
        rgb_combined = np.concatenate([
            obs.left_shoulder_rgb,
            obs.right_shoulder_rgb,
            obs.wrist_rgb,
            obs.front_rgb
        ], axis=1)
        plt.imshow(rgb_combined.astype(np.uint8))
        plt.axis('off')
        plt.show()

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
