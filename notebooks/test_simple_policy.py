# test_env.py
import os
import ctypes
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict
from genrobo3d.rlbench.environments import RLBenchEnv, Mover
from genrobo3d.train.utils.misc import set_random_seed
from genrobo3d.evaluation.eval_simple_policy import Actioner
from rlbench.backend.utils import task_file_to_task_class
from time import sleep

os.chdir('..') # locate in the robot-3dlotus directory

# Preload CoppeliaSim lib
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/uhcc/Desktop/robot-3dlotus/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
os.environ['LD_LIBRARY_PATH'] = "/home/uhcc/Desktop/robot-3dlotus/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04:" + os.environ.get('LD_LIBRARY_PATH', '')
ctypes.CDLL("/home/uhcc/Desktop/robot-3dlotus/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/libcoppeliaSim.so.1")


model_args = EasyDict(
    exp_config='/home/uhcc/Desktop/robot-3dlotus/data/experiments/gembench/3dlotus/v2/logs/training_config.yaml',
    checkpoint='/home/uhcc/Desktop/robot-3dlotus/data/experiments/gembench/3dlotus/v2/ckpts/model_step_40000.pt',
    device='cuda',
    real_robot=False,
    save_obs_outs_dir=None,
    best_disc_pos='max',
    num_ensembles=1,
    remained_args={},
)

seed = 10
set_random_seed(seed)

image_size = [256, 256]
mover_max_tries = 10
max_steps = 25

actioner = Actioner(model_args)
# taskvar = 'open_door+0'
# taskvar = 'close_microwave+0'
# taskvar = 'push_button+0'
taskvar = 'close_jar_peract+16'
# taskvar = 'pick_up_cup+9'

# taskvar = 'close_laptop_lid+0'
# taskvar = 'close_laptop_lid+0'
# taskvar = 'close_fridge+0'
task_str, variation_id = taskvar.split('+')
variation_id = int(variation_id)

image_size = [256, 256]
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

move = Mover(task, max_tries=10)

demo_id = 0

instructions, obs = task.reset()

print('Instructions:', instructions)

obs_state_dict = env.get_observation(obs)
move.reset(obs_state_dict['gripper'])


print('Initial observation')
plt.imshow(np.concatenate([obs.left_shoulder_rgb, obs.right_shoulder_rgb, obs.wrist_rgb, obs.front_rgb], 1))
plt.show()


for step_id in range(max_steps):
    # fetch the current observation, and predict one action
    batch = {
        'task_str': task_str,
        'variation': variation_id,
        'step_id': step_id,
        'obs_state_dict': obs_state_dict,
        'episode_id': demo_id,
        'instructions': instructions,
    }

    output = actioner.predict(**batch)
    action = output["action"]
    print (action)

    if action is None:
        break

    # update the observation based on the predicted action
    try:
        obs, reward, terminate, _ = move(action, verbose=False)
        print('Step id:', step_id+1)
        plt.imshow(np.concatenate([obs.left_shoulder_rgb, obs.right_shoulder_rgb, obs.wrist_rgb, obs.front_rgb], 1))
        plt.show()
        
        obs_state_dict = env.get_observation(obs)  # type: ignore

        if reward == 1:
            break
        if terminate:
            print("The episode has terminated!")
    except (IKError, ConfigurationPathError, InvalidActionError) as e:
        print(taskvar, demo_id, step_id, e)
        reward = 0
        break

print('Reward:', reward)

env.env.shutdown()