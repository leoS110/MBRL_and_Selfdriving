#basing off of: https://arxiv.org/abs/1708.02596

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field 
import gymnasium as gym 
from tqdm import tqdm #for terminal progress bar
import time
from stable_baselines3.common.logger import configure #for logging
from pathlib import Path

#import definitions

from config_videocollection import TrainConfig, TransitionConfig

from transitionbatch import TransitionBatch 
from replay_buffer import ReplayBuffer 
from transition_model import pytorchNN
from MPC_run import CEM_loop

from velocity_plot_wrapper import VelocityPlotWrapper  

train_params = TrainConfig()


#load transition model ensemble: 
nn_params = TransitionConfig()
AGENT_DIR = Path(__file__).resolve().parent 
saved_checkpoint_path = AGENT_DIR / "models" / "3009ensemble.pt"
saved_checkpoint = torch.load(saved_checkpoint_path, weights_only=True)

#load normalisation stats
x_means_tensor = saved_checkpoint["x_means_tensor"]
x_stds_tensor = saved_checkpoint["x_stds_tensor"]
statediff_means_tensor = saved_checkpoint["statediff_means_tensor"]
statediff_stds_tensor = saved_checkpoint["statediff_stds_tensor"]

#load model list
model_list = []
for sd in saved_checkpoint["state_dicts"]:
    m = pytorchNN(nn_params.dimension_in, nn_params.dimension_out, nn_params.n_width, nn_params.n_layers)
    m.load_state_dict(sd)
    m.eval()
    model_list.append(m)


#setup env:
cheetah_xml = str(AGENT_DIR / "custom_env" / "halfcheetah_custom.xml") #edited to have nicer looking floor
video_env = gym.wrappers.RecordVideo(
    VelocityPlotWrapper(
        gym.make('HalfCheetah-v5', xml_file=cheetah_xml, forward_reward_weight=train_params.forward_reward_weight, ctrl_cost_weight=train_params.ctrl_cost_weight, reset_noise_scale=train_params.reset_noise_scale, exclude_current_positions_from_observation=train_params.exclude_current_positions_from_observation, frame_skip=train_params.frame_skip, render_mode="rgb_array"),
        target_velocity = 3.5 #must pass manually into MPC_run too + change file name
    ),
    video_folder="mbrl_videocollection/cheetah/tracking_vel",
    episode_trigger=lambda ep: True,        # each window starts with a reset, so each window = one clip
    video_length=train_params.rollout_steps_steps_per_video,
    name_prefix="mpc",
)


observation, info = video_env.reset()






#MPC rollout progress bar
video_rollout_bar = tqdm(range(train_params.video_n), desc="  video rollout", position=1, leave=False, unit="step")

for rollout_i in video_rollout_bar: #range(train_params.rollouts_per_aggregation)
    t0 = time.perf_counter()
    steps_rollout_bar = tqdm(range(train_params.rollout_steps_steps_per_video), desc="  steps per video rollout", position=2, leave=False, unit="step")


    for step_in in steps_rollout_bar:

        #MPC logic:

        #get current state: already in variable: observation

        #run MPC loop to get A(s)
        observation = observation.reshape(1, train_params.dimension_o) #make the right shape
        observation = torch.as_tensor(observation, dtype=torch.float32)
        #print("tip x velocity: ", observation[0, 8])
        initial_state_tensor = observation
        A = CEM_loop(model_list, initial_state_tensor, statediff_means_tensor, statediff_stds_tensor, x_means_tensor, x_stds_tensor)

        #execute first action
        action = A[0,:]
        action = action.numpy() #back to numpy for env
        next_observation, reward, terminated, truncated, info = video_env.step(action)

        #for next loop
        if terminated or truncated:
            observation, info = video_env.reset()
        else:
            observation = next_observation

    observation, info = video_env.reset()










