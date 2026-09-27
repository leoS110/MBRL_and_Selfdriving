#basing off of: https://arxiv.org/abs/1708.02596

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field 
import gymnasium as gym 
from tqdm import tqdm #for terminal progress bar
import time
from stable_baselines3.common.logger import configure #for logging

#import definitions

from config import TrainConfig, TransitionConfig

from transitionbatch import TransitionBatch 
from replay_buffer import ReplayBuffer 
from transition_model import pytorchNN
from MPC import CEM_loop

train_params = TrainConfig()

#setup env:
env = gym.make(
    'HalfCheetah-v5',
    forward_reward_weight=train_params.forward_reward_weight,
    ctrl_cost_weight=train_params.ctrl_cost_weight,
    reset_noise_scale=train_params.reset_noise_scale,
    exclude_current_positions_from_observation=train_params.exclude_current_positions_from_observation,
    frame_skip=train_params.frame_skip,
    render_mode = "human",
    #frametime=train_params.frametime
)
observation, info = env.reset()


    
#ensemble models in a list, each with its own optimiser, load: 

nn_params = TransitionConfig()
#pytorch should generate independant random seeds automatically


#load combined buffer for normalisation statistics:


for i in range(train_params.ensemble_size):

    transition_model = pytorchNN(nn_params.dimension_in, nn_params.dimension_out, nn_params.n_width, nn_params.n_layers)
    optimiser = torch.optim.SGD(transition_model.parameters(), lr=nn_params.lr) 

    model_list.append(transition_model)
    optimiser_list.append(optimiser)

loss_fn = nn.MSELoss()   



observation, info = env.reset()

#aggregation and training loop:
aggregation_bar = tqdm(range(train_params.aggregation_iterations), desc="Aggregation", position=0, unit="iter") #for pbar



#get normalisation stats:
statediff_means, statediff_stds, state_means, state_stds, act_means, act_stds = D_combined.get_statistics()

#MPC rollout progress bar
MPC_rollout_bar = tqdm(range(train_params.rollouts_per_aggregation), desc="  MPC rollout", position=1, leave=False, unit="step")

for rollout_i in MPC_rollout_bar: #range(train_params.rollouts_per_aggregation)
    t0 = time.perf_counter()

    #MPC logic:

    #get current state: already in variable: observation

    #run MPC loop to get A(s)
    observation = observation.reshape(1, train_params.dimension_o) #make the right shape
    observation = torch.as_tensor(observation, dtype=torch.float32)
    initial_state_tensor = observation
    A = CEM_loop(model_list, initial_state_tensor, statediff_means_tensor, statediff_stds_tensor, x_means_tensor, x_stds_tensor)

    #execute first action
    action = A[0,:]
    action = action.numpy() #back to numpy for env
    next_observation, reward, terminated, truncated, info = env.step(action)

    #aggregate transition to D_RL (in numpy)
    D_combined.add(obs=observation, action=action, next_obs=next_observation, reward=reward, terminated=terminated, truncated=truncated)

    #for next loop
    if terminated or truncated:
        observation, info = env.reset()
    else:
        observation = next_observation









