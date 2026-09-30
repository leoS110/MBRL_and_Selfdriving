#basing off of: https://arxiv.org/abs/1708.02596

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field 
import gymnasium as gym 
from tqdm import tqdm #for terminal progress bar
import time
from stable_baselines3.common.logger import configure #for logging
import os #for model saving
import shutil

#import definitions

from config_train import TrainConfig, TransitionConfig
import config_train

from transitionbatch import TransitionBatch 
from replay_buffer import ReplayBuffer 
from transition_model import pytorchNN
#from MPC import CEM_loop
from MPC_vectorised import CEM_loop

train_params = TrainConfig()

#setup env:
env = gym.make(
    'HalfCheetah-v5',
    forward_reward_weight=train_params.forward_reward_weight,
    ctrl_cost_weight=train_params.ctrl_cost_weight,
    reset_noise_scale=train_params.reset_noise_scale,
    exclude_current_positions_from_observation=train_params.exclude_current_positions_from_observation,
    frame_skip=train_params.frame_skip,
    #render_mode = train_params.render_mode,
    #frametime=train_params.frametime
)

video_env = gym.wrappers.RecordVideo(
    gym.make('HalfCheetah-v5', forward_reward_weight=train_params.forward_reward_weight, ctrl_cost_weight=train_params.ctrl_cost_weight, reset_noise_scale=train_params.reset_noise_scale, exclude_current_positions_from_observation=train_params.exclude_current_positions_from_observation, frame_skip=train_params.frame_skip, render_mode="rgb_array"),
    video_folder="mbrl_runs/cheetah/videos",
    episode_trigger=lambda ep: True,        # each window starts with a reset, so each window = one clip
    video_length=train_params.render_length,
    name_prefix="mpc",
)

observation, info = env.reset()


#initialise D_RL and transitions models

D_RAND = ReplayBuffer(
    capacity= train_params.D_RAND_capacity,     
    obs_shape= env.observation_space.shape, #returns tuple with dimensions
    action_shape= env.action_space.shape,
)

D_RL = ReplayBuffer(
    capacity= train_params.D_RL_capacity,   
    obs_shape=env.observation_space.shape,
    action_shape=env.action_space.shape,
)

D_combined = ReplayBuffer( #for means and standard devitations to normalise state differences
    capacity= train_params.D_RL_capacity + train_params.D_RAND_capacity,   
    obs_shape=env.observation_space.shape,
    action_shape=env.action_space.shape,
)


#function for generating a mixed training set in correct proportion across D_rand and D_rl
def sample_mixed(D_RAND, D_RL, batch_size, Drl_proportion) -> TransitionBatch:

    n_rl = min(round(batch_size * Drl_proportion), D_RL.num_stored) #allows for D_RL not having enough transitions yet for n_rl
    n_rand = batch_size - n_rl

    transitions_rand = D_RAND.sample(n_rand) #sample from D_rand, save as a transition batch (a tuple)
    transitions_rl = D_RL.sample(n_rl) #sample from D_rl, save as a transition batch (a tuple)(sampling 0 elements should hopefully not crash)

    #return a new transition batch
    return TransitionBatch(
        obs=np.concatenate([transitions_rand.obs, transitions_rl.obs]),
        act=np.concatenate([transitions_rand.act, transitions_rl.act]),
        next_obs=np.concatenate([transitions_rand.next_obs, transitions_rl.next_obs]),
        rewards=np.concatenate([transitions_rand.rewards, transitions_rl.rewards]),
        terminateds=np.concatenate([transitions_rand.terminateds, transitions_rl.terminateds]),
        truncateds=np.concatenate([transitions_rand.truncateds, transitions_rl.truncateds]),
    )

    
#ensemble models in a list, each with its own optimiser 
model_list = []
optimiser_list = []
nn_params = TransitionConfig()
#pytorch should generate independant random seeds automatically
#all currently running on CPU for now, can transfer

for i in range(train_params.ensemble_size):

    transition_model = pytorchNN(nn_params.dimension_in, nn_params.dimension_out, nn_params.n_width, nn_params.n_layers)
    #optimiser = torch.optim.SGD(transition_model.parameters(), lr=nn_params.lr)
    optimiser = optim.Adam(transition_model.parameters(), lr=nn_params.lr)

    model_list.append(transition_model)
    optimiser_list.append(optimiser)

loss_fn = nn.MSELoss()   

#setup log:
logger = configure("mbrl_runs/cheetah", ["stdout", "csv", "tensorboard"])
total_env_steps = 0

#Gather set of random trajectories and add to dedicated reply buffer (+ combined replay buffer):
rng = np.random.default_rng()
n_rand = train_params.rand_traj_n * train_params.rand_traj_length #total number of random env steps, for logging & progress bar
rand_rewards = [] #for logging

with tqdm(total=n_rand, desc="Random data", unit="step") as pbar:

    for rand_traj_i in range(train_params.rand_traj_n):
            
            for step_i in range(train_params.rand_traj_length):

                action = rng.uniform(train_params.a_min, train_params.a_max, size=(1, train_params.dimension_a))

                action_squeezed = action.squeeze()
                next_observation, reward, terminated, truncated, info = env.step(action_squeezed)

                D_RAND.add(obs=observation, action=action, next_obs=next_observation, reward=reward, terminated=terminated, truncated=truncated)
                D_combined.add(obs=observation, action=action, next_obs=next_observation, reward=reward, terminated=terminated, truncated=truncated)

                if terminated or truncated:
                    observation, info = env.reset()
                else:
                    observation = next_observation

                #for logging & pbar:
                rand_rewards.append(reward)
                pbar.update(1)

            observation, info = env.reset()

#for logging:
total_env_steps += n_rand
logger.record("baseline/random_step_reward", float(np.mean(rand_rewards)))   

#for saving the model ensemble, every aggregation loop if it better than the last
CKPT_DIR = f"mbrl_runs/cheetah/checkpoints/{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(CKPT_DIR, exist_ok=True)
shutil.copy(config_train.__file__, CKPT_DIR) #make a copy of the training copy for safekeeping
best_score = -float("inf")

observation, info = env.reset()

#aggregation and training loop:
aggregation_bar = tqdm(range(train_params.aggregation_iterations), desc="Aggregation", position=0, unit="iter") #for pbar

for loop_i in aggregation_bar: #equivalent to range(train_params.aggregation_iterations), just changed to allow for progress bar
    t_iter = time.perf_counter() #for pbar

    #get normalisation stats:
    statediff_means, statediff_stds, state_means, state_stds, act_means, act_stds = D_combined.get_statistics()

    #train transition model on buffer
    #pbar:
    train_bar = tqdm(range(train_params.SGD_steps), desc="  Train ensemble", position=1, leave=False, unit="step")
    for optimisation_i in train_bar: #range(train_params.SGD_steps)
        
        #can evaluate loss for loss curve here

        step_losses = [] #for logging, stores one loss value per model

        for model_i in range(train_params.ensemble_size):
            
            model = model_list[model_i]
            optimiser = optimiser_list[model_i]

            optimiser.zero_grad(set_to_none=True) #zero the gradients from the last step so that they don't accumulate
            
            #sampling a training data batch, different one for each model
            D_training = sample_mixed(D_RAND, D_RL, train_params.SGD_batch_size, train_params.Drl_proportion)
            #normalise training data with mean and standard deviation:

            #prepare data for pytorch neural network 
            obs, act, next_obs, rewards, terminateds, truncateds = D_training.astuple() #unpack tuple, each item now a pytorch tensor of individual 

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            act_tensor = torch.as_tensor(act, dtype=torch.float32)
            next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(-1) #make (batchsize, 1) rather than (batchsize,) in the case of scalar observations or actions
            if act_tensor.ndim == 1:
                act_tensor = act_tensor.unsqueeze(-1) 
            x_training = torch.cat([obs_tensor, act_tensor], dim=-1)

            #normalisation, both of x and y 
            #x:
            state_means_tensor = torch.as_tensor(state_means, dtype=torch.float32)
            state_stds_tensor = torch.as_tensor(state_stds, dtype=torch.float32)        
            act_means_tensor = torch.as_tensor(act_means, dtype=torch.float32)
            act_stds_tensor = torch.as_tensor(act_stds, dtype=torch.float32)  
            x_means_tensor = torch.cat([state_means_tensor, act_means_tensor], dim=-1)
            x_stds_tensor = torch.cat([state_stds_tensor, act_stds_tensor], dim=-1)
            x_training = (x_training - x_means_tensor)/x_stds_tensor


            #y:
            statediff_means_tensor = torch.as_tensor(statediff_means, dtype=torch.float32)
            statediff_stds_tensor = torch.as_tensor(statediff_stds, dtype=torch.float32)
            y_training = ((next_obs_tensor - obs_tensor) - statediff_means_tensor)/statediff_stds_tensor

            #could add gaussian noise to x_training & y_training here:

            #debugging:
            #print(f"x_training shape: {x_training.shape}")
            #print(f"y_training shape: {y_training.shape}")
            #print("x_training, x_means_tensor, x_stds_tensor sizes: ", x_training.size(), x_means_tensor.size(), x_stds_tensor.size())
            #print("y_training, (next_obs_tensor - obs_tensor), statediff_means_tensor, statediff_stds_tensor: ", y_training.size(), (next_obs_tensor - obs_tensor).size(), statediff_means_tensor.size(), statediff_stds_tensor.size())

            #h(x), forward pass, autograd caching
            model_vals = model(x_training)  
        
            #evaluate loss on batch
            loss = loss_fn(model_vals, y_training) 

            #for logging:
            step_losses.append(loss.item()) 
        
            #uses backprop + autograd to generate loss gradients (fills .grad graph)
            loss.backward()
            optimiser.step() 

        mean_loss = float(np.mean(step_losses)) #for logging, mean across models
        if optimisation_i == 0:
            logger.record("train/loss_first", mean_loss)
        if optimisation_i % 25 == 0:
            train_bar.set_postfix(loss=f"{mean_loss:.4f}")

    logger.record("train/loss_last", mean_loss) #at the end of training, mean loss across models, for comparison against loss_first

    #MPC rollout progress bar & logs
    MPC_rollout_bar = tqdm(range(train_params.rollout_steps_steps_per_aggregation // train_params.MPC_actions_per_A), desc="  MPC rollout", position=1, leave=False, unit="step")
    mean_step_reward = 0.0
    env_rollout_steps = 0.0

    active_env = env 
    
    for step_i in MPC_rollout_bar: #range(train_params.rollout_steps_per_aggregation), this is just total steps, not trajectories
        t0 = time.perf_counter()

        #MPC logic:

        #get current state: already in variable: observation

        #logic to render every train_params.render_period for a length of render_steps env steps
        phase = step_i % (train_params.render_period // train_params.MPC_actions_per_A) #might be a bug here
        if phase == 0: #window opens
            active_env = video_env
            observation, info = active_env.reset()
        elif phase == (train_params.render_period // train_params.MPC_actions_per_A): #window closes
            active_env = env
            observation, info = active_env.reset()

        #run MPC loop to get A(s)
        observation = observation.reshape(1, train_params.dimension_o) #make the right shape
        observation = torch.as_tensor(observation, dtype=torch.float32)
        initial_state_tensor = observation
        A = CEM_loop(model_list, initial_state_tensor, statediff_means_tensor, statediff_stds_tensor, x_means_tensor, x_stds_tensor)

        #execute first action in A

        for action_i in range(train_params.MPC_actions_per_A):

            action = A[action_i,:]
            action = action.numpy() #back to numpy for env
            next_observation, reward, terminated, truncated, info = active_env.step(action)

            #for logging: 
            mean_step_reward += reward
            env_rollout_steps += 1
            total_env_steps += 1

            #aggregate transition to D_RL (in numpy)
            D_RL.add(obs=observation, action=action, next_obs=next_observation, reward=reward, terminated=terminated, truncated=truncated)
            D_combined.add(obs=observation, action=action, next_obs=next_observation, reward=reward, terminated=terminated, truncated=truncated)

            #for next loop
            if terminated or truncated:
                observation, info = active_env.reset()
                break #go back to CEM loop, no point in carrying on actions
            else:
                observation = next_observation

    mean_step_reward = mean_step_reward / env_rollout_steps

    #check if ensemble performed better than the last and save if so
    if mean_step_reward > best_score:
        best_score = mean_step_reward
        torch.save({
            "state_dicts": [m.state_dict() for m in model_list],
            "x_means_tensor": x_means_tensor,  "x_stds_tensor": x_stds_tensor,
            "statediff_means_tensor": statediff_means_tensor, "statediff_stds_tensor": statediff_stds_tensor,
            "loop_i": loop_i, "mean_step_reward": float(mean_step_reward),
        }, f"{CKPT_DIR}/best.pt")
    
    #logging
    logger.record("time/iteration", loop_i)
    logger.record("time/total_env_steps", total_env_steps)
    logger.record("time/iter_s", time.perf_counter() - t_iter)
    logger.record("data/D_RAND", D_RAND.num_stored)
    logger.record("data/D_RL", D_RL.num_stored)
    logger.record("rollout/mean_step_reward", mean_step_reward)
    logger.record("rollout/best_step_reward", best_score)
    logger.dump(step=total_env_steps)
    

     
logger.close()









