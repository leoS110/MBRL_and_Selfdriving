#basing off of: https://arxiv.org/abs/1708.02596

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field 
import gymnasium as gym 

#import definitions

from config import TrainConfig, TransitionConfig

from transitionbatch import TransitionBatch 
from replay_buffer import ReplayBuffer 
from transition_model import pytorchNN
from MPC import CEM_loop

train_params = TrainConfig()

#setup env:
env = gym.make('HalfCheetah-v5') #cpst weight defaults in 
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
    optimiser = torch.optim.SGD(transition_model.parameters(), lr=nn_params.lr) 

    model_list.append(transition_model)
    optimiser_list.append(optimiser)

loss_fn = nn.MSELoss()   

#Gather set of random trajectories and add to dedicated reply buffer (+ combined replay buffer):
rng = np.random.default_rng()

for rand_traj_i in range(train_params.rand_traj_n):
        
        for step_i in range(train_params.rand_traj_length):

            action = rng.uniform(train_params.a_min, train_params.a_max, size=(1, train_params.dimension_a))

            next_observation, reward, terminated, truncated, info = env.step(action)

            D_RAND.add(obs=observation, action=action, next_obs=next_observation, reward=reward, terminated=terminated, truncated=truncated)
            D_combined.add(obs=observation, action=action, next_obs=next_observation, reward=reward, terminated=terminated, truncated=truncated)

            if terminated or truncated:
                observation, info = env.reset()
            else:
                observation = next_observation

        observation, info = env.reset()
    


observation, info = env.reset()

#aggregation and training loop
for loop_i in range(train_params.aggregation_iterations):

    #get normalisation stats:
    statediff_means, statediff_stds, state_means, state_stds, act_means, act_stds = D_combined.get_statistics()

    #train transition model on buffer
    for optimisation_i in range(train_params.SGD_steps):
        
        #can evaluate loss for loss curve here
    
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
            y_training = ((next_obs_tensor - obs_tensor) - statediff_means)/statediff_stds

            #could add gaussian noise to x_training & y_training here:


            #h(x), forward pass, autograd caching
            model_vals = model(x_training)  #CHECK X FORMAT HERE
        
            #evaluate loss on batch
            loss = loss_fn(model_vals, y_training) 
        
            #uses backprop + autograd to generate loss gradients (fills .grad graph)
            loss.backward()
            optimiser.step() 
    
    for rollout_i in range(train_params.rollouts_per_aggregation):

        #MPC logic:

        #get current state: already in variable: observation

        #run MPC loop to get A(s)
        initial_state = observation
        A = CEM_loop(model_list, initial_state, statediff_means_tensor, statediff_stds_tensor, x_means_tensor, x_stds_tensor)

        #execute first action
        action = A[0,:]
        next_observation, reward, terminated, truncated, info = env.step(action)

        #aggregate transition to D_RL: use Transition batch setup?
        D_RL.add(obs=observation, action=action, next_obs=next_observation, reward=reward, terminated=terminated, truncated=truncated)
        D_combined.add(obs=observation, action=action, next_obs=next_observation, reward=reward, terminated=terminated, truncated=truncated)

        #for next loop
        if terminated or truncated:
            observation, info = env.reset()
        else:
            observation = next_observation


        



#save replay buffers:



#save models:






