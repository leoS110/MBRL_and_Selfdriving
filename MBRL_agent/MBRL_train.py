#basing off of: https://arxiv.org/abs/1708.02596

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field 
import gymnasium as gym 

#import key class definitions
from transitionbatch import TransitionBatch 
from replay_buffer import ReplayBuffer 
from transition_model import TransitionConfig
from transition_model import pytorchNN

#NN paramaters
@dataclass                                   
class TrainConfig:
    #environment specific paramaters: 
    dimension_o: int = 1                                
    dimension_a: int = 2 

    aggregation_iterations: int = 5

    #algorithm parameters:
    SGG_batch_size: int = 512 
    Drand_proportion: float = 0.1
    Drl_proportion: float = 0.9
    MPC_horizon: int = 10 
    CEM_trajn: int = 1000
    SGD_steps: int = 60
    rollouts_per_aggregation: int = 400

    ensemble_size: int = 5

    D_RAND_capacity: int = 1e6  #(just set to be large enough to never replace)
    D_RL_capacity: int = 1e6

    #random trajectory parameters: (fully eyeballed values)
    rand_traj_length: int = 200
    rand_traj_n: int = 10
     
train_params = TrainConfig()

#setup env:
env = gym.make("CartPole-v1", render_mode="human")
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


#Gather set of random trajectories and add to dedicated reply buffer:




#aggregation and training loop
for loop_i in range(train_params.aggregation_iterations):

    #train transition model on buffer
    for optimisation_i in range(train_params.SGD_steps):
        
        #can evaluate loss for loss curve here
    
        #for SGD, generate a batch from training data:
        D_RAND_tuple = D_RAND.sample(np.floor(train_params.batch_size*train_params.Drand_proportion))
        D_RL_tuple = D_RL.sample(np.floor(train_params.batch_size*train_params.Drl_proportion))
        #combine tuples:
       

        #need to add functionality that if not enough D_RL points exist yet for required sampling then just use more D_RAND

    
        for model_i in range(train_params.ensemble_size):

            #update to this application

            #optimizer.zero_grad(set_to_none=True) #zero the gradients from the last step so that they don't accumulate

            #h(x), forward pass, autograd caching
            #h_val = model(x_TR_batch)  
        
            #evaluate loss on batch
            #loss = loss_fn(h_val, y_TR_batch) 
        
            #uses backprop + autograd to generate loss gradients (fills .grad graph)
            #loss.backward()
            #optimizer.step() 
    
    for rollout_i in range(train_params.rollouts_per_aggregation):

        #MPC logic:

        #get current state: already in variable: observation

        #run MPC loop to get A(s)
        A = CEM_loop(observation)

        #execute first action
        action = A[1]
        next_observation, reward, terminated, truncated, info = env.step(action)

        #aggregate transition to D_RL: use Transition batch setup?
        D_RL.add(obs=observation, action=action, next_obs=next_observation, reward=reward, terminated=terminated, truncated=truncated)

        #for next loop
        observation = next_observation



#save replay buffers:



#save models:






