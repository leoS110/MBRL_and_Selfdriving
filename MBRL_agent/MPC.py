#CEM
#(using state and observation interchangably, assuming fully observed / no representation learning)
from config import TrainConfig
import random

import replay_buffer
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from transition_model import get_state_dif 


train_params = TrainConfig()


def CEM_loop(model_list, initial_state_tensor, statediff_means_tensor, statediff_stds_tensor, x_means_tensor, x_stds_tensor):
    #print("in CEM loop")
    #initialise A distribution p(A): (https://docs.pytorch.org/docs/2.14/distributions.html#multivariatenormal)
    A_distribution = MultivariateNormal(torch.zeros(train_params.MPC_horizon*train_params.dimension_a), torch.eye(train_params.MPC_horizon*train_params.dimension_a))
    #whole action sequence in one flat vector

    #trajectory_store = np.empty(train_params.MPC_horizon*train_params.dimension_a, train_params.CEM_trajn)

   
    

    for CEM_loop_i in range(train_params.CEM_loopn):

        reward_store = np.empty((1, train_params.CEM_trajn))

        #resample CEM_trajn trajectories from p(A):
        samples = A_distribution.sample((train_params.CEM_trajn,)) #tensor: (number of trajectories, trajectory vector length)
        #print("A sample, size: ", samples.size())
        #reshape samples and clamp actions to bounds
        samples_arrangedandbounded = samples.reshape(train_params.CEM_trajn, train_params.MPC_horizon, train_params.dimension_a).clamp(train_params.a_min, train_params.a_max)  #should probably check reshaping syntax 
        #print("A sample reshaped, size: ", samples_arrangedandbounded.size())                 

        #Evaluate reward associated to each trajectory J(A), loop through samples
        for sampled_traj_i in range(train_params.CEM_trajn):

            A_i = samples_arrangedandbounded[sampled_traj_i, :, :]
            #print("individual A_i for rollout, size: ", A_i.size())
            
            reward_val = expected_MPC_reward(model_list, initial_state_tensor, A_i, statediff_means_tensor, statediff_stds_tensor, x_means_tensor, x_stds_tensor)

            reward_store[0, sampled_traj_i] = reward_val

        #pick elites 
        top_n_indices = np.argsort(-reward_store[0,:])[:train_params.CEM_elitespicked] #negates, sorts, and slices the first terms
        #print("top n indices of best A_i, size: ", top_n_indices.size, top_n_indices)
        samples_elites = samples_arrangedandbounded[top_n_indices, :, :] #(elite_n, H, n_a)
        #print("elite samples picked, size: ", samples_elites.size())

        #refit p(A) to elites: define a new multivariate normal based off of elite action paths, tensors
        samples_elites = samples_elites.reshape(train_params.CEM_elitespicked, train_params.MPC_horizon*train_params.dimension_a) #reshape back into (number of elites, vector length of multivariate sampling)
        #print("samples elites reshaped, size: ", samples_elites.size())
        new_mean_tensor = samples_elites.mean(dim = 0)
        #print("new_mean_tensor, size: ", new_mean_tensor.size())
        #new_cov_tensor = torch.cov(samples_elites.T)  #pytorch estimates full covariance matrix, transpose is just to align with what pytorch function expects
        new_var_tensor = samples_elites.var(dim=0).clamp(min=train_params.CEM_min_std) #(H*da,) vector of standard deviations. Clamping means variance can't get too low, ensuring exploration
        #print("new_var_tensor, size: ", new_var_tensor.size())
        new_cov_tensor = torch.diag(new_var_tensor) #make the full covariance matrix, only filling diagonal

        A_distribution = MultivariateNormal(new_mean_tensor, covariance_matrix=new_cov_tensor)


    #return best A as a numpy array
    top_1_index = np.argsort(-reward_store[0,:])[:1]
    sample_elite = samples_arrangedandbounded[top_1_index, :, :]
    #print("samples_elite at the end, size: ", sample_elite.size())
    A_best = sample_elite.reshape(train_params.MPC_horizon, train_params.dimension_a)
    #print("A_best, size: ", A_best.size())

    return A_best #(H, n_a), tensor


def expected_MPC_reward(model_list, initial_state, A, statediff_means_tensor, statediff_stds_tensor, x_means_tensor, x_stds_tensor): 
    
    #curious about potential to alter reward to penalise model disagreement 

    #for model in model_list: #sample based expectation under model parameter uncertainty, or just pick a random one out of model list?
    model = random.choice(model_list)

    reward_val = 0.0
    state = initial_state

    for step_i in range(train_params.MPC_horizon):
        #take open loop action, change of state stepped through learnt model
        action = A[step_i,:]
        action = action.unsqueeze(0) #make action (1, n_a) size
        with torch.no_grad():
            delta_state = get_state_dif(model, state, action, statediff_means_tensor, statediff_stds_tensor, x_means_tensor, x_stds_tensor) #assuming de-normalised by this point
            new_state = state + delta_state

        #calculate reward 
        reward_val += step_reward_eval(new_state, action) #new state and action that caused it is r(s', a): intuitively matches an action being good to cause more velocity otherwise action is unrelated. 
        state = new_state
    
    reward_val *= 1/train_params.ensemble_size

    return reward_val

def step_reward_eval(state, action): #pytorch tensors
    #environment dependant: define once demonstration env is decided

    #small modification from standard half-cheetah: can't do lookahead to find discrete delta_x into the future (or at least don't want to use learnt model to do that)
    #instead using the instantaneous value and hoping MPC does enough lookahead

    #assuming state (1, 17)
    dx_dt_tip = state[0,8]
    squared_l2_action = action.pow(2).sum(dim=-1)

    reward_val = train_params.forward_reward_weight * dx_dt_tip - train_params.ctrl_cost_weight * squared_l2_action

    return reward_val
