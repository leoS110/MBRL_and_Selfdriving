#CEM
#(using state and observation interchangably, assuming fully observed / no representation learning)
from config import TrainConfig

import replay_buffer
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from transition_model import get_state_dif 


train_params = TrainConfig()


def CEM_loop(model_list, initial_state, statediff_means_tensor, statediff_stds_tensor, x_means_tensor, x_stds_tensor):

    #initialise A distribution p(A): (https://docs.pytorch.org/docs/2.14/distributions.html#multivariatenormal)
    A_distribution = MultivariateNormal(torch.zeros(train_params.MPC_horizon*train_params.dimension_a), torch.eye(train_params.MPC_horizon*train_params.dimension_a))
    #whole action sequence in one flat vector

    initial_state_tensor = torch.as_tensor(initial_state, dtype=torch.float32)

    #trajectory_store = np.empty(train_params.MPC_horizon*train_params.dimension_a, train_params.CEM_trajn)
    

    for CEM_loop_i in range(train_params.CEM_loopn):

        reward_store = np.empty(1, train_params.CEM_trajn)

        for sample_traj_i in range(train_params.CEM_trajn):

            #resample CEM_trajn trajectories from p(A):
            samples = A_distribution.sample((train_params.CEM_trajn,)) #tensor: (number of trajectories, trajectory vector length)
            #reshape samples and clamp actions to bounds
            samples_arrangedandbounded = samples.reshape(train_params.CEM_trajn, train_params.MPC_horizon, train_params.dimension_a).clamp(train_params.a_min, train_params.a_max)  #should probably check reshaping syntax                  

            #Evaluate reward associated to each trajectory J(A), loop through samples
            for sampled_traj_i in range(train_params.CEM_trajn):

                A_i = samples_arrangedandbounded[sampled_traj_i, :, :]

                reward_val = expected_MPC_reward(model_list, initial_state_tensor, A_i, statediff_means_tensor, statediff_stds_tensor, x_means_tensor, x_stds_tensor)

                reward_store[sampled_traj_i] = reward_val

        #pick elites 
        top_n_indices = np.argsort(-reward_store)[:train_params.CEM_elitespicked] #negates, sorts, and slices the first terms
        samples_elites = samples_arrangedandbounded[top_n_indices, :, :] #not sure about indexing here

        #refit p(A) to elites: define a new multivariate normal based off of elite action paths, tensors
        samples_elites = samples_elites.reshape(train_params.CEM_elitespicked, train_params.MPC_horizon*train_params.dimension_a) #reshape back into (number of elites, vector length of multivariate sampling)
        new_mean_tensor = samples_elites.mean(dim = 0)
        new_cov_tensor = torch.cov(samples_elites.T)  #pytorch estimates full covariance matrix, transpose is just to align with what pytorch function expects
        #claude recommends doing a jitter here, check:
        new_cov_tensor = new_cov_tensor + 1e-4 * torch.eye(train_params.CEM_elitespicked)  

        A_distribution = MultivariateNormal(new_mean_tensor, covariance_matrix=new_cov_tensor)


    #return best A as a numpy array
    top_1_index = np.argsort(-reward_store)[:1]
    A_best = samples_arrangedandbounded[top_1_index, :, :] #(horizon, dimension_a)

    return A_best


def expected_MPC_reward(model_list, initial_state, A, statediff_means_tensor, statediff_stds_tensor, x_means_tensor, x_stds_tensor): 
    
    #curious about potential to alter reward to penalise model disagreement 

    for model in model_list: #sample based expectation under model parameter uncertainty

        reward_val = 0.0
        state = initial_state

        for step_i in range(train_params.MPC_horizon):
            #take open loop action, change of state stepped through learnt model
            action = A[step_i,:]
            delta_state = get_state_dif(model, state, action, statediff_means_tensor, statediff_stds_tensor, x_means_tensor, x_stds_tensor) #assuming de-normalised by this point
            new_state = state + delta_state

            #calculate reward 
            reward_val += step_reward_eval(new_state)
            state = new_state
    
    reward_val *= 1/train_params.ensemble_size

    return reward_val

def step_reward_eval(state):
    #environment dependant: define once demonstration env is decided

    return reward_val
