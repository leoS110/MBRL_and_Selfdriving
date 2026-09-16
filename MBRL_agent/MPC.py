#CEM
#(using state and observation interchangably, assuming fully observed / no representation learning)

import replay_buffer
import numpy as np
from MBRL_train import train_params
from transition_model import get_state_dif 


def CEM_loop():

    return A







def expected_MPC_reward(model_list, initial_state, A, state_diff_mean, state_diff_standarddev): 
    
    reward_val = 0.0
    state = initial_state

    for model in model_list: #sample based expectation under model parameter uncertainty

        for step_i in range(train_params.MPC_horizon):
            #take open loop action, change of state stepped through learnt model
            action = A[:,step_i]
            delta_state = get_state_dif(model, state, action, state_diff_mean, state_diff_standarddev) #assuming de-normalised by this point
            new_state = state + delta_state

            #calculate reward 
            reward_val += step_reward_eval(new_state)
            state = new_state
    
    reward_val *= 1/train_params.ensemble_size

    return reward_val

def step_reward_eval(state):
    #environment dependant: define once demonstration env is decided

    return reward_val
