
from dataclasses import dataclass, field 
import torch


@dataclass                                   
class TrainConfig:
    #environment specific paramaters: (env specific: half-cheetah)
    dimension_o: int = 17  #17 default, 18 if x is included in state vector, see below                    
    dimension_a: int = 6
    a_min: float = -1.0
    a_max: float = 1.0
    forward_reward_weight: float = 1.0 #default 1
    ctrl_cost_weight: float = 0.1 #default 0.1
    reset_noise_scale: float = 0.1 #default 0.1
    exclude_current_positions_from_observation: bool = True #default true
    frame_skip: int = 5 #default 5
    frametime: float = 0.01 #default 0.01
    render_mode: str = "None"

    tip_cost_weight: float = 0.1 #manually addition to penalise flip in predictions


    #algorithm parameters:
    aggregation_iterations: int = 8 #7 in paper
    SGD_batch_size: int = 512 #512 in paper, i think 
    Drand_proportion: float = 0.5 #0.1 in paper
    Drl_proportion: float = 0.5 #0.9 in paper
    
    MPC_horizon: int = 10 #10 in paper: 0.2s with frame_skip = 5, *0.05 to get corresponding in s
    CEM_trajn: int = 200 #1000 in paper, but this is for random shooting
    SGD_steps: int = 4000 #not sure,plateaus at 800 on random steps, but later 800 doesn't impact much. Careful not to pull repeated points, check this
    #rollout_steps_steps_per_aggregation: int = 9000 #9 full loops of 1000 steps before truncation: 9000 in paper
    MPC_actions_per_A: int = 1

    rollout_episodes: int = 60
    rollout_stepsperepisode: int = 150

    ensemble_size: int = 8 #no idea

    #really not sure, check: 
    CEM_loopn: int = 5 #5 is standard 
    CEM_elitespicked: int = 10 #not sure
    MPC_variance_init: float = 0.5 #notsure
    CEM_min_variance: float = 0.01 #mot sure

    D_RAND_capacity: int = 5_000_000  #(just set to be large enough to never replace)
    D_RL_capacity: int = 2_000_000

    #random trajectory parameters: (fully eyeballed values)
    rand_traj_length: int = 200
    rand_traj_n: int = 500 #10 in paper

    #for in window rendering
    render_period: float = 300 #300 env steps, dt * frame_skip * render_period = 0.01 * 5 * 300?
    render_length: float = 100
    #for video saving mid training:



#NN paramaters
@dataclass                                   
class TransitionConfig:

    #input and output data: (env specific)
    dimension_in: int = 23                           
    dimension_out: int = 17

    #NN parameters:   #check paper what they used, 4x200 in PETS? 150x3 worked well also                          
    n_width: int = 300                       
    n_layers: int = 4 #hidden layers, not including output

    #optimisation parameters:
    lr: float = 1e-3                                            
                
    #devide allocation:
    device: torch.device = torch.device("cuda") #"cuda" or "cpu"
