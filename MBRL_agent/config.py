
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


    #algorithm parameters:
    aggregation_iterations: int = 20 #7 in paper
    SGD_batch_size: int = 512 #512 in paper, i think 
    Drand_proportion: float = 0.1 #0.1 in paper
    Drl_proportion: float = 0.9 #0.9 in paper
    
    MPC_horizon: int = 12 #5 in paper: 0.2s with frame_skip = 5, *0.05 to get corresponding in s
    CEM_trajn: int = 150 #1000 in paper, but this is for random shooting
    SGD_steps: int = 4000 #not sure,plateaus at 800 on random steps, but later 800 doesn't impact much
    rollout_steps_per_aggregation: int = 1000 #9 full loops of 1000 steps before truncation: 9000 in paper

    ensemble_size: int = 5 #no idea

    #really not sure, check: 
    CEM_loopn: int = 5 #5 is standard 
    CEM_elitespicked: int = 10
    CEM_min_std: float = 0.03

    D_RAND_capacity: int = 2_000_000  #(just set to be large enough to never replace)
    D_RL_capacity: int = 1_000_000

    #random trajectory parameters: (fully eyeballed values)
    rand_traj_length: int = 500
    rand_traj_n: int = 500 #10 in paper

    #for in window rendering
    render_period: float = 300
    render_length: float = 100
    #for video saving mid training:



#NN paramaters
@dataclass                                   
class TransitionConfig:

    #input and output data: (env specific)
    dimension_in: int = 23                           
    dimension_out: int = 17

    #NN parameters:   #check paper what they used, 4x200 in PETS?                           
    n_width: int = 150                       
    n_layers: int = 3 #hidden layers, not including output

    #optimisation parameters:
    lr: float = 1e-3                                            
                
    #devide allocation:
    device: torch.device = torch.device("cuda") #"cuda" or "cpu"
