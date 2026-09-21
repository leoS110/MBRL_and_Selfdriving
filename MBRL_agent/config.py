
from dataclasses import dataclass, field 
import torch


@dataclass                                   
class TrainConfig:
    #environment specific paramaters: (env specific)
    dimension_o: int = 1                                
    dimension_a: int = 2 
    a_min: float = 0.0
    a_max: float = 1.0

    aggregation_iterations: int = 5

    #algorithm parameters:
    SGD_batch_size: int = 512 
    Drand_proportion: float = 0.1
    Drl_proportion: float = 0.9
    MPC_horizon: int = 10 
    CEM_trajn: int = 1000
    SGD_steps: int = 60
    rollouts_per_aggregation: int = 400

    ensemble_size: int = 5

    #really not sure, check: 
    CEM_loopn: int = 30 
    CEM_elitespicked: int = 100

    D_RAND_capacity: int = 1_000_000  #(just set to be large enough to never replace)
    D_RL_capacity: int = 1_000_000

    #random trajectory parameters: (fully eyeballed values)
    rand_traj_length: int = 200
    rand_traj_n: int = 100


#NN paramaters
@dataclass                                   
class TransitionConfig:

    #input and output data: (env specific)
    dimension_in: int = 3                             
    dimension_out: int = 2

    #NN parameters:   #check paper what they used                           
    n_width: int = 50                       
    n_layers: int = 4 #hidden layers, not including output

    #optimisation parameters:
    lr: float = 1e-3                                            
                
    #devide allocation:
    device: torch.device = torch.device("cpu") #or "cuda"
    seed: int = 0 