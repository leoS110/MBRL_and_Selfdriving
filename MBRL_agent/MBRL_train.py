#following alg in:
import torch
import torch.nn as nn
from dataclasses import dataclass, field  

#NN paramaters
@dataclass                                   
class Config:
    #input and output data:
    dimension_o: int = 1                                
    dimension_a: int = 2 
   

train_params = Config()