import torch

print("PyTorch version:   ", torch.__version__)
print("Compiled with CUDA:", torch.backends.cuda.is_built())
print("CUDA toolkit built:", torch.version.cuda)
print("Is CUDA available to PyTorch?:", torch.cuda.is_available())

import inspect, stable_baselines3
from stable_baselines3 import DQN
print(stable_baselines3.__version__)
print(inspect.getsourcefile(DQN))    
print(inspect.getsource(DQN.train))  