import torch

print("PyTorch version:   ", torch.__version__)
print("Compiled with CUDA:", torch.backends.cuda.is_built())
print("CUDA toolkit built:", torch.version.cuda)
print("Is CUDA available to PyTorch?:", torch.cuda.is_available())