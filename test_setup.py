import torch

print("PyTorch version:   ", torch.__version__)
print("Compiled with CUDA:", torch.backends.cuda.is_built())
print("CUDA toolkit built:", torch.version.cuda)
print("Is CUDA available to PyTorch?:", torch.cuda.is_available())

import gymnasium as gym, mujoco
import time
N_STEPS = 1000  # one full episode (HalfCheetah truncates at 1000 steps)
 
env = gym.make("HalfCheetah-v5", render_mode="human")
obs, info = env.reset(seed=0)
 
total_reward = 0.0
for step in range(N_STEPS):
    action = env.action_space.sample()  # random torques in [-1, 1] for the 6 joints
    obs, reward, terminated, truncated, info = env.step(action)  # step() also draws the frame
    total_reward += reward
    time.sleep(env.unwrapped.dt)  # slow down to roughly real time (dt = 0.05 s)
 
    if terminated or truncated:
        print(f"Episode done after {step + 1} steps, return = {total_reward:.1f}")
        obs, info = env.reset()
        total_reward = 0.0
 
env.close()
 