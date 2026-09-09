import gymnasium as gym, highway_env
from stable_baselines3 import PPO

from highway_env_setup import env_params
env = gym.make("highway-fast-v0", config=env_params)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(10_000)
model.save("ppo_highway_2")

#delete & reload SB3 model
env.close() 
del model 