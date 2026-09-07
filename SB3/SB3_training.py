import gymnasium as gym, highway_env
from stable_baselines3 import PPO

env = gym.make("highway-fast-v0")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(20_000)
model.save("ppo_highway_1")

#delete & reload SB3 model
env.close() 
del model 