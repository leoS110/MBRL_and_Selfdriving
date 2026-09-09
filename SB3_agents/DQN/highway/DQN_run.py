#env
import gymnasium as gym, highway_env
from highway_env_setup import env_params_discreteA
env = gym.make("highway-v0", config=env_params_discreteA, render_mode="human")

#agent
from stable_baselines3 import DQN
from pathlib import Path
MODEL_PATH = Path(__file__).parent / "dqn_highway_3"
model = DQN.load(MODEL_PATH) 
import time

num_rollouts = 10
for episode in range(num_rollouts):
    # Reset the environment to generate the first observation
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action, _states = model.predict(obs)

        # step (transition) through the environment with the action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.05)

env.close()    
