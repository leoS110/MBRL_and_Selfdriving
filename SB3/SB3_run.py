import gymnasium as gym, highway_env
from stable_baselines3 import PPO
from pathlib import Path
from gymnasium.wrappers import RecordVideo

MODEL_PATH = Path(__file__).parent / "ppo_highway_1"
model = PPO.load(MODEL_PATH) 
env = gym.make("highway-fast-v0", render_mode="rgb_array")

#video recording
env = RecordVideo(env, video_folder="SB3/videos", episode_trigger=lambda e: e == 0 , fps=110) 
env.unwrapped.set_record_video_wrapper(env)

# Reset the environment to generate the first observation
obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action, _states = model.predict(obs)

    # step (transition) through the environment with the action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()    
