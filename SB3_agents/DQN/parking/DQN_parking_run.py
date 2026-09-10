import gymnasium as gym, highway_env
from parking_env_setup import env_params_discreteA
env = gym.make("parking-v0", config=env_params_discreteA,  render_mode="rgb_array")
from gymnasium.wrappers import RecordVideo
import imageio

#agent
from stable_baselines3 import DQN
from pathlib import Path
MODEL_PATH = Path(__file__).parent / "dqn_parking_4"
model = DQN.load(MODEL_PATH) 
import time

#video recording
#env = RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True ) #gym by default can only record one rollout
#env.unwrapped.set_record_video_wrapper(env)
video_path = "videos/all_episodes_combined.mp4"
writer = imageio.get_writer(video_path, fps=30)

num_rollouts = 10
for episode in range(num_rollouts):
    # Reset the environment to generate the first observation
    obs, info = env.reset()
    writer.append_data(env.render())

    terminated = truncated = False
    while not (terminated or truncated):
        action, _states = model.predict(obs)

        # step (transition) through the environment with the action
        obs, reward, terminated, truncated, info = env.step(action)

        #env.render()
        #time.sleep(0.05)
        writer.append_data(env.render())

env.close()    

writer.close()
env.close()
