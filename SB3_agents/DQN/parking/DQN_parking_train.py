#env
import gymnasium as gym, highway_env
from parking_env_setup import env_params_discreteA
env = gym.make("parking-v0", config=env_params_discreteA)

#agent
from stable_baselines3 import DQN

dqn_params = dict( #any parameters not mentioned are set as their default
    learning_rate=8e-4,
    buffer_size=15_000,
    learning_starts=200,
    batch_size=50,
    gamma=0.98,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=200,
    policy_kwargs=dict(net_arch=[256, 256]), #MLP policy
    #tensorboard_log="runs/",
    seed=0,
    verbose=1,
)

model = DQN("MultiInputPolicy", env, **dqn_params) #multi input policy to work with parking env
model.learn(total_timesteps=20_000)
model.save("dqn_parking_1")

