import gymnasium as gym, highway_env
from parking_env_setup import env_params_discreteA
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import time

#for visualisation
class VisualEvalCallback(BaseCallback): #creates a rendering environment at periodic callback 
    def __init__(self, eval_freq: int, n_eval_episodes: int = 2, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print(f"\nVisual Evaluation at Step {self.num_timesteps}")
            
            eval_env = gym.make("parking-v0", config=env_params_discreteA, render_mode="human")
            
            for episode in range(self.n_eval_episodes):
                obs, info = eval_env.reset()
                done = False
                truncated = False
        
                while not (done or truncated):
                    action, _states = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = eval_env.step(action)
                    
                    # time.sleep(0.05) 
            eval_env.close()    
        return True

#DQN agent
dqn_params = dict(
    learning_rate=8e-4,
    buffer_size=20_000,
    learning_starts=200,
    batch_size=150,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=200,
    policy_kwargs=dict(net_arch=[256, 256]), 
    seed=0,
    verbose=1,
    device="cpu", #options: "cpu", "cuda", "cuda:0", or "mps"
    exploration_fraction=0.1, #fraction of entire training period over which the exploration rate is reduced, seems low?
    exploration_initial_eps=1.0, 
    exploration_final_eps=0.05, 
)
env = gym.make("parking-v0", config=env_params_discreteA)
env = Monitor(env)
model = DQN("MultiInputPolicy", env, **dqn_params) 

#for callbacks & visualisation 
headless_eval_env = gym.make("parking-v0", config=env_params_discreteA)
headless_eval_env = Monitor(headless_eval_env)
eval_callback = EvalCallback(
    headless_eval_env,
    eval_freq=5_000,                  
    n_eval_episodes=2,                      
    deterministic=True,                    
    render=False                             
)
visual_callback = VisualEvalCallback(eval_freq=5_000, n_eval_episodes=2)


#train
print(f"Model is training on: {model.device}")
model.learn(total_timesteps=80_000, callback=[eval_callback, visual_callback])
model.save("dqn_parking_5")