import gymnasium as gym, highway_env
#for env modification
from highway_env.envs.parking_env import ParkingEnv
from highway_env.vehicle.kinematics import Vehicle
from parking_env_setup import env_params_discreteA_withcars

from stable_baselines3 import DQN

#for callbacks & vis
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import time

#for double DQN addition
import numpy as np
import torch as th
from torch.nn import functional as F

#alter the environment to have parked cars be stationary, i.e. their physics are not fully modelled each time step to significantly speed up eval & make full parking lot training possible
class ParkedVehicle(Vehicle):
 
    def step(self, dt: float) -> None:
        if self.impact is not None or self.road.record_history:  #skips the dynamics model eval unless the car has been collided into
            super().step(dt)
 
class StaticParkingEnv(ParkingEnv):
    def _create_vehicles(self) -> None:
        super()._create_vehicles()  #unchanged layout, goal and RNG usage
        ego = self.controlled_vehicles
        self.road.vehicles = [v if v in ego else ParkedVehicle.create_from(v) for v in self.road.vehicles]
        for obj in self.road.vehicles + self.road.objects:
            if obj not in ego:
                obj.check_collisions = False  #a pair is still checked if one side is the ego

gym.register(id="parking-static-v0", entry_point=StaticParkingEnv)

#redefine the SB3 train function from DQN for double DQN (de-correlating the noise in target eval) - define this new model as a class as a function of DQN
#only changed the target value generation loop
class DoubleDQN(DQN):

    #train function copied directly from SB3 docs: function to do optimisation steps on Q network to targets 
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            #generating target values:
            with th.no_grad():
                #normal SB3 DQN:
                # Compute the next Q-values using the target network (at s') (uses target network)
                #next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value (argmax_a (Q(s',a')))
                #next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                #next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target: (generate bellman step target value for Q function)
                #target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

                #Double DQN: 
                #argmax action selection a' at s' using current Q network:
                next_q_values = self.q_net(replay_data.next_observations)
                next_optimal_actions = next_q_values.argmax(dim = 1, keepdim=True)

                #evaluate Q values at (s', a') using target network:
                all_next_q_values_target = self.q_net_target(replay_data.next_observations) #over all actions 
                next_q_values_target = th.gather(all_next_q_values_target, dim=1, index=next_optimal_actions) #keep the actions corresponding to the actions selected from the current Q network

                #generate SL targets, one step Bellman
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values_target


            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            #added: to check no shape mismatch
            assert current_q_values.shape == target_q_values.shape, (current_q_values.shape, target_q_values.shape)

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


#for visualisation
class VisualEvalCallback(BaseCallback): #creates a rendering environment at periodic callback 
    def __init__(self, eval_freq: int, n_eval_episodes: int = 2, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print(f"\nVisual Evaluation at Step {self.num_timesteps}")
            
            eval_env = gym.make("parking-v0", config=env_params_discreteA_withcars, render_mode="human")
            
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


#DQN agent params
dqn_params = dict(
    learning_rate=6e-4,
    buffer_size=20_000,
    learning_starts=200,
    batch_size=150,
    gamma=0.98,
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

#make env & agent
#env = gym.make("parking-v0", config=env_params_discreteA_withcars)
env = gym.make("parking-static-v0", config=env_params_discreteA_withcars) 
env = Monitor(env)
model = DoubleDQN("MultiInputPolicy", env, **dqn_params) 

#for callbacks & visualisation 
headless_eval_env = gym.make("parking-v0", config=env_params_discreteA_withcars)
headless_eval_env = Monitor(headless_eval_env)
eval_callback = EvalCallback(
    headless_eval_env,
    eval_freq=5_000,                  
    n_eval_episodes=2,                      
    deterministic=True,                    
    render=False                             
)
visual_callback = VisualEvalCallback(eval_freq=1000, n_eval_episodes=2)


#train
print(f"Model is training on: {model.device}")
model.learn(total_timesteps=110_000, callback=[eval_callback, visual_callback], progress_bar=True)
model.save("ddqn_parking_1")