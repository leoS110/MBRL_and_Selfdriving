import numpy as np

env_params_discreteA: dict =  {
    "observation": {
        "type": "KinematicsGoal",
        "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False
    },
    "action": {
        "type": "DiscreteAction"
    },
    "reward_weights": [1.5, 0.9, 0.3, 0.3, 0.02, 0.02],
    "success_goal_reward": 0.2,
    "collision_reward": -4,
    "steering_range": np.deg2rad(45),
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 15,
    "controlled_vehicles": 1,
    "vehicles_count": 0,
    "add_walls": False,
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": None
}