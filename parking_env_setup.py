import numpy as np

env_params_discreteA_withcars: dict =  {
    "observation": {
        "type": "KinematicsGoal",
        "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False
    },
    "action": {
        "type": "DiscreteAction",
        "actions_per_axis": 7, #standard is 3
        "acceleration_range": (-3.0, 3.0) #standard is -5, 5
    },
    "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
    "success_goal_reward": 0.12,
    "collision_reward": -20,
    "steering_range": np.deg2rad(45),
    "simulation_frequency": 10,
    "policy_frequency": 5,
    "duration": 14,
    "controlled_vehicles": 1,
    "vehicles_count": 26,
    "add_walls": True,
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": None
}

env_params_discreteA_empty: dict =  {
    "observation": {
        "type": "KinematicsGoal",
        "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False
    },
    "action": {
        "type": "DiscreteAction",
        "actions_per_axis": 7, #standard is 3
        "acceleration_range": (-3.0, 3.0) #standard is -5, 5
    },
    "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
    "success_goal_reward": 0.12,
    "collision_reward": -20,
    "steering_range": np.deg2rad(45),
    "simulation_frequency": 10,
    "policy_frequency": 5,
    "duration": 14,
    "controlled_vehicles": 1,
    "vehicles_count": 0,
    "add_walls": False,
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": None
}

env_params_discreteA_empty_modreward: dict =  {
    "observation": {
        "type": "KinematicsGoal",
        "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False
    },
    "action": {
        "type": "DiscreteAction",
        "actions_per_axis": 7, #standard is 3
        "acceleration_range": (-3.0, 3.0) #standard is -5, 5
    },
    "reward_weights": [1, 0.4, 0.1, 0.1, 0.03, 0.03],
    "success_goal_reward": 0.12,
    "collision_reward": -20,
    "steering_range": np.deg2rad(45),
    "simulation_frequency": 10,
    "policy_frequency": 5,
    "duration": 14,
    "controlled_vehicles": 1,
    "vehicles_count": 0,
    "add_walls": False,
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": None
}

