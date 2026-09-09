
env_params: dict = {
    "observation": {
        "type": "Kinematics"
    },
    "action": {
        "type": "ContinuousAction",
    },
    "lanes_count": 4,
    "vehicles_count": 5,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 10,  # [s]
    "ego_spacing": 2,
    "vehicles_density": 1,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes.
    "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0,  # The reward received at each lane change action.
    "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, 1].
    "normalize_reward": True,
    "offroad_terminal": False,
    "simulation_frequency": 10,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": None
}



env_params_default: dict = {
    "observation": {
        "type": "Kinematics"
    },
    "action": {
        "type": "ContinuousAction",
    },
    "lanes_count": 4,
    "vehicles_count": 5,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 10,  # [s]
    "ego_spacing": 2,
    "vehicles_density": 1,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes.
    "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0,  # The reward received at each lane change action.
    "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, 1].
    "normalize_reward": True,
    "offroad_terminal": False,
    "simulation_frequency": 10,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": None
}


env_params_discreteA: dict = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 3,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "absolute": True,
        "order": "sorted", 
        "normalize": True,
        "clip": True,
        "see_behind": True, 
        #"observe_intentions": False,
        #"include_obstacles": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    #env
    "lanes_count": 2,
    "vehicles_count": 6,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 15,  # [s]
    "ego_spacing": 2,
    "vehicles_density": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    #reward
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes.
    "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0,  # The reward received at each lane change action.
    "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, 1].
    "normalize_reward": True,
    "offroad_terminal": False,
    #freq
    "simulation_frequency": 5,  # [Hz]
    "policy_frequency": 5,  # [Hz]
    #view
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": None
}



#default setup:
#{
#    "observation": {
#        "type": "Kinematics"
#    },
#    "action": {
#        "type": "DiscreteMetaAction",
#    },
#    "lanes_count": 4,
#    "vehicles_count": 50,
#    "controlled_vehicles": 1,
#    "initial_lane_id": None,
#    "duration": 40,  # [s]
#    "ego_spacing": 2,
#    "vehicles_density": 1,
#    "collision_reward": -1,  # The reward received when colliding with a vehicle.
#    "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes.
#    "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"].
#    "lane_change_reward": 0,  # The reward received at each lane change action.
#    "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, 1].
#    "normalize_reward": True,
#    "offroad_terminal": False,
#    "simulation_frequency": 15,  # [Hz]
#    "policy_frequency": 1,  # [Hz]
#    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
#    "screen_width": 600,  # [px]
#    "screen_height": 150,  # [px]
#    "centering_position": [0.3, 0.5],
#    "scaling": 5.5,
#    "show_trajectories": False,
#    "render_agent": True,
#    "offscreen_rendering": None
#}