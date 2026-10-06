# MBRL_selfdriving

This is a collection of notes and mini projects I've built in summer 2026, around a full time internship, to lay the groundwork for future projects in RL and control. 


Theory: (months 1 and 2)  

Independantly followed the following courses:  
- Supervised Learning: Cornell CS4780
- Intro to RL: UC Berkeley CS189
- Deep RL: UC Berkeley CS285
Notes are attached in the notes folder of this repo  

Projects: (month 3)  

Made 2 projects focusing on different skills:  
1) Adapting an exisiting model-free algorithm to an application: Adapted the SB3 DQN algorithm to self-park a car around obstacles in the highwayenv parking environment. Adaption included implementing manually adding double DQN, editing the environment to improve compute speed, and parameter tuning.
See SB3_agents folder  

2) Building from the ground up: Implemented model-based RL from scratch in PyTorch, a PETS variant (Chua et al., 2018): hand-coded the dynamics ensemble, CEM MPC planner and data-aggregation loop (MuJoCo HalfCheetah).   
See MBRL_agent folder

File structure:  
Training: 
config_train: dataclass holding key training algorithm, MPC, and transition model parameters, used during training  
MBRL_train: central training algorithm + save model ensemble  
MPC_vectorised: CEM MPC loop, vectorised due to implementation of passing all candidates to the transition model at once  

Run:  
config_run: dataclass holding key training algorithm, MPC, and transition model parameters, used when model is ran  
MBRL_run_liverender: load saved model ensemble + run control loop in real time human view render  
MBRL_run_videocollection: load saved model ensemble + run control loop in real time to collect demo videos  
MPC_run: CEM MPC loop (also vectorised), can change the step_reward_eval(state, action) function to change the goal at run time  


Utils:
replay_buffer & transitionbatch: taken from since only used as a util: code taken directly from facebook research / UC Berkeley MBRL library   
https://github.com/facebookresearch/mbrl-lib/tree/main
https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/types.py
https://arxiv.org/abs/2104.10159  
only the core functions kept: all references to storing / manipulating trajectories removed (incl. in add & save functions)  
velocity_plot_wrapper: just a plotting tool to have a subplot with x velocity in the video collection, AI coded since just plotting

Implementation notes for both of these are in the notes folder. 

Also see my repo https://github.com/leoS110/ML-Summer-26-NN-by-hand for where after following Cornell CS4780 I hand coded a neural network + backprop in numpy