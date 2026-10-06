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

Also see my repo https://github.com/leoS110/ML-Summer-26-NN-by-hand for where after following Cornell CS4780 I hand coded a neural network + backprop in numpy