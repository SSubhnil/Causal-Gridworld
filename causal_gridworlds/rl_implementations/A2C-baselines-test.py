# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:12:31 2023

@author: shubh
@details: A2C baseline implementation for GridWorld
"""

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from custom_envs.BL3_stoch_king_windy_gridworld_env import StochKingWindyGridWorldEnv

env = make_vec_env(StochKingWindyGridWorldEnv, n_envs = 2)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000)
model.save("a2c_stoch_king")

obs = env.reset()
steps = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    steps += 1
    print(steps)