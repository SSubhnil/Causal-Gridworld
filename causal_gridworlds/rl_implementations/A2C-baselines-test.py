# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:12:31 2023

@author: shubh
@details: A2C baseline implementation for GridWorld
"""

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from custom_envs.BL3_stoch_king_windy_gridworld_env import StochKingWindyGridWorldEnv

import wandb

env = make_vec_env(StochKingWindyGridWorldEnv, n_envs=4)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("a2c_stoch_king")



for i in range(0, 1000):
    obs = env.reset()
    done = False
    while not done:
