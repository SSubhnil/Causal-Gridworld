# -*- coding: utf-8 -*-
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

from envs.stoch_windy_gridworld_env_v2 import StochWindyGridWorldEnv_V2

env = StochWindyGridWorldEnv_V2()
observation = env.reset()

done = False
sample_path = []
while not done:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    sample_path.append(info)
    print(observation, reward, done, info)
    