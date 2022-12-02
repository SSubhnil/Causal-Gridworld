# -*- coding: utf-8 -*-
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

from envs.stoch_windy_gridworld_env import StochWindyGridWorldEnv

env = StochWindyGridWorldEnv()
observation = env.reset()

done = False
sample_path = []
while not done:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action, force_noise=None)
    sample_path.append(info)
    print(observation, reward, done, info)
    