# -*- coding: utf-8 -*-
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

from envs.king_windy_gridworld_env import KingWindyGridWorldEnv

env = KingWindyGridWorldEnv()
observation = env.reset()

env.reset()

done = False

while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(observation, reward, done, info)
    