# -*- coding: utf-8 -*-
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

from envs.stoch_king_windy_gridworld_env import StochKingWindyGridWorldEnv

env = StochKingWindyGridWorldEnv()
observation = env.reset()

done = False

while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    
    