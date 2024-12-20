# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys

class KingWindyGridWorldEnv(gym.Env):
    '''Creates the King Windy GridWorld Environment''' # [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    def __init__(self, GRID_HEIGHT=7, GRID_WIDTH=10,\
                 WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0], \
                 START_STATE = (3, 0), GOAL_STATE = (3, 7),\
                 REWARD = -1):
        super(KingWindyGridWorldEnv, self).__init__()
        self.grid_height = GRID_HEIGHT
        self.grid_width = GRID_WIDTH
        self.wind = WIND
        self.start_state = START_STATE
        self.goal_state = GOAL_STATE
        self.observation = START_STATE
        self.reward = REWARD
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.grid_height),
                spaces.Discrete(self.grid_width)))
        self.actions = { 'U':0,   #up
                         'R':1,   #right
                         'D':2,   #down
                         'L':3,   #left
                         'UR':4,  #up-right
                         'DR':5,  #down-right
                         'DL':6,  #down-left
                         'UL':7 } #up-left
        self.nA = len(self.actions)
        self.step_counter = 0
        
        # set up destinations for each action in each state
        self.action_destination = np.empty((self.grid_height,self.grid_width), dtype=dict)
        for i in range(0, self.grid_height):
            for j in range(0, self.grid_width):
                destination = dict()
                destination[self.actions['U']] = (max(i - 1 - self.wind[j], 0), j)
                destination[self.actions['D']] = (max(min(i + 1 - self.wind[j], \
                                                    self.grid_height - 1), 0), j)
                destination[self.actions['L']] = (max(i - self.wind[j], 0),\
                                                       max(j - 1, 0))
                destination[self.actions['R']] = (max(i - self.wind[j], 0),\
                                                   min(j + 1, self.grid_width - 1))
                destination[self.actions['UR']] = (max(i - 1 - self.wind[j], 0),\
                                                   min(j + 1, self.grid_width - 1))
                destination[self.actions['DR']] = (max(min(i + 1 - self.wind[j],\
                                                   self.grid_height - 1), 0), min(j + 1,\
                                                   self.grid_width - 1))
                destination[self.actions['DL']] = (max(min(i + 1 - self.wind[j],\
                                              self.grid_height - 1), 0), max(j - 1, 0))         
                destination[self.actions['UL']] = (max(i - 1 - self.wind[j], 0),\
                                                   max(j - 1, 0))
                self.action_destination[i,j]=destination
                
        
    def step(self, action):
        """
        Parameters
        ----------
        action : 0 = Up, 1 = Right, 2 = Down, 3 = Left, 4 = Up-right, 
                 5 = Down-right, 6 = Down-left, 7 = Up-left
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                 Agent current position in the grid.
            reward (float) :
                 Reward is -1 at every step.
            episode_over (bool) :
                 True if the agent reaches the goal, False otherwise.
            info (dict) :
                 Contains no additional information.
        """
        self.step_counter += 1
        assert self.action_space.contains(action)
        self.observation = self.action_destination[self.observation][action]
        reward = -0.5  # Default reward for each step

        if self.observation == self.goal_state:
            reward = 1
            done = True
        elif self.step_counter == 300:
            reward = -1
            done = True
        else:
            done = False

        return self.observation, reward, done, {}
        
    def reset(self):
        ''' resets the agent position back to the starting position'''
        self.step_counter = 0
        self.observation = self.start_state
        return self.observation   

    def render(self, mode='human', close=False):
        ''' Renders the environment. Code borrowed and then modified 
            from
            https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py'''
        outfile = sys.stdout
        nS = self.grid_height * self.grid_width
        shape = (self.grid_height, self. grid_width)

        outboard = ""
        for y in range(-1, self.grid_height + 1):
            outline = ""
            for x in range(-1, self.grid_width + 1):
                position = (y, x)
                if self.observation == position:
                    output = "X"
                elif position == self.goal_state:
                    output = "G"
                elif position == self.start_state:
                    output = "S"
                elif x in {-1, self.grid_width } or y in {-1, self.grid_height}:
                    output = "#"
                else:
                    output = " "

                if position[1] == shape[1]:
                    output += '\n'
                outline += output
            outboard += outline
        outboard += '\n'
        outfile.write(outboard)

    def seed(self, seed=None):
        pass