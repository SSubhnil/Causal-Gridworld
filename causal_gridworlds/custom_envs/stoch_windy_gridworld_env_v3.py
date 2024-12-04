# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:36:08 2023

@author: shubh
@details: Stochastic Windy Gridworld environment where wind changes every episode.
"""

# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import sys


class StochWindyGridWorldEnv_V3(gym.Env):
    '''Creates the Stochastic Windy GridWorld Environment
       NOISE_CASE = 1: the noise is a scalar added to the wind tiles, i.e,
                       all wind tiles are changed by the same amount
       NOISE_CASE = 2: the noise is a vector added to the wind tiles, i.e,
                       wind tiles are changed by different amounts.
    '''
    def __init__(self, GRID_HEIGHT=7, GRID_WIDTH=10,\
                 WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0], \
                 START_STATE = (3, 0), GOAL_STATE = (3, 7),\
                 REWARD = -1, RANGE_RANDOM_WIND=1,\
                 PROB=[1./3, 1./3, 1./3],\
                 NOISE_CASE = 2):
        self.seed()
        self.grid_height = GRID_HEIGHT
        self.grid_width = GRID_WIDTH
        self.wind = np.array(WIND)
        self.realized_wind = np.array(WIND)
        self.start_state = START_STATE
        self.goal_state = GOAL_STATE
        self.reward = REWARD
        self.range_random_wind = RANGE_RANDOM_WIND
        self.probabilities = PROB
        self.action_space =  spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.grid_height),
                spaces.Discrete(self.grid_width)))
        self.actions = { 'U':0,   #up
                         'R':1,   #right
                         'D':2,   #down
                         'L':3}  #left
        self.num_wind_tiles = np.count_nonzero(self.wind)
        self.noise_case = NOISE_CASE
        self.nA = len(self.actions)
        self.step_counter = 0
    
    def wind_generator(self):
        "Updates the wind profile every time reset() is called"
        rang = np.arange(-self.range_random_wind, self.range_random_wind + 1 )
        ##############
        # case 1 where all wind tiles are affected by the same noise scalar, 
        # noise1 is a scalar value added to wind
        noise1 = self.np_random.choice(rang, 1, self.probabilities)[0]
        # case 2  where each wind tile is affected by a different noise 
        # noise2 is a vector added to wind
        noise2 = self.np_random.choice(rang, self.num_wind_tiles, self.probabilities)
        noise = noise1 if self.noise_case==1 else noise2
        wind = np.copy(self.wind)
        wind[np.where( wind > 0 )] += noise
        self.realized_wind = wind        
                
    def action_destination(self, state, action):
        '''set up destinations for each action in each state'''
        i, j = state
        if j < 0 or j >= len(self.realized_wind):
            raise ValueError(f"Invalid column index: {j}, valid range is 0 to {len(self.realized_wind) - 1}")

        ##############
        destination = dict()
        destination[self.actions['U']] = self.clamp_to_grid(i - 1 - self.realized_wind[j], j)
        destination[self.actions['D']] = self.clamp_to_grid(i + 1 - self.realized_wind[j], j)
        destination[self.actions['L']] = self.clamp_to_grid(i - self.realized_wind[j], j - 1)
        destination[self.actions['R']] = self.clamp_to_grid(i - self.realized_wind[j], j + 1)
       
        return destination[action]

    def clamp_to_grid(self, *args):
        """Clamp either indices or a state tuple to grid boundaries."""
        if len(args) == 1 and isinstance(args[0], tuple):
            # Handle state tuple
            i, j = args[0]
        elif len(args) == 2:
            # Handle separate row and column arguments
            i, j = args
        else:
            raise TypeError("Invalid arguments: Provide either a tuple or two integers.")

        return (
            max(0, min(self.grid_height - 1, i)),  # Clamp row index
            max(0, min(self.grid_width - 1, j))  # Clamp column index
        )
    
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
                 Contains the realized noise that is added to the wind in each 
                 step. However, official evaluations of your agent are not 
                 allowed to use this for learning.
        """
        self.step_counter += 1
        assert self.action_space.contains(action)
        self.observation = self.action_destination(self.observation, action)
        if self.step_counter == 300:
            return self.observation, -1, True, {}
        if self.observation == self.goal_state:
            return self.observation, 1, True, {}
        return self.observation, -0.5, False, {}
        
    def reset(self):
        ''' resets the agent position back to the starting position'''
        self.step_counter = 0
        self.observation = self.start_state
        self.wind_generator()
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
        ''' sets the seed for the envirnment'''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]