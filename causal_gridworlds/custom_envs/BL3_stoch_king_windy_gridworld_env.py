"Stable BaseLines3 compatible custom Gymnasium environment"

# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import sys


class StochKingWindyGridWorldEnv(gym.Env):
    '''Creates the Stochastic Windy GridWorld Environment
       NOISE_CASE = 1: the noise is a scalar added to the wind tiles, i.e,
                       all wind tiles are changed by the same amount
       NOISE_CASE = 2: the noise is a vector added to the wind tiles, i.e,
                       wind tiles are changed by different amounts.
    '''

    def __init__(self, GRID_HEIGHT=7, GRID_WIDTH=10, \
                 WIND=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0], \
                 START_STATE=np.array([3, 0]), GOAL_STATE=np.array([3, 7]), \
                 REWARD=-1, RANGE_RANDOM_WIND=1, \
                 PROB=[1. / 3, 1. / 3, 1. / 3], \
                 NOISE_CASE=2):
        super(StochKingWindyGridWorldEnv, self).__init__()
        self.seed_value = None
        np.random.seed(self.seed_value)
        # self.seed(self.seed_value)
        self.grid_height = GRID_HEIGHT
        self.grid_width = GRID_WIDTH
        self.wind = np.array(WIND)
        self.realized_wind = np.array(WIND)
        self.start_state = START_STATE
        self.goal_state = GOAL_STATE
        self.reward = REWARD
        self.range_random_wind = RANGE_RANDOM_WIND
        self.probablities = PROB
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Discrete(self.grid_height * self.grid_width)
        # self.observation_space = spaces.Tuple((
        #         spaces.Discrete(self.grid_height),
        #          spaces.Discrete(self.grid_width)
        # ))

        # spaces.Box(self.grid_height, self.grid_width)
        self.actions = {'U': 0,  # up
                        'R': 1,  # right
                        'D': 2,  # down
                        'L': 3,  # left
                        'UR': 4,  # up-right
                        'DR': 5,  # down-right
                        'DL': 6,  # down-left
                        'UL': 7}  # up-left
        self.num_wind_tiles = np.count_nonzero(self.wind)
        self.noise_case = NOISE_CASE
        self.nA = len(self.actions)
        self.step_counter = 0

    def wind_generator(self):
        "Updates the wind profile every time reset() is called"
        rang = np.arange(-self.range_random_wind, self.range_random_wind + 1)
        ##############
        # case 1 where all wind tiles are affected by the same noise scalar,
        # noise1 is a scalar value added to wind
        noise1 = self.np_random.choice(rang, 1, self.probablities)[0]
        # case 2  where each wind tile is affected by a different noise
        # noise2 is a vector added to wind
        noise2 = self.np_random.choice(rang, self.num_wind_tiles, self.probablities)
        noise = noise1 if self.noise_case == 1 else noise2
        wind = np.copy(self.wind)
        wind[np.where(wind > 0)] += noise
        self.realized_wind = wind

    def action_destination(self, state, action):
        '''set up destinations for each action in each state'''
        i, j = state

        ##############
        destination = dict()
        destination[self.actions['U']] = np.array([max(i - 1 - self.realized_wind[j], 0), j])
        destination[self.actions['D']] = np.array([max(min(i + 1 - self.realized_wind[j], \
                                                  self.grid_height - 1), 0), j])
        destination[self.actions['L']] = np.array([max(i - self.realized_wind[j], 0), \
                                          max(j - 1, 0)])
        destination[self.actions['R']] = np.array([max(i - self.realized_wind[j], 0), \
                                          min(j + 1, self.grid_width - 1)])
        destination[self.actions['UR']] = np.array([max(i - 1 - self.realized_wind[j], 0), \
                                           min(j + 1, self.grid_width - 1)])
        destination[self.actions['DR']] = np.array([max(min(i + 1 - self.realized_wind[j], \
                                                   self.grid_height - 1), 0), min(j + 1, \
                                                                                  self.grid_width - 1)])
        destination[self.actions['DL']] = np.array([max(min(i + 1 - self.realized_wind[j], \
                                                   self.grid_height - 1), 0), max(j - 1, 0)])
        destination[self.actions['UL']] = np.array([max(i - 1 - self.realized_wind[j], 0), \
                                           max(j - 1, 0)])

        return destination[action]

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
            truncated (bool):
                 True if the episode terminates after N steps.
            info (dict) :
                 Contains the realized noise that is added to the wind in each
                 step. However, official evaluations of your agent are not
                 allowed to use this for learning.
        """
        self.step_counter += 1
        assert self.action_space.contains(action)
        self.observation = self.action_destination(self.observation, action)
        if self.step_counter == 300:
            return self.observation, -1, True, True, {} # Second bool for truncation
        if np.array_equal(self.observation, self.goal_state):
            return self.observation, 1, True, False, {}
        return self.observation, -0.5, False, False, {}

    def reset(self, seed=None):
        ''' resets the agent position back to the starting position'''
        self.observation = self.start_state
        self.step_counter = 0
        self.wind_generator()
        return self.observation

    def render(self, mode='human', close=False):
        ''' Renders the environment. Code borrowed and then modified
            from
            https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py'''
        outfile = sys.stdout
        nS = self.grid_height * self.grid_width
        shape = (self.grid_height, self.grid_width)

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
                elif x in {-1, self.grid_width} or y in {-1, self.grid_height}:
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
        np.random.seed(seed)
        # self.np_random, seed = seeding.np_random(seed)
        return [seed]