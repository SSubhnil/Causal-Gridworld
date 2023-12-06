"Stable BaseLines3 compatible custom Gymnasium environment"

# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from gymnasium.utils import seeding
import sys


class StochKingWindyGridWorldEnv(gym.Env):
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
        self.observation_space = spaces.Box(low=0, high=max(self.grid_height, self.grid_width), shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)  # 8 actions including diagonals
        self.step_counter = 0
        self.render_mode = 'human'
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

    def reset(self, seed=None):
        ''' resets the agent position back to the starting position'''
        self.observation = self.start_state
        self.step_counter = 0
        self.wind_generator()
        return np.array(self.observation)

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
        trunc = False
        if self.step_counter == 300:
            trunc = True
            return np.array(self.observation), -1, False, trunc, {}
        if self.observation == self.goal_state:
            return np.array(self.observation), 1, True, trunc, {}
        return np.array(self.observation), -0.5, False, trunc, {}

    def action_destination(self, state, action):
        '''set up destinations for each action in each state'''
        i, j = state

        ##############
        destination = dict()
        destination[self.actions['U']] = (max(i - 1 - self.realized_wind[j], 0), j)
        destination[self.actions['D']] = (max(min(i + 1 - self.realized_wind[j], \
                                                  self.grid_height - 1), 0), j)
        destination[self.actions['L']] = (max(i - self.realized_wind[j], 0), \
                                          max(j - 1, 0))
        destination[self.actions['R']] = (max(i - self.realized_wind[j], 0), \
                                          min(j + 1, self.grid_width - 1))
        destination[self.actions['UR']] = (max(i - 1 - self.realized_wind[j], 0), \
                                           min(j + 1, self.grid_width - 1))
        destination[self.actions['DR']] = (max(min(i + 1 - self.realized_wind[j], \
                                                   self.grid_height - 1), 0), min(j + 1, \
                                                                                  self.grid_width - 1))
        destination[self.actions['DL']] = (max(min(i + 1 - self.realized_wind[j], \
                                                   self.grid_height - 1), 0), max(j - 1, 0))
        destination[self.actions['UL']] = (max(i - 1 - self.realized_wind[j], 0), \
                                           max(j - 1, 0))

        return destination[action]

    def wind_generator(self):
        "Updates the wind profile every time reset() is called"
        rang = np.arange(-self.range_random_wind, self.range_random_wind + 1 )
        ##############
        # case 1 where all wind tiles are affected by the same noise scalar,
        # noise1 is a scalar value added to wind
        noise1 = self.np_random.choice(rang, 1, self.probablities)[0]
        # case 2  where each wind tile is affected by a different noise
        # noise2 is a vector added to wind
        noise2 = self.np_random.choice(rang, self.num_wind_tiles, self.probablities)
        noise = noise1 if self.noise_case==1 else noise2
        wind = np.copy(self.wind)
        wind[np.where( wind > 0 )] += noise
        self.realized_wind = wind

    "Simple render"
    # def render(self):
    #     grid = np.zeros((self.grid_height, self.grid_width), dtype=str)
    #     grid[self.start_position[0], self.start_position[1]] = 'S'
    #     grid[self.goal_position[0], self.goal_position[1]] = 'G'
    #     grid[self.current_position[0], self.current_position[1]] = 'X'
    #
    #     for row in grid:
    #         print(" ".join(row))

    "PyGame render"
    def render(self, mode='human'):
        if mode != 'human':
            super().render(mode=mode)
            return

        pygame.init()
        cell_size = 50  # Adjust the cell size as needed

        screen_width = self.grid_width * cell_size
        screen_height = self.grid_height * cell_size
        screen = pygame.display.set_mode((screen_width, screen_height))

        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((255, 255, 255))  # White background
            # Draw the start state
            self._draw_state(screen, self.start_state, (0, 255, 0), cell_size)

            # Draw the goal state
            self._draw_state(screen, self.goal_state, (0, 0, 255), cell_size)

            # Draw the current position
            self._draw_state(screen, self.observation, (255, 0, 0), cell_size)

            pygame.display.flip()
            clock.tick(10)  # Adjust the speed of the environment


    def _draw_state(self, screen, position, color, cell_size):
        i, j = position
        pygame.draw.rect(screen, color, (j * cell_size, i * cell_size, cell_size, cell_size))

