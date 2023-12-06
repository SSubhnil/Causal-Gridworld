from __future__ import print_function
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import sys


class WindyGridWorldEnv(gym.Env):
    '''Creates the Windy GridWorld Environment'''  # [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def __init__(self, GRID_HEIGHT=7, GRID_WIDTH=10, \
                 WIND=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0], \
                 START_STATE=(3, 0), GOAL_STATE=(3, 7), \
                 REWARD=-1):
        super(WindyGridWorldEnv, self).__init__()
        self.grid_height = GRID_HEIGHT
        self.grid_width = GRID_WIDTH
        self.wind = WIND
        self.start_state = START_STATE
        self.goal_state = GOAL_STATE
        self.reward = REWARD
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=max(self.grid_height, self.grid_width), shape=(2,),
                                            dtype=np.float32)
        self.actions = {'U': 0,  # up
                        'R': 1,  # right
                        'D': 2,  # down
                        'L': 3}  # left
        self.step_counter = 0

        # set up destinations for each action in each state
        self.action_destination = np.empty((self.grid_height, self.grid_width), dtype=dict)
        for i in range(0, self.grid_height):
            for j in range(0, self.grid_width):
                destination = dict()
                destination[self.actions['U']] = (max(i - 1 - self.wind[j], 0), j)
                destination[self.actions['D']] = (max(min(i + 1 - self.wind[j], \
                                                          self.grid_height - 1), 0), j)
                destination[self.actions['L']] = (max(i - self.wind[j], 0), \
                                                  max(j - 1, 0))
                destination[self.actions['R']] = (max(i - self.wind[j], 0), \
                                                  min(j + 1, self.grid_width - 1))
                self.action_destination[i, j] = destination
        self.nA = len(self.actions)

    def step(self, action):
        """
        Parameters
        ----------
        action : 0 = Up, 1 = Right, 2 = Down, 3 = Left
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
        trunc = False
        if self.step_counter == 300:
            trunc = True
            return np.array(self.observation), -1, False, trunc, {}
        if self.observation == self.goal_state:
            return np.array(self.observation), 2, True, trunc, {}
        return np.array(self.observation), -0.5, False, trunc, {}

    def reset(self, seed = None):
        ''' resets the agent position back to the starting position'''
        self.step_counter = 0
        self.observation = self.start_state
        return np.array(self.observation)

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

    def seed(self, seed=None):
        pass
