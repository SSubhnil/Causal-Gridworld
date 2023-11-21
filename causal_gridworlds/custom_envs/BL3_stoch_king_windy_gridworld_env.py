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

    def __init__(self, grid_height=7, grid_width=10):
        super(StochKingWindyGridWorldEnv, self).__init__()

        self.grid_height = grid_height
        self.grid_width = grid_width
        self.observation_space = self.observation_space = spaces.Box(low=0, high=max(grid_height, grid_width), shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)  # 8 actions including diagonals

        self.start_position = np.array([3, 0])
        self.goal_position = np.array([3, 7])
        self.current_position = np.copy(self.start_position)
        self.step_counter = 0
        self.wind_strengths = None  # To be set during reset
        self.render_mode = 'human'

    def reset(self, seed=0):
        self.current_position = np.copy(self.start_position)
        self._generate_wind()
        self.step_counter = 0
        return self.current_position.astype(np.float32)

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"

        wind_effect = self.wind_strengths[self.current_position[1]]
        next_position = self._get_next_position(action, wind_effect)

        self.current_position = next_position
        done = np.array_equal(self.current_position, self.goal_position)
        trunc = True if self.step_counter == 300 else False
        reward = -1.0 if not done else 0.0
        self.step_counter+=1

        return self.current_position.astype(np.float32), reward, done, trunc, {}

    def _generate_wind(self):
        self.wind_strengths = np.random.randint(-1, 2, size=self.grid_width)

    def _get_next_position(self, action, wind_effect):
        next_position = np.copy(self.current_position)

        if action == 0:  # Up
            next_position[0] = max(0, self.current_position[0] - 1 - wind_effect)
        elif action == 1:  # Down
            next_position[0] = min(self.grid_height - 1, self.current_position[0] + 1 - wind_effect)
        elif action == 2:  # Left
            next_position[1] = max(0, self.current_position[1] - 1)
        elif action == 3:  # Right
            next_position[1] = min(self.grid_width - 1, self.current_position[1] + 1)
        elif action == 4:  # Up-Right
            next_position[0] = max(0, self.current_position[0] - 1 - wind_effect)
            next_position[1] = min(self.grid_width - 1, self.current_position[1] + 1)
        elif action == 5:  # Down-Right
            next_position[0] = min(self.grid_height - 1, self.current_position[0] + 1 - wind_effect)
            next_position[1] = min(self.grid_width - 1, self.current_position[1] + 1)
        elif action == 6:  # Down-Left
            next_position[0] = min(self.grid_height - 1, self.current_position[0] + 1 - wind_effect)
            next_position[1] = max(0, self.current_position[1] - 1)
        elif action == 7:  # Up-Left
            next_position[0] = max(0, self.current_position[0] - 1 - wind_effect)
            next_position[1] = max(0, self.current_position[1] - 1)

        return next_position

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
            self._draw_state(screen, self.start_position, (0, 255, 0), cell_size)

            # Draw the goal state
            self._draw_state(screen, self.goal_position, (0, 0, 255), cell_size)

            # Draw the current position
            self._draw_state(screen, self.current_position, (255, 0, 0), cell_size)

            pygame.display.flip()
            clock.tick(10)  # Adjust the speed of the environment


    def _draw_state(self, screen, position, color, cell_size):
        i, j = position
        pygame.draw.rect(screen, color, (j * cell_size, i * cell_size, cell_size, cell_size))

