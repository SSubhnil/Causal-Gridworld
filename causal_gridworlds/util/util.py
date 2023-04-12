# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:35:55 2023

@author: SSubhnil

@details: Utility class for data plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')
my_path = os.path.dirname(__file__)

class PlotUtil:
    def __init__(self, CMAP, GRID_DIMENSIONS, TITLE, START_STATE, GOAL_STATE, SAVE_FIG = 0):
        self.cmap = CMAP
        self.grid = GRID_DIMENSIONS
        self.save_fig = SAVE_FIG
        self.image_title = TITLE
        self.start_state = START_STATE
        self.goal_state = GOAL_STATE
        self.directory = os.path.join(os.getcwd(), "..", "images")
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
    
    def showArrowsQ(self, Q, cmap, i_episode):
        "Generate a heatmap over Q_max with arrows for Q_argmax per state"
        
        # Generate heatmap of Q-table
        Q_max = np.max(Q, axis = 2)
        
        plt.imshow(Q_max, cmap=cmap, vmin=np.min(Q), vmax=np.max(Q))
        # plt.imshow(Q_max.T, cmap=cmap)
        clb = plt.colorbar()
        clb.set_label('Current Q-values')
        plt.title(f"Q-learn King - Episode {i_episode}")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.text(self.start_state[1], self.start_state[0], "S", ha='center', va='center', color='orange', fontsize = 11)
        plt.text(self.goal_state[1], self.goal_state[0], "G", ha='center', va='center', color='red', fontsize = 11)
        
        for x in range(self.grid[0]):
            for y in range(self.grid[1]):
                best_action = np.argmax(Q[x, y])
                if best_action == 3: # Left
                    dx, dy = (-0.2, 0)
                elif best_action == 0: # Up
                    dx, dy = (0, -0.2)
                elif best_action == 1: # Right
                    dx, dy = (0.2, 0)
                elif best_action == 2: # Down
                    dx, dy = (0, 0.2)
                elif best_action == 4: # Up-Right
                    dx, dy = (0.2, -0.2)
                elif best_action == 5: # Down-Right
                    dx, dy = (0.2, 0.2)
                elif best_action == 6: # Down-Left
                    dx, dy = (-0.2, 0.2)
                elif best_action == 7: # Up-Left
                    dx, dy = (-0.2, -0.2)
                plt.arrow(y, x, dx, dy, head_width=0.15, head_length=0.15, fc='white', ec='white')
        # plt.show()
        # if self.save_fig == 1:
        plt.savefig(os.path.join(self.directory, self.image_title +"-ep{}.png".format(i_episode)), dpi = 300)
        plt.close()
        
    def showMinMaxQ(self, Q, cmap, i_episode):
        "Generate a heatmap over all 4 Q-values per state. Q_max is lightest color"
        plt.imshow(np.max(Q, axis = 2), cmap=cmap, vmin=np.min(Q), vmax=np.max(Q))
        plt.colorbar()
        plt.title(f'Q-values after {i_episode} iterations')
        for x in range(self.grid[0]):
            for y in range(self.grid[1]):
                max_q = np.max(Q[x, y])
                min_q = np.min(Q[x, y])
                text = f'{max_q:.2f}\n{min_q:.2f}'
                plt.text(y, x, text, ha='center', va='center', color='white', fontsize=8)   
        plt.show()
        
    def showArrowsMinMaxQ(self, Q, cmap, i_episode):
        "Generate a heatmap with arrows and min max Q-values for each state"
        plt.imshow(np.max(Q, axis = 2), cmap=cmap, vmin=np.min(Q), vmax=np.max(Q))
        plt.colorbar()
        plt.title(f'Q-values after {i_episode} episodes')
        for x in range(self.grid[0]):
            for y in range(self.grid[1]):
                max_q = np.max(Q[x, y])
                min_q = np.min(Q[x, y])
                best_action = np.argmax(Q[x, y])
                if best_action == 3: # Left
                    dx, dy = (-0.2, 0)
                elif best_action == 0: # Up
                    dx, dy = (0, -0.2)
                elif best_action == 1: # Right
                    dx, dy = (0.2, 0)
                elif best_action == 2: # Down
                    dx, dy = (0, 0.2)
                elif best_action == 4: # Up-Right
                    dx, dy = (0.2, -0.2)
                elif best_action == 5: # Down-Right
                    dx, dy = (0.2, 0.2)
                elif best_action == 6: # Down-Left
                    dx, dy = (-0.2, 0.2)
                elif best_action == 7: # Up-Left
                    dx, dy = (-0.2, -0.2)
                text = f'{max_q:.2f}\n{min_q:.2f}'
                plt.text(y, x, text, ha='center', va='center', color='white', fontsize=7)
                plt.arrow(y, x, dx, dy, head_width=0.15, head_length=0.15, fc='black', ec='black')
        plt.show()
