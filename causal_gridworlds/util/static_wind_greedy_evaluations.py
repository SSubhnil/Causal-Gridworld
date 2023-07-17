# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:39:09 2023

@author: SSubhnil
@details: Runs 100 greedy episodes per greedy evaluation for Static Windy GW.
          Available for both 4-actions and King-actions action spaces.
          If the wind is uniformaly random, the policy needs to show similar
          behavior over 100 greedy episodes.
          Implementiing this in the main code would make it very messy so a
          separate Q-value function is defined here.
"""

import numpy as np
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import random
import math
# matplotlib inline
import torch
import torch.nn as nn
import torch.optim as optim
# import check_test
# from plot_utils import plot_values


import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

np.random.seed(42)


class GreedyEvaluation:
    def __init__(self, ENV, GRID_DIMENSIONS):
        self.env = ENV
        self.grid_dimensions = GRID_DIMENSIONS
        self.num_episodes = 100
        
    def epsilon_greedy(self, Q, state, nA, eps = 0.01):
        #print("greedy_function_Q \n", Q)
        if np.random.uniform(0, 1) > eps:
            return np.argmax(Q[state[0]][state[1]])
        else:
            return random.choice(np.arange(nA))
    
    
    def Q_learn(self, Q, alpha, gamma=1.0):
        # Initialize action-value function (empty dictionary of arrays)
        nA = self.env.nA
        # Q = defaultdict(lambda: np.zeros(nA))
        
        step_count = np.empty((self.num_episodes, 1))
        
        # Loop over episodes
        for i_episode in range(1, self.num_episodes + 1):
            # monitor progress
            # if i_episode % 100 == 0:    
            
            # Initialize score
            score = 0
    
            # Observe S_0: the initial state
            state = self.env.reset()
    
            count = 0
            
            while True:
                count+=1
                
                # Choose action A_0 using policy derived from Q (e.g., eps-greedy)
                action = self.epsilon_greedy(Q, state, nA)
                
                next_state, reward, done, info = self.env.step(action)  # take action A, observe R', S'
                score += reward  # add reward to agent's score
                if not done:
                    # Choose action A_{t+1} using policy derived fromQ (e.g., eps-greedy)
                    next_action = self.epsilon_greedy(Q, state, nA)
    
                    # Update state and action
                    state = next_state  # S_t <- S_{t+1}
    
                if done:
                    step_count[i_episode - 1][0] = count                  
                    break
    
        return step_count    
    
    def run_algo(self, ALPHA, Q):
        step_count = self.Q_learn(Q, ALPHA)
        return step_count, np.mean(step_count)
        
class DQN_GreedyEvaluation:
    def __init__(self, ENV, GRID_DIMENSIONS, device):
        self.env = ENV
        self.grid_dimensions = GRID_DIMENSIONS
        self.num_episodes = 100
        # self.Q_store = np.empty((self.num_episodes, self.env.grid_height, self.env.grid_width, self.env.nA))
        # self.enco = encoder
        self.device = device
        
    def choose_action(self, model, state, eps = 0.01):
        #print("greedy_function_Q \n", Q)
        if np.random.rand() <= eps:
            return random.randrange(self.env.nA)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = model(state)
            # self.Q_store[self.enco.OneHotDecoder(state)] = q_values
            return torch.argmax(q_values).item()
    
    
    def DQN_learn(self, Q, gamma=0.98): # Q is the model and not the Q-table
        # Initialize action-value function (empty dictionary of arrays)
        nA = self.env.nA
        # Q = defaultdict(lambda: np.zeros(nA))
        
        step_count = np.empty((self.num_episodes, 1))
        
        
        # Loop over episodes
        for i_episode in range(1, self.num_episodes + 1):
            # monitor progress
            # if i_episode % 100 == 0:    
            
            # Initialize score
            score = 0
    
            # Observe S_0: the initial state
            state = torch.tensor(self.env.reset(), dtype=torch.float32).to(self.device)
            done = False
    
            count = 0
            
            while not done:
                count+=1
                
                # Choose action A_0 using policy derived from Q (e.g., eps-greedy)
                action = self.choose_action(Q, state)
                
                next_state, reward, done, info = self.env.step(action)  # take action A, observe R', S'
                score += reward  # add reward to agent's score
                
                # Choose action A_{t+1} using policy derived fromQ (e.g., eps-greedy)
                next_action = self.choose_action(Q, state, nA)

                # Update state and action
                state = torch.tensor(next_state, dtype=torch.float32).to(self.device)  # S_t <- S_{t+1}
    
            step_count[i_episode - 1][0] = count
    
        return step_count
    
    
    def run_algo(self, ALPHA, model):
        step_count = self.DQN_learn(model)
        return step_count, np.mean(step_count)
        
        


