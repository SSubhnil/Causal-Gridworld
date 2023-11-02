# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:13:10 2023

@author: SSubhnil
@details: DQN for Stochastic King-actions Windy Gridworld_v3. With wandb sweep.
"""
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

from util.util import PlotUtil
from util.util import RepresentationTools as rpt
from util.wind_greedy_evaluations import DQN_GreedyEvaluation as evaluate
from envs.stoch_king_windy_gridworld_env import StochKingWindyGridWorldEnv

import wandb
wandb.login(key="576d985d69bfd39f567224809a6a3dd329326993")

env = StochKingWindyGridWorldEnv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using device:", device)

num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
grid_dimensions = (env.grid_height, env.grid_width)

# Define the DQN class
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.fci = nn.Linear(state_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fcf = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fci(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fcf(x))
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.cat(states).to(device),
            torch.tensor(actions).to(device),
            torch.tensor(rewards).to(device),
            torch.cat(next_states).to(device),
            torch.tensor(dones).to(device)
        )
    
    def unpack(self):
        batch = self.memory
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.cat(states).to(device),
            torch.tensor(actions).to(device),
            torch.tensor(np.mean(rewards)/(np.std(rewards)+1e-5)).to(device),
            torch.cat(next_states).to(device),
            torch.tensor(dones).to(device)
        )

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_dim, learning_rate, discount_rate, epsilon, epsilon_decay, buffer_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.epsilon_min = 0.01
        self.final_lr = 1e-6
        self.replay_memory = ReplayMemory(self.buffer_size)
        self.batch_size = batch_size
        self.model = DQN(state_size, action_size, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.Q_table = np.empty((env.grid_height, env.grid_width, env.nA))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state).reshape(1, -1), dtype=torch.float32).to(device)
        next_state = torch.tensor(np.array(next_state).reshape(1, -1), dtype=torch.float32).to(device)
        self.replay_memory.push(state, action, reward, next_state, done)

    def replay(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        if len(self.replay_memory) == self.buffer_size:
            for i in range(int(self.buffer_size/self.batch_size)):
        
                random.shuffle(self.replay_memory.memory)
                states, actions, rewards, next_states, dones = self.replay_memory.sample(self.batch_size)
                
        
                current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1).to(device)
                next_q_values = self.model(next_states).max(1)[0].detach()
                target_q_values = rewards + self.discount_rate * next_q_values * (1 - dones.float()).to(device)
                
                loss = self.loss(current_q_values, target_q_values)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.replay_memory.memory.clear()

    def train(self, env, episodes):
        # Initialize the greedy evaluation
        total_reward_per_param = 0
        
        # lr_decay = (self.learning_rate - self.final_lr) / episodes
        for episode in range(episodes):
            state = env.reset()
            percent_completion = (episode+1)/episodes
            
            done = False
            
            # Epsilon annealing - exponential decay
            if self.epsilon > self.epsilon_min:
                # self.epsilon -= self.epsilon_decay
                self.epsilon = math.exp(-2*math.pow(percent_completion,3.5)/0.4)
            
            episode_reward = 0
            while not done:
                action = self.choose_action(state)
                
                next_state, reward, done = env.step(action)
                # next_state = enco.OneHotEncoder(next_state)
                self.replay(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                
            
            total_reward_per_param += episode_reward
            wandb.log({'Reward':episode_reward})
            
            if episode%500 == 0:
                print("Episode: {}/{}, Reward: {}, Epsilon: {:.2f}, Learning Rate: {}".format(episode+1, episodes, episode_reward, self.epsilon, self.learning_rate))
        
        return total_reward_per_param

def train_params(config):
    state_size = 2
    action_size = env.nA
    batch_size = config.batch_size
    buffer_size = config.buffer_size
    num_episodes = 60000
    alpha = config.alpha
    discount_rate = 0.98
    epsilon_start = 1.0
    epsilon_decay = epsilon_start/num_episodes
    hidden_dim = 32
    agent = DQNAgent(state_size, action_size, hidden_dim, alpha, discount_rate,\
                     epsilon_start, epsilon_decay, buffer_size, batch_size)
    
    total_reward_per_param = agent.train(env, num_episodes)
    return total_reward_per_param

def main():
    wandb.init(project="Sweep-DQN-King-Stoch-Windy-GW-2x32")#, mode="disabled")
    total_reward_per_param = train_params(wandb.config)
    wandb.log({'Total_Reward':total_reward_per_param})
    

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "total_reward_per_param"},
    "parameters":{
        "alpha": {"values": [7e-4, 3e-4, 1e-4, 7e-5, 3e-5, 1e-5]},
        "buffer_size": {"values": [512, 1024]},
        "batch_size": {"values": [256, 512]}}}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Sweep-DQN-King-Stoch-Windy-GW-2x32")

wandb.agent(sweep_id, function = main, count=24)
wandb.finish()
    
























