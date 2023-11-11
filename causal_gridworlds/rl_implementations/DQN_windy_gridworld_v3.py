# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:47:07 2023


@author: SSubhnil
@details: DQN code for Windy GridWorld_v3 - For sweep and debugging with WandB
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
from util.static_wind_greedy_evaluations import DQN_GreedyEvaluation as evaluate
from envs.windy_gridworld_env import WindyGridWorldEnv

import wandb

wandb.login(key="576d985d69bfd39f567224809a6a3dd329326993")

wandb.init(project="DQN-4A-Windy-GW-2x64", mode="disabled")

# wandb.init(
#     project="DQN-4A-Windy-GW", mode='disabled')

env = WindyGridWorldEnv()
enco = rpt(env.observation_space) # Import OneHotEncoder for state representation
np.random.seed(42)
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
        state_size = state_size
        action_size = action_size
        hidden_dim = hidden_size
        self.fci = nn.Linear(state_size, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc34 = nn.Linear(hidden_dim, hidden_dim)
        self.fcf = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = torch.relu(self.fci(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fcf(x))
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
        return states, actions, rewards, next_states, dones
    
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
        self.epsilon_min = 0.01
        self.final_lr = 1e-5
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(buffer_size)
        self.model = DQN(state_size, action_size, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.Q_table = np.empty((env.grid_height, env.grid_width, env.nA))
        self.total_reward_per_param = 0

    def choose_greedy_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        q_values = self.policy_net(state)
        return torch.argmax(q_values).item()

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state).reshape(1, -1), dtype=torch.float32).to(device)
        print("state", state)
        next_state = torch.tensor(np.array(next_state).reshape(1, -1), dtype=torch.float32).to(device)
        self.replay_memory.push(state, action, reward, next_state, done)

    def replay(self, percent_completion):
        if len(self.replay_memory) == self.buffer_size:
            states, actions, rewards, next_states, dones = self.replay_memory.sample(self.batch_size)

            states = torch.FloatTensor(states).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            actions = torch.FloatTensor(actions).to(device=device, dtype=int)
            rewards = torch.FloatTensor(rewards).to(device)  #
            dones = torch.FloatTensor(dones).to(device)  # shape = [512]

            current_q_values = (self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)).to(device)
            next_q_values = self.target_net(next_states).max(1)[0].detach()
            target_q_values = (rewards + (self.discount_rate * next_q_values * (1 - dones.float()))).to(device)

            loss = self.loss(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, episodes):
        "Fill the buffer with random exploration"
        done = False
        state = env.reset()
        while len(self.replay_memory) < self.buffer_size:
            if not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
            else:
                done = False
                state = env.reset()
        "Training Loop"
        for episode in range(episodes):

            percent_completion = (episode + 1) / episodes

            state = env.reset()
            done = False

            "Epsilon annealing - exponential decay"
            if self.epsilon > self.epsilon_min:
                # self.epsilon -= self.epsilon_decay
                self.epsilon = math.exp(-2 * math.pow(percent_completion, 3.5) / 0.4)

            episode_reward = 0
            greedy_episode_reward = 0

            "Greedy Evaluation + Target network soft-update"
            if episode % self.greedy_interval == 0:
                "Soft update of the target network's weights"
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                            1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                "Greedy evaluation"
                state = env.reset()
                while not done:
                    action = self.choose_greedy_action(state)

                    next_state, reward, done, _ = env.step(action)

                    greedy_episode_reward += reward
                    state = next_state

                wandb.log({'Evaluation reward': greedy_episode_reward})
                print(
                    "Episode: {}/{}, Eval. Reward: {}, Epsilon: {:.2f}, Learning Rate: {}".format(episode + 1, episodes,
                                                                                                  greedy_episode_reward,
                                                                                                  self.epsilon,
                                                                                                  self.learning_rate))
            state = env.reset()
            "Policy net optimization"
            while not done:
                action = self.choose_action(state)

                next_state, reward, done, _ = env.step(action)

                self.remember(state, action, reward, next_state, done)

                episode_reward += reward
                state = next_state
                self.replay(percent_completion)

            wandb.log({'Reward': episode_reward, 'Epsilon': self.epsilon, \
                       'Learning rate': self.learning_rate})

            self.total_reward_per_param += episode_reward
        return self.total_reward_per_param


state_size = 2
action_size = env.nA
batch_size = 512
buffer_size = 10000
num_episodes = 40000
alpha = 1e-4
discount_rate = 0.98
greedy_interval = 500
epsilon_start = 0.95
epsilon_decay = epsilon_start/num_episodes
hidden_dim = 64


    
agent = DQNAgent(state_size, action_size, hidden_dim, alpha, discount_rate,\
                 epsilon_start, epsilon_decay, buffer_size, batch_size)

total_reward_per_param = agent.train(num_episodes)

wandb.finish()