# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:17:31 2023

@author: shubh
@details: A hyperparameter sweep code for A2C Windy GridWorld
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


import random

from collections import namedtuple, deque

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

from causal_gridworlds.custom_envs.windy_gridworld_env import WindyGridWorldEnv

env = WindyGridWorldEnv()
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

import wandb
wandb.login(key="576d985d69bfd39f567224809a6a3dd329326993")

grid_dimensions = (env.grid_height, env.grid_width)

# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fci = nn.Linear(state_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fcf = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fci(state))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fcf(x), dim=-1)
        return x
    
    # Temperature for softmax output scaling.
    # Closer to 10.0 makes the logits uniform
    # Closer to 0.0 makes logits biased to one value
    def temperature_scaled_softmax(self, logits, temperature = 5.0):
        logits = logits / temperature
        return torch.softmax(logits, dim=0)
    
# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fci = nn.Linear(state_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fcf = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fci(state))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fcf(x)
        return value

# Define a named tuple for experiences
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
        self.position = 0
    
    def add_experience(self, experience):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.buffer_size

    
    def sample_batch(self, batch_size):
     
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
class A2C:
    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, buffer_size, batch_size, entropy_weight, gamma=0.98):
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.current_lr_actor = lr_actor
        self.current_lr_critic = lr_critic
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.action_dim = action_dim
        self.entropy_weight = entropy_weight
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs = self.actor(state)
        # Debugging statements
        # print(f"State: {state}")
        # print(f"Action Probabilities: {action_probs}")
        if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
            # print(f"NaN or Inf in action_probs: {action_probs}")
            action_probs = torch.ones_like(action_probs) / action_probs.size(-1)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def train(self):

        if len(self.replay_buffer) < self.batch_size:
            print("buffer not full!")
            return
        # Add experience to the replay buffer

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)

        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        values = self.critic(states)
        next_values = self.critic(next_states)
        delta = rewards + self.gamma * next_values * (1 - dones) - values
        advantage = delta.detach()

        # Actor loss
        action_probs = self.actor(states)
        log_probs = torch.log(action_probs.gather(1, actions))
        actor_loss = -log_probs * advantage
        actor_loss = actor_loss.mean()

        # Critic loss
        critic_loss = delta.pow(2).mean()

        # Policy entropy
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1).mean()

        # Total loss
        total_loss = actor_loss - self.entropy_weight * entropy

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        total_loss.backward()
        critic_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

def train_params():
    # Define environment dimensions and action space
    state_dim = 2  # Dimension of the state
    action_dim = env.nA  # Number of actions
    hidden_dim = 64
    num_episodes = 500
    lr_actor = 1e-4
    lr_critic = 3e-4
    batch_size = 512
    buffer_size = 10000
    entropy_weight = 0.1 # Low:[<0.001]; Moderate:[0.01, 0.1]; High:[>1.0]
    total_reward_per_param = 0
           
    # Create the A2C agent
    agent = A2C(state_dim, action_dim, hidden_dim, lr_actor, lr_critic, buffer_size, batch_size, entropy_weight)
    
    # wandb.watch(agent.actor, log='gradients', log_freq = 500, idx = 1, log_graph = True)
    # wandb.watch(agent.critic, log='gradients', log_freq = 500, idx = 2, log_graph = True)
    
    # Training loop (replace this with environment and data)

    "Fill the buffer with random exploration"
    done = False
    state = env.reset()
    while len(agent.replay_buffer.buffer) < buffer_size:
        if not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add_experience((state, action, reward, next_state, done))
            state = next_state

        else:
            done = False
            state = env.reset()

    total_reward_per_param = 0

    for episode in range(num_episodes):
        # Greedy evaluation
        # if episode+1 % greedy_interval == 0:
        #     greedy_step_count[sampling_counter], avg_greedy_step_count[sampling_counter]\
        #         = greedy_evaluation.run_algo(self.learning_rate, self.model)
        #     sampling_counter += 1
        
        state = env.reset()
        done = False
        
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add_experience((state, action, reward, next_state, done))
            agent.train()
            episode_reward += reward        
            state = next_state

        if episode % 100 == 0:
            print("Episode: {}/{}, Reward: {}".format(episode+1, num_episodes, episode_reward))
        
        wandb.log({'Reward':episode_reward})
        total_reward_per_param += episode_reward

    return total_reward_per_param

def main():
    wandb.init(project="A2C-4A-Windy-GW", mode="offline")
    total_reward_per_param = train_params()
    wandb.log({'Total_Reward': total_reward_per_param})
    wandb.finish()

if __name__ == "__main__":
    main()