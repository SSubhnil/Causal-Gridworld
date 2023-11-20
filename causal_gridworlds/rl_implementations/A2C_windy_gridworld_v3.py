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

from custom_envs.windy_gridworld_env import WindyGridWorldEnv

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
        self.buffer = list()
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
        state = torch.FloatTensor(state).to(device)
        action_probs = self.actor(state)
        action = torch.multinomial(action_probs, 1).item()
        # print(q_values)
        # self.Q_table[enco.OneHotDecoder(state)] = q_values # q_value requires grad(). That's why use detach()
        return action

    def train(self, state, action, reward, next_state, done):
        # Add experience to the replay buffer
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.add_experience(experience)
        
        # Check if the buffer is filled
        if len(self.replay_buffer.buffer) == self.buffer_size:
            # Sample a batch of experience from the replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)

            states = torch.FloatTensor(states).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            actions = torch.FloatTensor(actions).unsqueeze(1).to(device) # added unsqueeze after removing view(-1, 1) from actor_loss
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device) #
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device) #  shape = [512]

            #Calculate advantages
            values = self.critic(states) # shape = [512, 1]
            next_values = self.critic(next_states) # shape = [512, 1]
            delta = rewards + (self.gamma * next_values * (1 - dones)) - values
            advantage = delta.detach()

            # Actor loss
            action_probs = self.actor(states)
            actions = actions.view(-1, 1).long() # Convert actions into int64
            log_probs = -torch.log(action_probs.gather(1, actions))
            actor_loss = log_probs * advantage#.view(-1, 1)
            actor_loss = actor_loss.mean()

            # Critic loss
            critic_loss = delta.pow(2).mean()

            # Calculate policy entropy
            entropy = -torch.sum(action_probs * torch.log(action_probs), dim=1).mean()

            # Introducing total_loss instead of just actor_loss
            total_loss = actor_loss - self.entropy_weight * entropy

            # Train network on total_loss instead of actor_loss
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            total_loss.backward()
            self.optimizer_actor.step()
            critic_loss.backward()
            self.optimizer_critic.step()

            # Clear the replay buffer after updating
            # self.replay_buffer.buffer.clear()

def train_params(config):
    # Define environment dimensions and action space
    state_dim = 2
    action_dim = env.nA
    hidden_dim = 32
    num_episodes = 20000
    learning_rate_actor = config.lr_actor
    learning_rate_critic = 7e-4
    batch_size = config.batch_size
    buffer_size = 10000
    entropy_weight = 0.1 # Low:[<0.001]; Moderate:[0.01, 0.1]; High:[>1.0]
    total_reward_per_param = 0
           
    # Create the A2C agent
    agent = A2C(state_dim, action_dim, hidden_dim, learning_rate_actor, learning_rate_critic, buffer_size, batch_size, entropy_weight)
    
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
            agent.train(state, action, reward, next_state, done)
            state = next_state
        else:
            done = False
            state = env.reset()

    for episode in range(num_episodes):
        # Greedy evaluation
        # if episode+1 % greedy_interval == 0:
        #     greedy_step_count[sampling_counter], avg_greedy_step_count[sampling_counter]\
        #         = greedy_evaluation.run_algo(self.learning_rate, self.model)
        #     sampling_counter += 1
        
        state = env.reset()
        # state = env.reset()
        done = False
        
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.train(state, action, reward, next_state, done)
            episode_reward += reward        
            state = next_state

        if episode % 500 == 0:
            print("Episode: {}/{}, Reward: {}".format(episode+1, num_episodes, episode_reward))
        
        wandb.log({'Reward':episode_reward})
    
        total_reward_per_param += episode_reward
    return total_reward_per_param

def main():
    wandb.init(project="Sweep-A2C-4A-Windy-GW", mode="disabled")
    total_reward_per_param = train_params(wandb.config)
    wandb.log({'Total_Reward':total_reward_per_param})
    

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "total_reward_per_param"},
    "parameters":{
        "lr_actor": {"values": [1e-3, 7e-4, 3e-4, 1e-4]},
        "batch_size": {"values": [256, 512]}}}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Sweep-A2C-4A-Windy-GW")

wandb.agent(sweep_id, function = main, count=8)
wandb.finish()
    