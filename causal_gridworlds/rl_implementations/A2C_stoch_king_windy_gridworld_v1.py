# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:17:31 2023

@author: shubh
@details: A hyperparameter sweep code for A2C King Windy GridWorld
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

from custom_envs.stoch_king_windy_gridworld_env import StochKingWindyGridWorldEnv

env = StochKingWindyGridWorldEnv()
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

import wandb

wandb.login(key="576d985d69bfd39f567224809a6a3dd329326993")
wandb.init(project="A2C-Stoch-King-Windy-GW", mode="offline")

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
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fcf(x), dim=-1)
        return x

    # Temperature for softmax output scaling.
    # Closer to 10.0 makes the logits uniform
    # Closer to 0.0 makes logits biased to one value
    def temperature_scaled_softmax(self, logits, temperature=5.0):
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
        x = F.leaky_relu(self.fc2(x))
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
    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, buffer_size, batch_size, entropy_weight,
                 wind_distribution, gamma=0.98, update_frequency=50):
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.action_dim = action_dim
        self.entropy_weight = entropy_weight
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.steps_since_update = 0
        self.wind_distribution_ok = wind_distribution

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        logits = self.actor(state)
        action_probs = F.softmax(logits, dim=-1) + 1e-8
        action = torch.multinomial(action_probs, 1).item()
        return action

    def train(self, state, action, reward, next_state, done):
        # Add experience to the replay buffer
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.add_experience(experience)

        # Update only when the buffer is full and update frequency is met
        if len(self.replay_buffer.buffer) == self.buffer_size and self.steps_since_update >= self.update_frequency:
            # Iterate through the buffer in batches
            for _ in range(int(self.buffer_size / self.batch_size)):
                # Sample a batch of experience from the replay buffer
                states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)

                # Convert to tensors
                states = torch.FloatTensor(states).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                actions = torch.LongTensor(actions).view(-1, 1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                # Calculate next values based on `wind_distribution_ok`
                if self.wind_distribution_ok:
                    # Marginalize over wind effects to calculate the expected next state value
                    expected_next_values = []
                    for wind_effect, prob in zip(
                            np.arange(-env.range_random_wind, env.range_random_wind + 1), env.probablities
                    ):
                        # Simulate next states for each wind effect
                        next_state_wind = [
                            env.clamp_to_grid((state[0] - wind_effect, state[1]))
                            for state in next_states.cpu().numpy()
                        ]
                        next_state_wind = torch.FloatTensor(next_state_wind).to(device)

                        # Weight the value estimates by their probabilities
                        next_value = prob * self.critic(next_state_wind)
                        expected_next_values.append(next_value)

                    # Sum over all wind effects for the expected value
                    next_values = torch.stack(expected_next_values).sum(dim=0)
                else:
                    # Use only the observed next state
                    next_values = self.critic(next_states)

                # Calculate advantages
                values = self.critic(states)  # Current state values
                delta = rewards + (self.gamma * next_values * (1 - dones)) - values
                advantage = delta.detach()

                # Normalize advantage for numerical stability
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                # Actor loss
                action_probs = self.actor(states)
                log_probs = -torch.log(action_probs.gather(1, actions) + 1e-8)  # Stability with +1e-8
                actor_loss = (log_probs * advantage).mean()

                # Critic loss
                critic_loss = delta.pow(2).mean()

                # Policy entropy for exploration
                entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=1).mean()

                # Total loss with entropy regularization
                total_loss = actor_loss - self.entropy_weight * entropy

                # Backpropagation
                self.optimizer_actor.zero_grad()
                total_loss.backward()
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

            # Reset update counter and clear the buffer
            self.steps_since_update = 0
            self.replay_buffer.buffer.clear()

        # Increment the update counter
        self.steps_since_update += 1


def train_params(config):
    # Extract hyperparameters from the config
    state_dim = 2
    action_dim = env.nA
    hidden_dim = 64
    num_episodes = 40000
    learning_rate_actor = 1e-4
    learning_rate_critic = 1e-4
    batch_size = config["batch_size"]
    buffer_size = config["buffer_size"]
    wind_distribution_ok = config["wind_distribution_ok"]  # Use wind distribution setting
    entropy_weight = 0.1

    # Create the A2C agent
    agent = A2C(
        state_dim, action_dim, hidden_dim, learning_rate_actor, learning_rate_critic,
        buffer_size, batch_size, entropy_weight
    )

    # Initialize replay buffer with random exploration
    done = False
    state = env.reset()
    while len(agent.replay_buffer.buffer) < buffer_size:
        if not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.train(state, action, reward, next_state, done)
            state = next_state
        else:
            done = False
            state = env.reset()

    total_reward_per_param = 0

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.train(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

        # Log rewards to WandB
        wandb.log({"Reward": episode_reward})


        # Print progress periodically
        if episode % 500 == 0:
            print(f"Episode: {episode + 1}/{num_episodes}, Reward: {episode_reward}")

        total_reward_per_param += episode_reward

    return total_reward_per_param

def main():
    # Initialize WandB for this run
    wandb.init(project="A2C-Stoch-Windy-GW", config=wandb.config)

    # Call train_params with the current configuration
    total_reward_per_param = train_params(wandb.config)

    # Log the final result to WandB
    wandb.log({"Total_Reward": total_reward_per_param})

    # Finish the WandB run
    wandb.finish()

if __name__ == "__main__":
    import wandb

    # Define the sweep configuration
    sweep_config = {
        "method": "grid",  # Exhaustive search
        "metric": {"goal": "maximize", "name": "Total_Reward"},
        "parameters": {
            "batch_size": {"values": [256, 512]},
            "buffer_size": {"values": [1024, 2048]},
            "wind_distribution_ok": {"values": [True, False]},
        },
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project="A2C-Stoch-Windy-GW")

    # Run the sweep
    wandb.agent(sweep_id, function=main)

