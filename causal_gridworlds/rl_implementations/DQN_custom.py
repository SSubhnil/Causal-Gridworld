# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:23:19 2023

@author: OpenAI Assistant
@details: DQN for Stochastic Windy Gridworld. Sweep implementation for hyperparameter search.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import wandb

wandb.login(key="576d985d69bfd39f567224809a6a3dd329326993")  # Replace with your actual WandB API key

import random
import matplotlib

from collections import namedtuple, deque

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.wind_greedy_evaluations import DQN_GreedyEvaluation as evaluate
from custom_envs.stoch_windy_gridworld_env_v3 import StochWindyGridWorldEnv_V3
from custom_envs.stoch_king_windy_gridworld_env import StochKingWindyGridWorldEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

is_ipython = 'inline' in matplotlib.get_backend()


# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fci = nn.Linear(state_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fcf = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.fci(state)
        x = F.relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        q_values = self.fcf(x)
        return q_values


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


class DQNAgent:
    def __init__(self, env, state_dim, action_dim, hidden_dim, lr, buffer_size,
                 batch_size, wind_distribution_ok, gamma=0.98):
        self.env = env
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.action_dim = action_dim

        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.wind_distribution_ok = wind_distribution_ok
        self.update_frequency = 200
        self.steps_since_update = 0

        # For epsilon-greedy policy
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Log model gradients and parameters to WandB
        wandb.watch(self.q_network, log="all", log_freq=100)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            # Random action
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                q_values = self.q_network(state.unsqueeze(0)).squeeze(0)
                action = torch.argmax(q_values).item()
            return action

    def select_action_wd(self, state):
        if np.random.rand() < self.epsilon:
            # Random action
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).to(device)
            if self.wind_distribution_ok:
                # Marginalize over wind effects to calculate expected Q-values
                expected_q_values = torch.zeros(self.action_dim).to(device)
                # Simulate possible wind effects
                for wind_effect, prob in zip(
                        np.arange(-self.env.range_random_wind, self.env.range_random_wind + 1), self.env.probabilities
                ):
                    # Adjust the row (vertical) index based on the wind effect
                    row, col = state.cpu().numpy()
                    adjusted_row = max(0, min(self.env.grid_height - 1, row - wind_effect))  # Clamp row index
                    adjusted_state = torch.FloatTensor([adjusted_row, col]).to(device)

                    # Get Q-values for the adjusted state
                    q_values = self.q_network(adjusted_state.unsqueeze(0)).squeeze(0)
                    expected_q_values += prob * q_values
                # Choose the action with the highest expected Q-value
                action = torch.argmax(expected_q_values).item()
            else:
                # Use Q-values for the current state without adjustment
                with torch.no_grad():
                    q_values = self.q_network(state.unsqueeze(0)).squeeze(0)
                    action = torch.argmax(q_values).item()
            return action

    def train(self, state, action, reward, next_state, done):
        # Add experience to the replay buffer
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.add_experience(experience)
        self.steps_since_update += 1

        # Update the network if enough experiences are collected
        if len(self.replay_buffer.buffer) >= self.batch_size:
            # Sample a batch of experiences
            states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)

            # Convert to tensors
            states = torch.FloatTensor(states).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            actions = torch.LongTensor(actions).view(-1, 1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            # Compute Q-values of current states
            q_values = self.q_network(states).gather(1, actions)

            # Compute target Q-values
            with torch.no_grad():
                if self.wind_distribution_ok:
                    # Vectorized wind effect processing
                    wind_effects = torch.arange(-self.env.range_random_wind, self.env.range_random_wind + 1).to(device)
                    wind_probs = torch.FloatTensor(self.env.probabilities).to(device)

                    # Expand next_states for each wind effect
                    expanded_next_states = next_states.unsqueeze(1).repeat(1, len(wind_effects), 1)

                    # Apply wind effects to the vertical (row) dimension
                    expanded_next_states[:, :, 0] = torch.clamp(
                        expanded_next_states[:, :, 0] - wind_effects.view(1, -1), 0, self.env.grid_height - 1
                    )

                    # Flatten expanded_next_states for batch processing
                    flattened_next_states = expanded_next_states.view(-1, next_states.shape[1])

                    # Compute Q-values for all adjusted next states
                    all_next_q_values = self.target_network(flattened_next_states)

                    # Reshape to [batch_size, num_wind_effects, action_dim]
                    all_next_q_values = all_next_q_values.view(next_states.shape[0], len(wind_effects), -1)

                    # Maximize over actions for each wind effect
                    max_next_q_values = all_next_q_values.max(2)[0]  # Shape: [batch_size, num_wind_effects]

                    # Marginalize over wind effects using probabilities
                    expected_next_q_values = (max_next_q_values * wind_probs).sum(dim=1)  # Shape: [batch_size]

                    # Compute target Q-values
                    target_q_values = rewards.squeeze() + self.gamma * expected_next_q_values * (1 - dones.squeeze())
                    target_q_values = target_q_values.unsqueeze(1)
                else:
                    # Use next_states as is
                    next_q_values = self.target_network(next_states)
                    max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
                    target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

            # Compute loss (MSE)
            loss = F.mse_loss(q_values, target_q_values)

            # Update the network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Update the target network
            if self.steps_since_update >= self.update_frequency:
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.steps_since_update = 0


def train_params(config):
    # Define environment dimensions and action space
    env = StochWindyGridWorldEnv_V3()
    grid_dimensions = (env.grid_height, env.grid_width)
    env.seed(config["seed"])
    state_dim = 2
    action_dim = env.nA
    hidden_dim = 128
    num_episodes = 15000
    batch_size = config["batch_size"]
    buffer_size = config["buffer_size"]
    learning_rate = config["lr"]
    wind_distribution_ok = config["wind_distribution_ok"]  # Use wind distribution setting
    total_reward_per_param = 0

    # Create the DQN agent
    agent = DQNAgent(env, state_dim, action_dim, hidden_dim, learning_rate,
                     buffer_size, batch_size, wind_distribution_ok)
    # Initialize the greedy evaluation
    greedy_evaluation = evaluate(env, grid_dimensions, device)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Select an action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Train the agent
            agent.train(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

        # Log reward for each episode
        wandb.log({"Reward": episode_reward})

        # Periodic logging
        if episode % 500 == 0:
            print(f"Episode: {episode + 1}/{num_episodes}, Reward: {episode_reward}")
            # Runs greedy evaluation
            greedy_rewards, avg_evaluation_reward,_ = greedy_evaluation.run_algo(agent.q_network)
            wandb.log({"Avg. Evaluation Reward": avg_evaluation_reward})

        total_reward_per_param += episode_reward

    return total_reward_per_param


def main(single_run=False):
    seeds = [42, 123, 456, 789, 101112]
    if single_run:
        for seed in seeds:
            # Perform a single run
            wandb.init(project="DQN-Stoch-GW-Wind_seen", config={
                "lr": 1e-4,
                "buffer_size": 512,
                "batch_size": 512,
                "wind_distribution_ok": True,
                "epsilon_decay": 0.995,
                "seed": seed,
            },
                       group="DQN-Multi-Action_select_only",
                       job_type=f"seed-{seed}",
                       mode="disabled",
                       )  # Set mode to "online" for actual runs

            # Set the seeds
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            wandb.log({"Seed": seed})  # Log the seed for reproducibility

            config = wandb.config
            total_reward_per_param = train_params(config)
            wandb.log({"Total_Reward": total_reward_per_param})
            wandb.finish()
    else:
        # Sweep logic
        def sweep_main():
            wandb.init()
            config = wandb.config  # Automatically set for each sweep run
            seed = np.random.randint(1e6)  # Random seed for each sweep run
            wandb.config.update({"seed": seed}, allow_val_change=True)

            # Set the seeds
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            wandb.log({"Seed": seed})

            total_reward_per_param = train_params(config)
            wandb.log({"Total_Reward": total_reward_per_param})
            wandb.finish()

        # Perform a sweep
        sweep_configuration = {
            "name": "DQNSweep",
            "method": "grid",  # Exhaustive search over all parameter combinations
            "metric": {"goal": "maximize", "name": "Total_Reward"},  # Optimize for total reward
            "parameters": {
                "lr": {"values": [3e-4, 1e-4]},
                "buffer_size": {"values": [1024, 512]},
                "batch_size": {"values": [512, 256]},
                "wind_distribution_ok": {"values": [False, True]},
                "epsilon_decay": {"values": [0.995]},
            },
        }

        # Initialize the sweep
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="DQN-Stoch-GW-Wind_seen")
        wandb.agent(sweep_id, function=sweep_main)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DQN training.")
    parser.add_argument("--single_run", action="store_true",
                        help="Run a single training instead of a sweep.")
    args = parser.parse_args()

    main(single_run=args.single_run)
