# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:23:19 2023

@author: shubh
@details: A2C for Stochastic Windy Gridworld version 2. Sweep implementation for
hyperparameter search.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import wandb
import argparse

wandb.login(key="576d985d69bfd39f567224809a6a3dd329326993")
import math
import random
import matplotlib
import multiprocessing
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import itertools

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from causal_gridworlds.util.util import PlotUtil
# from causal_gridworlds.util.util import RepresentationTools as rpt
from util.wind_greedy_evaluations import A2CGRU_GreedyEvaluation as evaluate
from custom_envs.stoch_windy_gridworld_env_v3 import StochWindyGridWorldEnv_V3
from custom_envs.stoch_king_windy_gridworld_env import StochKingWindyGridWorldEnv

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

is_ipython = 'inline' in matplotlib.get_backend()


# Define the actor network
class GRUActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers=2):
        super(GRUActor, self).__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)  # Add layer normalization
        self.fc = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, state_seq, hidden=None):
        """
        Args:
            state_seq: (batch_size, seq_length, state_dim)
            hidden: (num_layers, batch_size, hidden_dim), optional hidden states for GRU
        Returns:
            action_probs: (batch_size, seq_length, action_dim), action probabilities
            hidden: (num_layers, batch_size, hidden_dim), hidden states
        """
        gru_out, hidden = self.gru(state_seq, hidden)  # Process sequence through GRU
        gru_out = self.layer_norm(gru_out)  # Normalize GRU output
        logits = self.fc(gru_out)  # Project GRU output to action_dim
        action_probs = F.softmax(logits, dim=-1)  # Compute action probabilities
        return action_probs, hidden

    # Temperature for softmax output scaling.
    # Closer to 10.0 makes the logits uniform
    # Closer to 0.0 makes logits biased to one value
    def temperature_scaled_softmax(self, logits, temperature=5.0):
        logits = logits / temperature
        return torch.softmax(logits, dim=0)


# Define the Critic network
class GRUCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_layers=2):
        super(GRUCritic, self).__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)  # Add layer normalization
        self.fc = nn.Linear(hidden_dim, 1)

        # Initialize weights
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, state_seq, hidden=None):
        """
        Args:
            state_seq: (batch_size, seq_length, state_dim)
            hidden: (num_layers, batch_size, hidden_dim), optional hidden states for GRU
        Returns:
            values: (batch_size, seq_length, 1), state values
            hidden: (num_layers, batch_size, hidden_dim), hidden states
        """
        gru_out, hidden = self.gru(state_seq, hidden)  # Process sequence through GRU
        gru_out = self.layer_norm(gru_out)  # Normalize GRU output
        values = self.fc(gru_out)  # Project GRU output to a scalar value
        return values, hidden


# Define a named tuple for experiences
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, buffer_size, max_sequence_length):
        self.buffer = deque(maxlen=buffer_size)
        self.max_sequence_length = max_sequence_length

    def add_experience(self, trajectory):
        """
                Add a trajectory (sequence of experiences) to the buffer.
                Args:
                    trajectory: List of Experience namedtuples representing a trajectory.
                """
        self.buffer.append(trajectory)

    def sample_batch(self, batch_size):
        """
        Sample a batch of trajectories.
        Args:
            batch_size: Number of trajectories to sample.
        Returns:
            A batch of trajectories.
        """
        batch = random.sample(self.buffer, batch_size)
        return batch


class A2C:
    def __init__(self, env, state_dim, action_dim, hidden_dim,
                 lr_actor, lr_critic, buffer_size, batch_size, entropy_weight,
                 wind_distribution, gamma=0.98, max_sequence_length=50):
        self.env = env
        self.actor = GRUActor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = GRUCritic(state_dim, hidden_dim).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.action_dim = action_dim

        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(buffer_size, max_sequence_length)
        self.batch_size = batch_size
        self.entropy_weight = entropy_weight
        self.wind_distribution_ok = wind_distribution
        self.update_frequency = 200
        self.steps_since_update = 0

        # Add parameters for GRU hidden states
        self.hidden_actor = None
        self.hidden_critic = None

    def select_action(self, state, hidden_actor=None):
        """
        Selects an action using the GRU actor.
        Args:
            state: Current state (single step).
            hidden_actor: Hidden state from the previous timestep.
        Returns:
            action: Selected action.
            hidden_actor: Updated hidden state for the GRU.
        """
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, state_dim)
        action_probs, hidden_actor = self.actor(state, hidden_actor)  # Forward pass through GRU
        action = torch.multinomial(action_probs.squeeze(0), 1).item()
        return action, hidden_actor

    def train(self, state, action, reward, next_state, done, episode):
        # Add experience to the replay buffer
        trajectory = [Experience(state, action, reward, next_state, done)]
        self.replay_buffer.add_experience(trajectory)
        self.steps_since_update += 1
        # Check if the buffer is filled
        if len(self.replay_buffer.buffer) >= self.buffer_size and self.steps_since_update >= self.update_frequency:
            for _ in range(int()):
                # Sample a batch of trajectories
                batch = self.replay_buffer.sample_batch(self.batch_size)

                # Convert trajectories to tensors
                states, actions, rewards, next_states, dones = [], [], [], [], []
                for trajectory in batch:
                    states.append([exp.state for exp in trajectory])
                    actions.append([exp.action for exp in trajectory])
                    rewards.append([exp.reward for exp in trajectory])
                    next_states.append([exp.next_state for exp in trajectory])
                    dones.append([exp.done for exp in trajectory])

                # Prepare tensors
                states = torch.FloatTensor(states).to(device)  # (batch_size, seq_length, state_dim)
                actions = torch.LongTensor(actions).to(device)  # (batch_size, seq_length)
                rewards = torch.FloatTensor(rewards).to(device)  # (batch_size, seq_length)
                next_states = torch.FloatTensor(next_states).to(device)  # (batch_size, seq_length, state_dim)
                dones = torch.FloatTensor(dones).to(device)  # (batch_size, seq_length)

                # Initialize hidden states
                hidden_critic = None

                # Compute values
                values, hidden_critic = self.critic(states, hidden_critic)
                next_values, _ = self.critic(next_states, hidden_critic)

                if self.wind_distribution_ok and episode >= 1000:
                    # Marginalize over wind effects
                    # Remove sequence length dimension (512, 1, 2) -> (512, 2)
                    next_states_flat = next_states.squeeze(1).cpu().numpy()
                    expected_next_values = []
                    for wind_effect, prob in zip(np.arange(-self.env.range_random_wind,
                                                           self.env.range_random_wind + 1),
                        self.env.probabilities):
                        adjusted_next_states = [
                            self.env.clamp_to_grid((s[0] - wind_effect, s[1]))
                            for s in next_states_flat
                        ]
                        adjusted_next_states_tensor = torch.FloatTensor(adjusted_next_states).unsqueeze(1).to(device)
                        # Create a new hidden state for the adjusted states
                        hidden_critic_adjusted = torch.zeros(self.critic.gru.num_layers, len(adjusted_next_states),
                                                             self.critic.gru.hidden_size).to(device)
                        value, _ = self.critic(adjusted_next_states_tensor, hidden_critic_adjusted)

                        expected_next_values.append(prob * value)
                    next_values = torch.stack(expected_next_values).sum(dim=0)

                else:
                    # Use only the observed next state values
                    next_values, _ = self.critic(next_states, hidden_critic)

                # Compute advantages
                delta = rewards + self.gamma * next_values * (1 - dones) - values
                advantages = delta.detach()  # Detach advantages to prevent graph reuse
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Actor update
                action_probs, _ = self.actor(states)
                log_probs = torch.log(action_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)
                actor_loss = -(log_probs * advantages).mean()

                # Critic update
                critic_loss = delta.pow(2).mean()

                # Optimize actor and critic
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

            self.steps_since_update = 0


def train_params(config):
    # Define environment dimensions and action space
    env = StochKingWindyGridWorldEnv()
    grid_dimensions = (env.grid_height, env.grid_width)
    env.seed(42)
    state_dim = 2
    action_dim = env.nA
    hidden_dim = 128
    num_episodes = 15000
    # Debug: Print the config to ensure it contains the expected keys
    batch_size = config["batch_size"]
    buffer_size = config["buffer_size"]
    learning_rate_actor = config["lr_actor"]
    learning_rate_critic = 3e-4
    wind_distribution_ok = config["wind_distribution_ok"]  # Use wind distribution setting
    total_reward_per_param = 0
    entropy_weight = config["entropy_weight"]
    # Initialize hidden states for GRU


    # Create the A2C agent
    agent = A2C(env, state_dim, action_dim, hidden_dim, learning_rate_actor,
                learning_rate_critic, buffer_size, batch_size, entropy_weight,
                wind_distribution_ok)
    # Initialize the greedy evaluation
    greedy_evaluation = evaluate(env, grid_dimensions, device)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        hidden_actor = None
        hidden_critic = None

        while not done:
            # Select an action
            action, hidden_actor = agent.select_action(state, hidden_actor)

            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)

            # Train the agent
            agent.train(state, action, reward, next_state, done, episode)
            episode_reward += reward
            state = next_state

        # Log reward for each episode
        wandb.log({"Reward": episode_reward})

        # Periodic logging
        if episode % 500 == 0:
            print(f"Episode: {episode + 1}/{num_episodes}, Reward: {episode_reward}")
            # Runs greedy evaluation
            greedy_rewards, avg_evaluation_reward = greedy_evaluation.evaluate(agent.actor)
            wandb.log({"Avg. Evaluation Reward": avg_evaluation_reward})

        # wandb.log({"Learning rate": current_lr_actor})

        total_reward_per_param += episode_reward

    return total_reward_per_param


def main(single_run=False):
    if single_run:
        # Perform a single run
        wandb.init(project="A2C-King-Stoch-GW-Wind_seen", config={
            "lr_actor": 1e-4,
            "buffer_size": 1024,
            "batch_size": 512,
            "wind_distribution_ok": True,
            "entropy_weight": 0.1,
        }, mode="online")  # Disable online mode for a single run if needed

        config = wandb.config
        total_reward_per_param = train_params(config)
        wandb.log({"Total_Reward": total_reward_per_param})
        wandb.finish()
    else:
        # Sweep logic
        def sweep_main():
            wandb.init()
            config = wandb.config  # Automatically set for each sweep run
            total_reward_per_param = train_params(config)
            wandb.log({"Total_Reward": total_reward_per_param})
            wandb.finish()

        # Perform a sweep
        sweep_configuration = {
            "name": "GRUSweep",
            "method": "grid",  # Exhaustive search over all parameter combinations
            "metric": {"goal": "maximize", "name": "Total_Reward"},  # Optimize for total reward
            "parameters": {
                "lr_actor": {"values": [1e-4, 3e-4]},
                "buffer_size": {"values": [1024, 512]},
                "batch_size": {"values": [512, 256]},
                "wind_distribution_ok": {"values": [False, True]},
                "entropy_weight": {"values": [0.2, 0.1]},
            },
        }

        # Initialize the sweep
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="A2C-King-Stoch-GW-Wind_seen")
        wandb.agent(sweep_id, function=sweep_main)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A2C training.")
    parser.add_argument("--single_run", action="store_true",
                        help="Run a single training instead of a sweep.")
    args = parser.parse_args()

    main(single_run=args.single_run)
