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
import argparse
import wandb

wandb.login(key="576d985d69bfd39f567224809a6a3dd329326993")

import random
import matplotlib

from collections import namedtuple, deque

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from causal_gridworlds.util.util import PlotUtil
# from causal_gridworlds.util.util import RepresentationTools as rpt
from util.wind_greedy_evaluations import A2C_GreedyEvaluation as evaluate
from custom_envs.stoch_windy_gridworld_env_v3 import StochWindyGridWorldEnv_V3
from custom_envs.stoch_king_windy_gridworld_env import StochKingWindyGridWorldEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

is_ipython = 'inline' in matplotlib.get_backend()


# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fci = nn.Linear(state_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fcf = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.fci(state)
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
        x = self.fci(state)
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
    def __init__(self, env, state_dim, action_dim, hidden_dim,
                 lr_actor, lr_critic, buffer_size, batch_size, entropy_weight,
                 wind_distribution, gamma=0.98):
        self.env = env
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.action_dim = action_dim

        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.entropy_weight = entropy_weight
        self.wind_distribution_ok = wind_distribution
        self.update_frequency = 200
        self.steps_since_update = 0

        # Log model gradients and parameters to WandB
        wandb.watch(self.actor, log="all", log_freq=100)  # Log gradients and parameters for the actor
        wandb.watch(self.critic, log="all", log_freq=100)  # Log gradients and parameters for the critic

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action_probs = self.actor(state.unsqueeze(0)).squeeze(0) + 1e-8
        action = torch.multinomial(action_probs, 1).item()
        return action

    "This type of action selection is not valid. Use standard."

    def select_action_wd(self, state):
        state = torch.FloatTensor(state).to(device)
        if self.wind_distribution_ok:
            # Marginalize over wind effects to calculate expected action probabilities
            expected_action_probs = torch.zeros(self.action_dim).to(device)
            # Simulate possible wind effects
            for wind_effect, prob in zip(
                    np.arange(-self.env.range_random_wind, self.env.range_random_wind + 1), self.env.probabilities
            ):
                # Adjust the row (vertical) index based on the wind effect
                row, col = state.cpu().numpy()
                adjusted_row = max(0, min(self.env.grid_height - 1, row - wind_effect))  # Clamp row index
                adjusted_state = torch.FloatTensor([adjusted_row, col]).to(device)

                # Get action probabilities for the adjusted state
                action_probs = self.actor(adjusted_state.unsqueeze(0)).squeeze(0) + 1e-8
                expected_action_probs += prob * action_probs
            # Normalize the expected action probabilities
            expected_action_probs /= expected_action_probs.sum()

            # Sample action from the expected action probabilities
            action = torch.multinomial(expected_action_probs, 1).item()

        else:
            # Use action probabilities for the current state without adjustment
            action_probs = self.actor(state.unsqueeze(0)).squeeze(0) + 1e-8
            action = torch.multinomial(action_probs, 1).item()

        return action

    def train(self, state, action, reward, next_state, done, episode):
        # Add experience to the replay buffer
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.add_experience(experience)
        self.steps_since_update += 1

        # Check if the buffer is filled
        if len(self.replay_buffer.buffer) >= self.buffer_size and self.steps_since_update >= self.update_frequency:
            for _ in range(2):
                # Sample a batch of experience from the replay buffer
                states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)

                # Convert to tensors
                states = torch.FloatTensor(states).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                actions = torch.LongTensor(actions).view(-1, 1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                # Compute critic next values
                if self.wind_distribution_ok:
                    batch_size = states.shape[0]

                    # Initialize next values tensor
                    next_values = torch.zeros(batch_size).to(device)

                    # Convert states and actions to NumPy for environment function compatibility
                    states_np = states.cpu().numpy().astype(int)  # Shape: (batch_size, 2)
                    actions_np = actions.squeeze(-1).cpu().numpy().astype(int)  # Shape: (batch_size,)

                    # Compute realized states by applying actions
                    realized_states = np.array([
                        self.env.action_destination(tuple(state), action)
                        for state, action in zip(states_np, actions_np)
                    ])  # Shape: (batch_size, 2)

                    # Get wind strengths at realized states' columns
                    realized_cols = realized_states[:, 1]
                    wind_strengths = self.env.wind[realized_cols]  # Shape: (batch_size,)

                    # Create masks for samples with and without wind
                    wind_mask = wind_strengths > 0  # Shape: (batch_size,)

                    # Process samples without wind
                    if np.any(~wind_mask):
                        no_wind_indices = np.where(~wind_mask)[0]
                        no_wind_states = realized_states[no_wind_indices]
                        no_wind_states_tensor = torch.FloatTensor(no_wind_states).to(device)
                        next_values_no_wind = self.critic(no_wind_states_tensor).squeeze(-1)
                        next_values[no_wind_indices] = next_values_no_wind

                    # Process samples with wind
                    if np.any(wind_mask):
                        wind_indices = np.where(wind_mask)[0]
                        wind_realized_states = realized_states[wind_indices]

                        # Initialize next values for wind samples
                        next_values_wind = torch.zeros(len(wind_indices)).to(device)

                        # Prepare wind effects and probabilities
                        wind_effects = np.arange(-self.env.range_random_wind, self.env.range_random_wind + 1)
                        probs = np.array(self.env.probabilities)

                        # Expand wind effects and probabilities for broadcasting
                        wind_effects_expanded = wind_effects.reshape(1, -1)  # Shape: (1, num_wind_effects)
                        probs_expanded = probs.reshape(1, -1)  # Shape: (1, num_wind_effects)

                        # Repeat wind_realized_states for each wind effect
                        wind_realized_states_expanded = np.repeat(
                            wind_realized_states[:, np.newaxis, :], len(wind_effects), axis=1
                        )  # Shape: (num_wind_samples, num_wind_effects, 2)

                        # Adjust the row indices based on wind effects
                        adjusted_rows = wind_realized_states_expanded[:, :,
                                        0] - wind_effects_expanded  # Subtract wind effects
                        adjusted_rows = np.clip(adjusted_rows, 0, self.env.grid_height - 1)  # Clamp to grid

                        # Columns remain the same
                        adjusted_cols = wind_realized_states_expanded[:, :, 1]

                        # Combine adjusted rows and columns
                        adjusted_next_states = np.stack((adjusted_rows, adjusted_cols),
                                                        axis=2)  # Shape: (num_wind_samples, num_wind_effects, 2)

                        # Reshape for critic evaluation
                        adjusted_next_states_flat = adjusted_next_states.reshape(-1,
                                                                                 2)  # Shape: (num_wind_samples * num_wind_effects, 2)

                        # Convert to tensor
                        adjusted_next_states_tensor = torch.FloatTensor(adjusted_next_states_flat).to(device)

                        # Evaluate critic on all adjusted next states
                        V_s_primes = self.critic(adjusted_next_states_tensor).squeeze(
                            -1)  # Shape: (num_wind_samples * num_wind_effects,)

                        # Reshape back to (num_wind_samples, num_wind_effects)
                        V_s_primes = V_s_primes.view(len(wind_indices), len(wind_effects))

                        # Compute expected next values
                        expected_V_s_primes = (V_s_primes * torch.FloatTensor(probs_expanded).to(device)).sum(dim=1)

                        # Update next_values for wind samples
                        next_values[wind_indices] = expected_V_s_primes
                else:
                    next_values = self.critic(next_states).squeeze(-1)

                # Calculate advantages
                values = self.critic(states).squeeze(-1)  # Current state values
                delta = rewards + (self.gamma * next_values * (1 - dones)) - values
                advantage = delta.detach()

                # Normalize advantage for numerical stability
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                action_probs = self.actor(states) + 1e-8

                # Compute log probabilities of the taken actions
                log_probs = -torch.log(action_probs.gather(1, actions) + 1e-8)

                # Actor loss
                actor_loss = (log_probs * advantage).mean()

                # Critic loss
                critic_loss = delta.pow(2).mean()

                # Policy entropy for exploration
                entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=1).mean()

                # Total loss with entropy regularization
                total_loss = actor_loss - self.entropy_weight * entropy

                # Update actor network
                self.optimizer_actor.zero_grad()
                total_loss.backward()
                self.optimizer_actor.step()

                # Update critic network
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

            # Reset the update counter
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
    # Debug: Print the config to ensure it contains the expected keys
    batch_size = config["batch_size"]
    buffer_size = config["buffer_size"]
    learning_rate_actor = config["lr_actor"]
    learning_rate_critic = 3e-4
    wind_distribution_ok = config["wind_distribution_ok"]  # Use wind distribution setting
    total_reward_per_param = 0
    entropy_weight = config["entropy_weight"]

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

        while not done:
            # Select an action
            action = agent.select_action_wd(state)
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
    seeds = [42, 123, 456, 789, 101112]
    if single_run:
        for seed in seeds:
            # Perform a single run
            wandb.init(project="A2C-Stoch-GW-Wind_seen", config={
                "lr_actor": 3e-4,
                "buffer_size": 512,
                "batch_size": 512,
                "wind_distribution_ok": True,
                "entropy_weight": 0.2,
                "seed": seed,
            },
                       group="A2C-Multi-Action_select_only",
                       job_type=f"seed-{seed}",
                       mode="disabled",
                       )  # Disable online mode for a single run if needed

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
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="A2C-Stoch-GW-Wind_seen")
        wandb.agent(sweep_id, function=sweep_main)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A2C training.")
    parser.add_argument("--single_run", action="store_true",
                        help="Run a single training instead of a sweep.")
    args = parser.parse_args()

    main(single_run=args.single_run)
