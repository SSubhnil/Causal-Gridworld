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
                    next_values = torch.zeros(next_states.shape[0]).to(device)

                    for wind_effect, prob in zip(
                            np.arange(-self.env.range_random_wind, self.env.range_random_wind + 1),
                            self.env.probabilities
                    ):
                        print("wind_effect:", wind_effect)
                        print("Probability:", prob)
                        # Adjust the row (vertical index) of next_states for the wind effect
                        adjusted_next_states = next_states.clone()
                        adjusted_next_states[:, 0] = torch.clamp(
                            adjusted_next_states[:, 0] - wind_effect, 0, self.env.grid_height - 1
                        )
                        print("next_states:", next_states.shape)
                        print("adjusted_state:", adjusted_next_states.shape)
                        # Accumulate weighted value estimates for the critic
                        next_values += prob * self.critic(adjusted_next_states).squeeze(-1)
                else:
                    next_values = self.critic(next_states).squeeze(-1)

                # Calculate advantages
                values = self.critic(states).squeeze(-1)  # Current state values
                delta = rewards + (self.gamma * next_values * (1 - dones)) - values
                advantage = delta.detach()

                # Normalize advantage for numerical stability
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                # Compute actor loss using wind-adjusted probabilities
                # Compute actor action probabilities
                # if self.wind_distribution_ok and episode >= 500:
                #     wind_effects = torch.arange(-self.env.range_random_wind, self.env.range_random_wind + 1).to(device)
                #     probs = torch.FloatTensor(self.env.probabilities).to(device)
                #
                #     # Compute adjusted states
                #     adjusted_states = states.unsqueeze(1).repeat(1, len(wind_effects),
                #                                                  1)  # (batch_size, num_wind_effects, 2)
                #     adjusted_states[:, :, 0] = torch.clamp(
                #         adjusted_states[:, :, 0] - wind_effects.view(1, -1), 0, self.env.grid_height - 1
                #     )
                #
                #     # Compute action probabilities for all adjusted states
                #     all_action_probs = self.actor(adjusted_states.view(-1, 2)).view(states.shape[0], len(wind_effects),
                #                                                                     self.env.action_space.n)
                #
                #     # Marginalize over wind effects
                #     action_probs = torch.einsum('k,bka->ba', probs, all_action_probs)
                # else:
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
