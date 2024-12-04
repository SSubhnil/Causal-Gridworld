# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:23:19 2023

@author: shubh
@details: SAC for Stochastic Windy Gridworld. Sweep implementation for
hyperparameter search.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import argparse
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import itertools
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set the specific CUDA device (e.g., device 0)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# from causal_gridworlds.util.util import PlotUtil
# from causal_gridworlds.util.util import RepresentationTools as rpt
from util.wind_greedy_evaluations import A2C_GreedyEvaluation as evaluate
from custom_envs.stoch_windy_gridworld_env_v3 import StochWindyGridWorldEnv_V3
from custom_envs.stoch_king_windy_gridworld_env import StochKingWindyGridWorldEnv

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

import wandb

wandb.login(key="576d985d69bfd39f567224809a6a3dd329326993")  # Replace with your actual WandB API key


# Define the Actor network
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
        logits = self.fcf(x)
        return logits  # Return logits; softmax will be applied later


# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fci = nn.Linear(state_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fcf = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fci(state))
        x = F.relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        q_values = self.fcf(x)
        return q_values  # Output Q-values for all actions


# Define a named tuple for experiences
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones


class SAC:
    def __init__(self, env, state_dim, action_dim, hidden_dim,
                 lr_actor, lr_critic, buffer_size, batch_size, entropy_weight,
                 wind_distribution, gamma=0.98, tau=0.005, target_entropy=None, alpha_lr=3e-4):
        self.env = env
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic_1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic_1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic_2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic_1 = optim.Adam(self.critic_1.parameters(), lr=lr_critic)
        self.optimizer_critic_2 = optim.Adam(self.critic_2.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau  # For soft update of target networks
        self.action_dim = action_dim

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.entropy_weight = entropy_weight
        self.wind_distribution_ok = wind_distribution

        # Initialize log_alpha as a learnable parameter
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()

        # Optimizer for alpha
        self.optimizer_alpha = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Define target entropy
        if target_entropy is None:
            # Set target entropy to -|A| where |A| is the number of actions
            self.target_entropy = -np.log(1.0 / action_dim)
        else:
            self.target_entropy = target_entropy

        # Watch the models for gradient logging
        wandb.watch(self.actor, log="all", log_freq=100)
        wandb.watch(self.critic_1, log="all", log_freq=100)
        wandb.watch(self.critic_2, log="all", log_freq=100)

    def select_action(self, state):
        ep = 1e-8
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        logits = self.actor(state)
        log_action_probs = F.log_softmax(logits, dim=-1)
        action_probs = torch.exp(log_action_probs)
        dist = torch.distributions.Categorical(probs=action_probs)
        action = dist.sample()
        return action.item()

    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.LongTensor(actions).view(-1, 1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Critic update
        with torch.no_grad():
            if self.wind_distribution_ok:
                expected_q_targets = []
                for wind_effect, prob in zip(
                        np.arange(-self.env.range_random_wind, self.env.range_random_wind + 1),
                        self.env.probabilities
                ):
                    # Simulate next states for each wind effect
                    next_state_wind = [
                        self.env.clamp_to_grid((state[0] - wind_effect, state[1]))
                        for state in next_states.cpu().numpy()
                    ]
                    next_state_wind = torch.FloatTensor(next_state_wind).to(device)
                    next_logits_wind = self.actor(next_state_wind)
                    next_action_probs_wind = F.softmax(next_logits_wind, dim=-1)
                    next_log_action_probs_wind = F.log_softmax(next_logits_wind, dim=-1)

                    # Compute next Q-values from target critics
                    target_q1_next_wind = self.target_critic_1(next_state_wind)
                    target_q2_next_wind = self.target_critic_2(next_state_wind)
                    target_q_next_wind = torch.min(target_q1_next_wind, target_q2_next_wind)

                    # Compute expected Q-values
                    q_target_wind = next_action_probs_wind * (target_q_next_wind -
                                                              self.alpha * next_log_action_probs_wind)
                    q_target_wind = q_target_wind.sum(dim=1, keepdim=True)
                    expected_q_targets.append(prob * q_target_wind)
                q_target = sum(expected_q_targets)
                q_target = rewards + (1 - dones) * self.gamma * q_target
            else:
                next_logits = self.actor(next_states)
                next_action_probs = F.softmax(next_logits, dim=-1)
                next_log_action_probs = F.log_softmax(next_logits, dim=-1)

                # Compute next Q-values from target critics
                target_q1_next = self.target_critic_1(next_states)
                target_q2_next = self.target_critic_2(next_states)
                target_q_next = torch.min(target_q1_next, target_q2_next)

                # Compute expected Q-values
                q_target = next_action_probs * (target_q_next - self.alpha * next_log_action_probs)
                q_target = q_target.sum(dim=1, keepdim=True)
                q_target = rewards + (1 - dones) * self.gamma * q_target

        # Get current Q estimates
        current_q1 = self.critic_1(states).gather(1, actions)
        current_q2 = self.critic_2(states).gather(1, actions)

        # Compute critic losses
        critic_1_loss = F.mse_loss(current_q1, q_target)
        critic_2_loss = F.mse_loss(current_q2, q_target)

        # Optimize critics
        self.optimizer_critic_1.zero_grad()
        critic_1_loss.backward()
        self.optimizer_critic_1.step()

        self.optimizer_critic_2.zero_grad()
        critic_2_loss.backward()
        self.optimizer_critic_2.step()

        # Actor update
        logits = self.actor(states)
        action_probs = F.softmax(logits, dim=-1)
        log_action_probs = F.log_softmax(logits, dim=-1)

        q1 = self.critic_1(states)
        q2 = self.critic_2(states)
        q_min = torch.min(q1, q2)

        # Actor loss computation
        actor_loss = (action_probs * (self.alpha.detach() * log_action_probs - q_min)).sum(dim=1).mean()

        # Optimize actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Compute entropy
        entropy = - (action_probs * log_action_probs).sum(dim=1, keepdim=True)

        # Compute alpha loss
        alpha_loss = -(self.log_alpha * (entropy.detach() + self.target_entropy)).mean()

        # Update alpha (log_alpha)
        self.optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer_alpha.step()

        # Update alpha value
        self.alpha = self.log_alpha.exp()

        # Soft update target critics
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def store_experience(self, experience):
        self.replay_buffer.add_experience(experience)


def train_params(config):
    # Define environment dimensions and action space
    env = StochKingWindyGridWorldEnv()
    grid_dimensions = (env.grid_height, env.grid_width)
    env.seed(42)
    state_dim = 2
    action_dim = env.nA
    hidden_dim = config["hidden_dim"]
    num_episodes = 15000
    batch_size = config["batch_size"]
    buffer_size = config["buffer_size"]
    learning_rate_actor = config["lr_actor"]
    learning_rate_critic = config["lr_critic"]
    wind_distribution_ok = config["wind_distribution_ok"]
    total_reward_per_param = 0
    entropy_weight = config["entropy_weight"]
    tau = config["tau"]
    gamma = config["gamma"]
    alpha_lr = config.get("alpha_lr", 1e-4)
    entropy_scale = config.get("entropy_scale", 1.0)

    # Create the SAC agent
    agent = SAC(env, state_dim, action_dim, hidden_dim, learning_rate_actor,
                learning_rate_critic, buffer_size, batch_size, entropy_weight,
                wind_distribution_ok, gamma=gamma, tau=tau, target_entropy=-np.log(1.0 / action_dim) * entropy_scale,
                alpha_lr=alpha_lr)
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

            # Store experience
            experience = Experience(state, action, reward, next_state, done)
            agent.store_experience(experience)

            # Train the agent
            agent.train()
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

        total_reward_per_param += episode_reward

    return total_reward_per_param


def main(single_run=False):
    if single_run:
        # Perform a single run
        wandb.init(project="SAC-King-Stoch-GW-Wind_seen", config={
            "lr_actor": 3e-4,
            "lr_critic": 3e-3,
            "buffer_size": 1024,
            "batch_size": 512,
            "wind_distribution_ok": True,
            "entropy_weight": 0.2,
            "hidden_dim": 128,
            "tau": 0.005,
            "gamma": 0.98,
        }, mode="disabled")

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
            "method": "grid",  # Exhaustive search over all parameter combinations
            "metric": {"goal": "maximize", "name": "Total_Reward"},  # Optimize for total reward
            "parameters": {
                "lr_actor": {"values": [3e-4, 1e-4]},
                "lr_critic": {"values": [1e-3, 3e-3]},
                "buffer_size": {"values": [1024, 2048]},
                "batch_size": {"values": [128, 256, 512]},
                "wind_distribution_ok": {"values": [False, True]},
                "entropy_weight": {"values": [0.1, 0.2]},
                "hidden_dim": {"values": [64]},
                "tau": {"values": [0.005]},
                "gamma": {"values": [0.95]},
                # "alpha_lr": {"values": [1e-3, 3e-4]},
                # "entropy_scale": {"values": [0.5, 1.0]}
            },
        }

        # Initialize the sweep
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="SAC-King-Stoch-GW-Wind_seen")
        wandb.agent(sweep_id, function=sweep_main)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAC training.")
    parser.add_argument("--single_run", action="store_true",
                        help="Run a single training instead of a sweep.")
    args = parser.parse_args()

    main(single_run=args.single_run)
