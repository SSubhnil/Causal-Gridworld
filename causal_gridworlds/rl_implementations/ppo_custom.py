import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import wandb
wandb.login(key="YOUR_WANDB_API_KEY")
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

from util.wind_greedy_evaluations import A2C_GreedyEvaluation as evaluate  # Update this if you have a specific evaluation for PPO
from custom_envs.stoch_windy_gridworld_env_v3 import StochWindyGridWorldEnv_V3

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

is_ipython = 'inline' in matplotlib.get_backend()

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fci = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = state
        x = F.leaky_relu(self.fci(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        # x = torch.tanh(x)
        action_probs = self.action_head(x)
        action_probs = self.softmax(action_probs)
        return action_probs


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fci = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.state_value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = state
        x = F.leaky_relu(self.fci(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        # x = torch.tanh(x)
        value = self.state_value(x)
        return value

# Define a named tuple for experiences
Experience = namedtuple("Experience", ["state", "action", "log_prob", "reward", "next_state", "done", "value"])

class PPO:
    def __init__(self, env, state_dim, action_dim, hidden_dim,
                 lr_actor, lr_critic, buffer_size, batch_size, entropy_weight,
                 wind_distribution, gamma=0.98, clip_epsilon=0.2, lamda=0.95, K_epochs=4):
        self.env = env
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.lamda = lamda
        self.K_epochs = K_epochs
        self.action_dim = action_dim

        self.buffer_size = buffer_size
        self.replay_buffer = []
        self.batch_size = batch_size
        self.entropy_weight = entropy_weight
        self.wind_distribution_ok = wind_distribution

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        logits = self.actor(state)
        action_probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        value = self.critic(state)
        return action.item(), action_logprob.item(), value.item()

    def store_experience(self, experience):
        self.replay_buffer.append(experience)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

    def train(self):
        # Convert experience list to tensors
        states = torch.FloatTensor([exp.state for exp in self.replay_buffer]).to(device)
        actions = torch.LongTensor([exp.action for exp in self.replay_buffer]).view(-1, 1).to(device)
        old_log_probs = torch.FloatTensor([exp.log_prob for exp in self.replay_buffer]).view(-1, 1).to(device)
        rewards = [exp.reward for exp in self.replay_buffer]
        dones = [exp.done for exp in self.replay_buffer]
        values = torch.FloatTensor([exp.value for exp in self.replay_buffer]).view(-1, 1).to(device)

        # **Compute expected next state values considering wind distribution**
        expected_next_values = []
        for i in range(len(self.replay_buffer)):
            if self.wind_distribution_ok:
                expected_value = 0
                next_state = self.replay_buffer[i].next_state
                # Iterate over possible wind effects
                for wind_effect, prob in zip(
                        np.arange(-self.env.range_random_wind, self.env.range_random_wind + 1),
                        self.env.probablities
                ):
                    # Adjust next_state by wind effect
                    adjusted_state = self.env.clamp_to_grid(
                        (next_state[0] - wind_effect, next_state[1])
                    )
                    adjusted_state_tensor = torch.FloatTensor(adjusted_state).unsqueeze(0).to(device)
                    value = self.critic(adjusted_state_tensor)
                    expected_value += prob * value.item()
                expected_next_values.append(expected_value)
            else:
                if i + 1 < len(values):
                    expected_next_values.append(values[i + 1].item())
                else:
                    expected_next_values.append(0)

        # Compute returns and advantages
        returns = []
        advantages = []
        G = 0
        gae = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                G = 0
                gae = 0
            G = rewards[i] + self.gamma * G
            delta = rewards[i] + self.gamma * expected_next_values[i] - values[i].item()
            gae = delta + self.gamma * self.lamda * gae
            returns.insert(0, G)
            advantages.insert(0, gae)

        returns = torch.FloatTensor(returns).view(-1, 1).to(device)
        advantages = torch.FloatTensor(advantages).view(-1, 1).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize policy and value network
        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.replay_buffer))), self.batch_size, False):
                batch_states = states[index]
                batch_actions = actions[index]
                batch_old_log_probs = old_log_probs[index]
                batch_returns = returns[index]
                batch_advantages = advantages[index]

                # Evaluate actions
                logits = self.actor(batch_states)
                action_probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                action_log_probs = dist.log_prob(batch_actions.squeeze()).view(-1, 1)
                entropy = dist.entropy().mean()

                # Compute ratios
                ratios = torch.exp(action_log_probs - batch_old_log_probs.detach())

                # Surrogate loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_weight * entropy

                # Critic loss
                values = self.critic(batch_states)
                critic_loss = F.mse_loss(values, batch_returns)

                # Total loss
                loss = actor_loss + 0.5 * critic_loss

                # Update networks
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Clear buffer
        self.replay_buffer = []

def train_params(config):
    # Define environment dimensions and action space
    env = StochWindyGridWorldEnv_V3()
    grid_dimensions = (env.grid_height, env.grid_width)
    env.seed(42)
    state_dim = 2
    action_dim = env.nA
    hidden_dim = 64
    num_episodes = 30000

    batch_size = config["batch_size"]
    buffer_size = config["buffer_size"]
    learning_rate_actor = config["lr_actor"]
    learning_rate_critic = 1e-3
    wind_distribution_ok = config["wind_distribution_ok"]
    total_reward_per_param = 0
    entropy_weight = config["entropy_weight"]

    # PPO-specific hyperparameters
    clip_epsilon = config.get("clip_epsilon", 0.2)
    lamda = config.get("lamda", 0.95)
    K_epochs = config.get("K_epochs", 10)

    # Create the PPO agent
    agent = PPO(env, state_dim, action_dim, hidden_dim, learning_rate_actor,
                learning_rate_critic, buffer_size, batch_size, entropy_weight,
                wind_distribution_ok, clip_epsilon=clip_epsilon, lamda=lamda, K_epochs=K_epochs)
    # Initialize the greedy evaluation
    greedy_evaluation = evaluate(env, grid_dimensions, device)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Select an action
            action, action_logprob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store experience
            experience = Experience(state, action, action_logprob, reward, next_state, done, value)
            agent.store_experience(experience)

            episode_reward += reward
            state = next_state

        # Update after every episode
        agent.train()

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

def main():
    # Initialize WandB with the current sweep configuration
    wandb.init(project="PPO-Stoch-GW-Wind_seen")

    # Access the configuration for the current run from WandB
    config = wandb.config

    # Call your existing train_params() function with the current config
    total_reward_per_param = train_params(config)

    # Log the final result
    wandb.log({"Total_Reward": total_reward_per_param})

    # Finish the WandB run
    wandb.finish()

def run_agent(sweep_id):
    wandb.agent(sweep_id, function=main, count=1)

if __name__ == "__main__":
    # Define the sweep configuration
    sweep_configuration = {
        "method": "grid",  # Exhaustive search over all parameter combinations
        "metric": {"goal": "maximize", "name": "Total_Reward"},  # Optimize for total reward
        "parameters": {
            "lr_actor": {"values": [3e-4]},
            "buffer_size": {"values": [1024, 512]},
            "batch_size": {"values": [64, 128, 256]},
            "wind_distribution_ok": {"values": [False, True]},
            "entropy_weight": {"values": [0.01]},
            "clip_epsilon": {"values": [0.2]},
            "lamda": {"values": [0.95]},
            "K_epochs": {"values": [4]}
        },
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="PPO-Stoch-GW-Wind_seen")

    # Run the sweep
    wandb.agent(sweep_id, function=main)