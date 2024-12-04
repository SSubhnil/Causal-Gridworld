# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:56:59 2023

@author: SSubhnil
@details: Runs 100 greedy episodes per greedy evaluation for Stochastic Windy GW.
          Available for both 4-actions and King-actions action spaces.
          If the wind is uniformaly random, the policy needs to show similar
          behavior over 100 greedy episodes.
          Implementiing this in the main code would make it very messy so a
          separate Q-value function is defined here.
"""

import numpy as np
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import random
import math
# matplotlib inline

import torch
import torch.nn as nn
import torch.optim as optim

# import check_test
# from plot_utils import plot_values


import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

np.random.seed(42)


class GreedyEvaluation:
    def __init__(self, ENV, GRID_DIMENSIONS, num_episodes=100):
        """
        Initializes the evaluation class for Q-learning.
        Args:
            ENV: The environment instance.
            GRID_DIMENSIONS: Dimensions of the grid (height, width).
            num_episodes: Number of evaluation episodes.
        """
        self.env = ENV
        self.grid_dimensions = GRID_DIMENSIONS
        self.num_episodes = num_episodes
        self.Q_store = np.empty((self.num_episodes, self.env.grid_height, self.env.grid_width, self.env.nA))

    def epsilon_greedy(self, Q, state):
        """
        Selects the greedy action (epsilon = 0).
        Args:
            Q: The Q-table.
            state: The current state of the environment.
        Returns:
            The greedy action as an integer.
        """
        return np.argmax(Q[state[0]][state[1]])

    def Q_learn(self, Q, gamma=1.0):
        """
        Runs the greedy evaluation for Q-learning.
        Args:
            Q: The Q-table.
            gamma: Discount factor (default=1.0).
        Returns:
            rewards_per_episode: List of rewards for each evaluation episode.
        """
        rewards_per_episode = []  # Store total rewards per episode

        # Loop over episodes
        for i_episode in range(1, self.num_episodes + 1):
            # Initialize the episode
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select greedy action
                action = self.epsilon_greedy(Q, state)

                # Take action and observe next state and reward
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                # Update state
                state = next_state

            # Store the total reward for this episode
            rewards_per_episode.append(episode_reward)

        return rewards_per_episode

    def run_algo(self, Q):
        """
        Runs the greedy evaluation and computes statistics.
        Args:
            Q: The Q-table.
        Returns:
            rewards_per_episode: A list of rewards for each evaluation episode.
            avg_reward: The mean reward over the evaluation episodes.
        """
        rewards_per_episode = self.Q_learn(Q)
        avg_reward = np.mean(rewards_per_episode)
        return rewards_per_episode, avg_reward


class DQN_GreedyEvaluation:
    def __init__(self, ENV, GRID_DIMENSIONS, device):
        self.env = ENV
        self.grid_dimensions = GRID_DIMENSIONS
        self.num_episodes = 100  # Number of episodes for evaluation
        self.device = device

    def choose_action(self, model, state, eps=0.01):
        """
        Selects an action using an epsilon-greedy policy with respect to the Q-network.
        """
        if np.random.rand() <= eps:
            # Random action
            return random.randrange(self.env.nA)
        else:
            # Forward pass through the model to get Q-values
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = model(state_tensor).squeeze(0)
            return torch.argmax(q_values).item()

    def DQN_learn(self, model, gamma=0.98):
        """
        Runs the greedy evaluation for DQN.
        """
        score_array = np.empty((self.num_episodes, 1))
        wind_profile = np.empty((self.num_episodes, self.env.grid_width))

        # Loop over episodes
        for i_episode in range(1, self.num_episodes + 1):
            # Initialize score
            score = 0

            # Reset the environment and get the initial state
            state = self.env.reset()

            while True:
                # Choose action using the policy derived from the Q-network
                action = self.choose_action(model, state)

                # Take the action and observe the next state and reward
                next_state, reward, done, info = self.env.step(action)
                score += reward  # Add reward to the total score

                # Update state
                state = next_state

                # Break if the episode ends
                if done:
                    wind_profile[i_episode - 1] = self.env.realized_wind
                    score_array[i_episode - 1][0] = score
                    break

        return score_array, wind_profile

    def run_algo(self, model):
        """
        Runs the evaluation and calculates the average score.
        """
        score, wind_profile = self.DQN_learn(model)
        avg_score = np.mean(score)
        return score, avg_score, wind_profile


class A2CGRU_GreedyEvaluation:
    def __init__(self, env, grid_dimensions, device, num_episodes=100):
        """
        Initializes the evaluation class for A2C.
        Args:
            env: The environment instance.
            grid_dimensions: Dimensions of the grid (height, width).
            device: Torch device (CPU/GPU).
            num_episodes: Number of evaluation episodes.
        """
        self.env = env
        self.grid_dimensions = grid_dimensions
        self.device = device
        self.num_episodes = num_episodes

    def greedy_action(self, actor, state, hidden_actor=None):
        """
        Selects the greedy action based on the highest probability from the GRU actor network.
        Args:
            actor: The GRU actor network of the A2C model.
            state: The current state of the environment.
            hidden_actor: The hidden state of the GRU from the previous timestep.
        Returns:
            action: The greedy action as an integer.
            hidden_actor: Updated hidden state for the GRU.
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
            self.device)  # Shape: (1, 1, state_dim)
        with torch.no_grad():
            action_probs, hidden_actor = actor(state, hidden_actor)  # Forward pass through GRU
            action_probs = action_probs.squeeze(0).cpu().numpy()  # Get action probabilities
        return np.argmax(action_probs), hidden_actor

    def evaluate(self, actor):
        """
        Runs greedy evaluation for a specified number of episodes and returns rewards.
        Args:
            actor: The GRU actor network of the A2C model.
        Returns:
            rewards_per_episode: A list of rewards for each evaluation episode.
            avg_reward: The mean reward over the evaluation episodes.
        """
        rewards_per_episode = []

        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            # Initialize hidden state for GRU
            hidden_actor = None

            while not done:
                # Get action and updated hidden state
                action, hidden_actor = self.greedy_action(actor, state, hidden_actor)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state

            # Record total reward for this episode
            rewards_per_episode.append(episode_reward)

        avg_reward = np.mean(rewards_per_episode)
        return rewards_per_episode, avg_reward

class A2C_GreedyEvaluation:
    def __init__(self, env, grid_dimensions, device, num_episodes=100):
        """
        Initializes the evaluation class for A2C.
        Args:
            env: The environment instance.
            grid_dimensions: Dimensions of the grid (height, width).
            device: Torch device (CPU/GPU).
            num_episodes: Number of evaluation episodes.
        """
        self.env = env
        self.grid_dimensions = grid_dimensions
        self.device = device
        self.num_episodes = num_episodes

    def greedy_action(self, actor, state):
        """
        Selects the greedy action based on the highest probability from the actor network.
        Args:
            actor: The actor network of the A2C model.
            state: The current state of the environment.
        Returns:
            The greedy action as an integer.
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)  # Add batch dimension
        with torch.no_grad():
            action_probs = actor(state).cpu().numpy().squeeze()  # Get action probabilities
        return np.argmax(action_probs)  # Choose action with the highest probability

    def evaluate(self, actor):
        """
        Runs greedy evaluation for a specified number of episodes and returns rewards.
        Args:
            actor: The actor network of the A2C model.
        Returns:
            rewards_per_episode: A list of rewards for each evaluation episode.
            avg_reward: The mean reward over the evaluation episodes.
        """
        rewards_per_episode = []

        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.greedy_action(actor, state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state

            # Record total reward for this episode
            rewards_per_episode.append(episode_reward)

        avg_reward = np.mean(rewards_per_episode)
        return rewards_per_episode, avg_reward

class SAC_GreedyEvaluation_JAX:
    def __init__(self, env, grid_dimensions, num_episodes=100):
        """
        Initializes the evaluation class for A2C using JAX.
        Args:
            env: The environment instance.
            grid_dimensions: Dimensions of the grid (height, width).
            num_episodes: Number of evaluation episodes.
        """
        self.env = env
        self.grid_dimensions = grid_dimensions
        self.num_episodes = num_episodes

    def greedy_action(self, actor_params, actor_apply_fn, state):
        """
        Selects the greedy action based on the highest probability from the actor network.
        Args:
            actor_params: Parameters of the actor network.
            actor_apply_fn: Apply function of the actor network.
            state: The current state of the environment.
        Returns:
            The greedy action as an integer.
        """
        state = jnp.array(state).reshape(1, -1)  # Add batch dimension

        logits = actor_apply_fn(actor_params, state)
        action_probs = nn.softmax(logits)
        action = jnp.argmax(action_probs, axis=-1)
        return int(action[0])

    def evaluate(self, actor_params, actor_apply_fn):
        """
        Runs greedy evaluation for a specified number of episodes and returns rewards.
        Args:
            actor_params: Parameters of the actor network.
            actor_apply_fn: Apply function of the actor network.
        Returns:
            rewards_per_episode: A list of rewards for each evaluation episode.
            avg_reward: The mean reward over the evaluation episodes.
        """
        rewards_per_episode = []

        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.greedy_action(actor_params, actor_apply_fn, state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state

            # Record total reward for this episode
            rewards_per_episode.append(episode_reward)

        avg_reward = np.mean(rewards_per_episode)
        return rewards_per_episode, avg_reward

