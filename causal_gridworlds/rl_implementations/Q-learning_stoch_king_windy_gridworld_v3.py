# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:51:42 2023

@author: SSubhnil

@details: Q-Learning for Stochastic-King-Windy-gridworld
          V3_version accounts for 10 episodes greedy evaluations.
          Changed to include WandB logging.
"""
import numpy as np
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import random
import math
from util.wind_greedy_evaluations import GreedyEvaluation as evaluate
import wandb
wandb.login(key="576d985d69bfd39f567224809a6a3dd329326993")
wandb.init(project="Q-learn-Stoch-King-Windy-GW", mode="offline")

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

from custom_envs.stoch_king_windy_gridworld_env import StochKingWindyGridWorldEnv

env = StochKingWindyGridWorldEnv()
grid_dimensions = (env.grid_height, env.grid_width)

done = False

np.random.seed(42)


def epsilon_greedy(Q, state, nA, eps):
    if np.random.uniform(0, 1) > eps:
        return np.argmax(Q[state[0]][state[1]])
    else:
        return random.choice(np.arange(nA))


def update_Q_table(alpha, gamma, Q, state, action, reward, next_state=None):
    # Estimate in Q-table (for current state, action pair) Q(S_t, A_t)
    current = Q[state[0]][state[1]][action]
    # Get value of state, action pair at next time step Q(S_{t+1}, A_{t+1})
    Qsa_next = np.max(Q[next_state[0]][next_state[1]]) if next_state is not None else 0
    # Construct TD target R_{t+1} + gamma * Q(S_{t+1}, A_{t+1})
    target = reward + (gamma * Qsa_next)
    # Get updated value Q(S_t, A_t) + alpha * (R_{t+1} + gamma * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
    new_value = current + alpha * (target - current)

    return new_value


def Q_learn(env, num_episodes, alpha, epsilon, greedy_interval, gamma=1.0):
    nA = env.nA
    Q = np.random.rand(env.grid_height, env.grid_width, nA)
    Q[env.goal_state] = np.zeros(8)

    # Initialize the greedy evaluation
    greedy_evaluation = evaluate(env, grid_dimensions)

    eps = epsilon
    total_reward_per_param = 0

    # Loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        # if i_episode % 100 == 0:
        print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
        sys.stdout.flush()

        # Runs greedy evaluation
        if i_episode % greedy_interval == 0:
            _, avg_evaluation_reward, _ = greedy_evaluation.run_algo(Q)
            wandb.log({"Avg. Evaluation Reward": avg_evaluation_reward})

        # Percent of experiment completed - use for exponential decay
        percent_completion = i_episode / num_episodes

        # Initialize score
        score = 0

        # Observe S_0: the initial state
        state = env.reset()

        count = 0

        while True:
            count += 1

            # Choose action A_0 using policy derived from Q (e.g., eps-greedy)
            action = epsilon_greedy(Q, state, nA, eps)
            next_state, reward, done = env.step(action)  # take action A, observe R', S'
            score += reward  # add reward to agent's score
            if not done:
                # Update Q
                Q[state[0]][state[1]][action] = update_Q_table(alpha, gamma, Q, state, action, reward,
                                                               next_state)

                # Update state and action
                state = next_state  # S_t <- S_{t+1}

            if done:
                Q[state[0]][state[1]][action] = update_Q_table(alpha, gamma, Q, state, action, reward)
                break
        wandb.log({'Reward': score, 'Epsilon': eps})
        # Exponential Decay
        eps = math.exp(-2 * math.pow(percent_completion, 3.5) / 0.4)
        total_reward_per_param += score

    return total_reward_per_param

def train_params(config):
    alpha = config.alpha
    num_episodes = 30000
    greedy_interval = 1000

    starting_eps = 1.0

    total_reward_per_param = Q_learn(env, num_episodes, alpha, starting_eps, greedy_interval)

    return total_reward_per_param


def main():
    wandb.init(project="Sweep-Q-learn-Stoch-King-Windy-GW", mode="offline")
    total_reward_per_param = train_params(wandb.config)
    wandb.log({'Total Reward per param': total_reward_per_param})

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "total_reward_per_param"},
    "parameters": {
        "alpha": {"values": [0.1, 0.01, 0.25, 0.5, 0.4]}}}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Sweep-Q-learn-Stoch-King-Windy-GW")

wandb.agent(sweep_id, function = main, count=5)
wandb.finish()

