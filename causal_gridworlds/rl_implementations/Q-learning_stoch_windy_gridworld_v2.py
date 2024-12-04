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

import wandb
wandb.login(key="576d985d69bfd39f567224809a6a3dd329326993")


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_envs.stoch_windy_gridworld_env_v3 import StochWindyGridWorldEnv_V3
from custom_envs.stoch_king_windy_gridworld_env import StochKingWindyGridWorldEnv
from util.wind_greedy_evaluations import GreedyEvaluation as evaluate


def epsilon_greedy(Q, state, nA, eps):
    if np.random.uniform(0, 1) > eps:
        return np.argmax(Q[state[0]][state[1]])
    else:
        return random.choice(np.arange(nA))

# Wind distribution is optional - choose expected_Qsa_next
def update_Q_table(env, alpha, gamma, Q, state, action, reward, wind_distribution_ok, next_state=None):
    # Estimate in Q-table (for current state, action pair) Q(S_t, A_t)
    current = Q[state[0]][state[1]][action]
    # Get value of state, action pair at next time step Q(S_{t+1}, A_{t+1})
    if wind_distribution_ok:
        # For making the wind distribution available
        Qsa_next = 0
        realized_state = env.action_destination(state, action)

        # Check if the column is windy
        wind_strength = env.wind[realized_state[1]] # Get wind strength for the column
        if wind_strength > 0:
            for wind_effect, prob in zip(np.arange(-env.range_random_wind, env.range_random_wind + 1), env.probabilities):
                # Simulate the next state given the wind effect
                realized_state = env.action_destination(state, action)
                # Apply wind effect and clamp to grid dimensions
                next_state = env.clamp_to_grid((realized_state[0] - wind_effect, realized_state[1]))
                Qsa_next += prob * np.max(Q[next_state[0]][next_state[1]])

        else:
            # No wind in this column, directly use the realized state
            next_state = realized_state
            Qsa_next = np.max(Q[next_state[0]][next_state[1]])

    else:
        Qsa_next = np.max(Q[next_state[0]][next_state[1]]) if next_state is not None else 0

    # Construct TD target R_{t+1} + gamma * Q(S_{t+1}, A_{t+1})
    target = reward + (gamma * Qsa_next)
    # Get updated value Q(S_t, A_t) + alpha * (R_{t+1} + gamma * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
    new_value = current + alpha * (target - current)
    return new_value


def Q_learn(env, num_episodes, alpha, epsilon, greedy_interval, grid_dimensions, wind_distribution_ok, gamma=1.0):
    nA = env.nA
    Q = np.random.rand(env.grid_height, env.grid_width, nA)
    Q[env.goal_state] = np.zeros(nA)

    # Initialize the greedy evaluation
    greedy_evaluation = evaluate(env, grid_dimensions)

    eps = epsilon
    eps_start = epsilon
    eps_end = 0.01
    total_reward_per_param = 0

    # Loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        # if i_episode % 100 == 0:
        print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
        sys.stdout.flush()

        # Runs greedy evaluation
        if i_episode % greedy_interval == 0:
            _, avg_evaluation_reward = greedy_evaluation.run_algo(Q)
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
            next_state, reward, done, _ = env.step(action)  # take action A, observe R', S'
            score += reward  # add reward to agent's score
            if not done:
                # Update Q
                Q[state[0]][state[1]][action] = update_Q_table(env, alpha, gamma, Q, state, action, reward,
                                                               wind_distribution_ok, next_state)

                # Update state and action
                state = next_state  # S_t <- S_{t+1}

            if done:
                Q[state[0]][state[1]][action] = update_Q_table(env, alpha, gamma, Q, state, action, reward, wind_distribution_ok)
                break
        wandb.log({'Reward': score, 'Epsilon': eps})
        # Exponential Decay
        # eps = math.exp(-2 * math.pow(percent_completion, 3.5) / 0.4)

        # Linear Decay
        eps = eps_start - (percent_completion * (eps_start - eps_end))  # Linear decay
        eps = max(eps, eps_end)  # Ensure epsilon doesn't go below eps_end

        total_reward_per_param += score

    return total_reward_per_param


def train_params(alpha, wind_distribution_ok, seed, env_actions):
    # Set the seed for reproducibility
    np.random.seed(seed)

    # Initialize the environment
    if env_actions == "King":
        env = StochKingWindyGridWorldEnv()
    else:
        env = StochWindyGridWorldEnv_V3()
    env.seed(seed)
    grid_dimensions = (env.grid_height, env.grid_width)
    num_episodes = 30000
    greedy_interval = 1000
    starting_eps = 1.0

    # Pass alpha and wind_distribution_ok to Q_learn
    total_reward_per_param = Q_learn(env, num_episodes, alpha, starting_eps, greedy_interval, grid_dimensions,
                                     wind_distribution_ok)
    return total_reward_per_param


def main():
    seeds = [42, 123, 456, 789, 101112]  # List of seeds to iterate over
    alpha = 0.4  # Example alpha value
    wind_distribution_ok = True  # Example setting for wind distribution visibility
    env_action_space = "King" # "4A"
    for seed in seeds:
        wandb.init(
            project="Q-learn-Wind_dist_known",
            config={
                'Alpha': alpha,
                'Wind Distribution Known': wind_distribution_ok,
                'Seed': seed,
                'env_actions': env_action_space,
            },
            group="Q_L-King-Multi-Seed",
            job_type=f"seed-{seed}",
            mode="online"  # Switch to "offline" for local logging
        )

        # Run training for the given seed
        total_reward_per_param = train_params(alpha, wind_distribution_ok, seed, env_action_space)

        # Log results to WandB
        wandb.log({
            'Alpha': alpha,
            'Wind Distribution Known': wind_distribution_ok,
            'Seed': seed,
            'Total Reward per param': total_reward_per_param
        })

        # Finish WandB logging for the current seed
        wandb.finish()


if __name__ == "__main__":
    main()


