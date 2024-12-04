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
from causal_gridworlds.util.wind_greedy_evaluations import GreedyEvaluation as evaluate
import wandb
wandb.login(key="576d985d69bfd39f567224809a6a3dd329326993")


import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

from causal_gridworlds.custom_envs.stoch_king_windy_gridworld_env import StochKingWindyGridWorldEnv

env = StochKingWindyGridWorldEnv()
grid_dimensions = (env.grid_height, env.grid_width)

done = False

np.random.seed(42)


def epsilon_greedy(Q, state, nA, eps):
    if np.random.uniform(0, 1) > eps:
        return np.argmax(Q[state[0]][state[1]])
    else:
        return random.choice(np.arange(nA))

# Wind distribution is optional - choose expected_Qsa_next
def update_Q_table(alpha, gamma, Q, state, action, reward, wind_distribution_ok, next_state=None):
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
            for wind_effect, prob in zip(np.arange(-env.range_random_wind, env.range_random_wind + 1), env.probablities):
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


def Q_learn(env, num_episodes, alpha, epsilon, greedy_interval, wind_distribution_ok, gamma=1.0):
    nA = env.nA
    Q = np.random.rand(env.grid_height, env.grid_width, nA)
    Q[env.goal_state] = np.zeros(8)

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
            next_state, reward, done, _ = env.step(action)  # take action A, observe R', S'
            score += reward  # add reward to agent's score
            if not done:
                # Update Q
                Q[state[0]][state[1]][action] = update_Q_table(alpha, gamma, Q, state, action, reward,
                                                               wind_distribution_ok, next_state)

                # Update state and action
                state = next_state  # S_t <- S_{t+1}

            if done:
                Q[state[0]][state[1]][action] = update_Q_table(alpha, gamma, Q, state, action, reward, wind_distribution_ok)
                break
        wandb.log({'Reward': score, 'Epsilon': eps})
        # Exponential Decay
        eps = math.exp(-2 * math.pow(percent_completion, 3.5) / 0.4)

        # Linear Decay
        #eps = eps_start - (percent_completion * (eps_start - eps_end))  # Linear decay
        #eps = max(eps, eps_end)  # Ensure epsilon doesn't go below eps_end

        total_reward_per_param += score

    return total_reward_per_param

def train_params(alpha, wind_distribution_ok, seed):
    np.random.seed(seed)
    num_episodes = 30000
    greedy_interval = 1000
    starting_eps = 1.0

    # Pass alpha and wind_distribution_ok to Q_learn
    total_reward_per_param = Q_learn(env, num_episodes, alpha, starting_eps, greedy_interval, wind_distribution_ok)
    return total_reward_per_param


def main():
    # Initialize WandB for logging
    wandb.init(project="Q-learn-Wind_dist_known", mode="online")  # Switch "mode" to "online" or "offline" as needed

    # Manually set parameters
    alpha = 0.4  # Example alpha value
    wind_distribution_ok = False  # Example setting for wind distribution visibility

    # Train the agent and log results
    total_reward_per_param = train_params(alpha, wind_distribution_ok)
    wandb.log({
        'Alpha': alpha,
        'Wind Distribution Known': wind_distribution_ok,
        'Total Reward per param': total_reward_per_param
    })

    # Finish WandB logging
    wandb.finish()


if __name__ == "__main__":
    main()


