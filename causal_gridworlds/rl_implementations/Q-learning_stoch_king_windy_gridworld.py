# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 19:37:40 2023

@author: SSubhnil

@details: Q-Learning for Stochastic-King-Windy-gridworld
"""
import numpy as np
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import random
import math
# matplotlib inline

# import check_test
# from plot_utils import plot_values


import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

from envs.stoch_king_windy_gridworld_env import StochKingWindyGridWorldEnv

env = StochKingWindyGridWorldEnv()

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
    # Initialize action-value function (empty dictionary of arrays)
    nA = env.nA
    # Q = defaultdict(lambda: np.zeros(nA))
    Q = np.random.rand(env.grid_height, env.grid_width, nA)
    Q[env.goal_state] = np.zeros(8)
    
    # To store wind profiles
    wind_profile = np.empty((num_episodes, env.grid_width))
    Q_store = np.empty((num_episodes, env.grid_height, env.grid_width, env.nA))
    greedy_episodes = np.empty((int(num_episodes/greedy_interval), 1)).reshape(int(num_episodes/greedy_interval))
    step_count = np.empty((num_episodes, 1))
    
    
    eps = epsilon
    greedy_flag = 0
    sampling_counter = -1
    greedy_eps = 0.01
    
    # Loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        # if i_episode % 100 == 0:
        print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
        sys.stdout.flush()
        
        if i_episode % greedy_interval == 0:
            greedy_flag = 1
            sampling_counter += 1
        
        # Percent of experiment completed - use for exponential decay
        percent_completion = i_episode/num_episodes
        
        # Initialize score
        score = 0
        
        # Check if greedy episode
        if greedy_flag == 0:
            # epsilon annealing - exponential decay
            eps = math.exp(-2*math.pow(percent_completion, 3.5)/0.4)
        else:
            eps = greedy_eps

        # Observe S_0: the initial state
        state = env.reset()
        
        count = 0
        
        while True:
            count += 1
            
            # Choose action A_0 using policy derived from Q (e.g., eps-greedy)
            action = epsilon_greedy(Q, state, nA, eps)
            next_state, reward, done, info = env.step(action)  # take action A, observe R', S'
            score += reward  # add reward to agent's score
            if not done:
                # Update Q
                if greedy_flag == 0:
                    Q[state[0]][state[1]][action] = update_Q_table(alpha, gamma, Q, state, action, reward,
                                                                   next_state)

                # Update state and action
                state = next_state  # S_t <- S_{t+1}
                # print(count)

            if done:
                Q[state[0]][state[1]][action] = update_Q_table(alpha, gamma, Q, state, action, reward)
                Q_store[i_episode-1] = Q
                wind_profile[i_episode - 1] = env.realized_wind
                if greedy_flag == 1:
                    greedy_episodes[sampling_counter] = count
                
                greedy_flag = 0
                step_count[i_episode - 1][0] = count
                break

    return Q, step_count, greedy_episodes, Q_store, wind_profile


def moving_average(step_count, n = 300):
    running_average = np.cumsum(step_count, dtype=float)
    running_average[n:] = running_average[n:] - running_average[:-n]
    return running_average[n - 1:] / n


# UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3

# Loop over hyper-params
alpha = [0.1, 0.25, 0.4, 0.5]
num_episodes = [10000, 20000, 40000]
greedy_interval = [1000, 2000, 4000]

# Use one hyper-param
# alpha = [0.25]
# num_episodes = [10000]
# greedy_interval = [1000]

starting_eps = 1.0
experiment_number = 1

for i in range(0, len(alpha)):
    for j in range(0, len(num_episodes)):

        Q_table, step_count, greedy_episodes, Q_store, wind_profile = Q_learn(env, num_episodes[j], alpha[i], starting_eps, greedy_interval[j])
        running_average = moving_average(step_count)        
        
        np.save("Q-learn-Stoch-King-Windy-GW-Q_values-new_{}.npy".format(experiment_number), Q_store)
        np.save("Q-learn-Stoch-King-Windy-GW-step_count-new_{}.npy".format(experiment_number), step_count)
        np.save("Q-learn-Stoch-King-Windy-GW-wind_profile-new_{}.npy".format(experiment_number), wind_profile)
        # print("Q_sarsa\n", Q_sarsa)
        
        spacer1 = np.arange(1, len(running_average)+1)
        spacer2 = np.arange(1, num_episodes[j], greedy_interval[j])
        
        print("\n 0: Up, 1: Right, 2: Down, 3: Left")
        print("\n Minimum steps:", min(step_count))
   
        # Plotting Running Average
        fig, ax1 = plt.subplots()
        
        color = 'tab:red'
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Running Average (steps/episode)', color=color)
        ax1.plot(spacer1, running_average, color=color, label="Running Avg.")
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        
        color = 'tab:blue'
        ax2.set_ylabel('Greedy Evaluations (steps/batch)', color=color)
        ax2.plot(spacer2, greedy_episodes, color=color, label ="Greedy Eval.")
        for x,y in zip(spacer2, greedy_episodes):
            ax2.annotate('%s' % y, xy=(x,y), textcoords = 'data')
        ax2.tick_params(axis='y', labelcolor=color)
        plt.title('Q-learn-Stoch-King-Windy-GW alp=%f' % alpha[i] )
        plt.legend(loc="upper right")
        plt.savefig('Q-learn-Stoch-King-Windy-GW-new_{}.png'.format(experiment_number), dpi = 1000)
        experiment_number += 1

