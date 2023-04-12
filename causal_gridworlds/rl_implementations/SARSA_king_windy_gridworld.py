# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 18:24:16 2022

@author: SSubhnil

@details: TD-SARSA for King-Windy-gridworld
"""
import numpy as np
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import random
import math
from util import PlotUtil
from wind_greedy_evaluations import GreedyEvaluation as evaluate
# matplotlib inline

# import check_test
# from plot_utils import plot_values


import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

from envs.king_windy_gridworld_env import KingWindyGridWorldEnv

env = KingWindyGridWorldEnv()

grid_dimensions = (env.grid_height, env.grid_width)

done = False

np.random.seed(42)


def epsilon_greedy(Q, state, nA, eps):
    if np.random.uniform(0, 1) > eps:
        return np.argmax(Q[state[0]][state[1]])
    else:
        return random.choice(np.arange(nA))


def update_Q_sarsa(alpha, gamma, Q, state, action, reward, next_state=None, next_action=None):
    # Estimate in Q-table (for current state, action pair) Q(S_t, A_t)
    current = Q[state[0]][state[1]][action]
    # Get value of state, action pair at next time step Q(S_{t+1}, A_{t+1})
    Qsa_next = Q[next_state[0]][next_state[1]][next_action] if next_state is not None else 0
    # Construct TD target R_{t+1} + gamma * Q(S_{t+1}, A_{t+1})
    target = reward + (gamma * Qsa_next)
    # Get updated value Q(S_t, A_t) + alpha * (R_{t+1} + gamma * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
    new_value = current + alpha * (target - current)

    return new_value


def sarsa(env, num_episodes, alpha, epsilon, greedy_interval, gamma=1.0):
    # Initialize action-value function (empty dictionary of arrays)
    nA = env.nA
    # Q = defaultdict(lambda: np.zeros(nA))
    Q = np.random.rand(env.grid_height, env.grid_width, nA)
    Q[env.goal_state] = np.zeros(8)
    
    # Initialize the greedy evaluation
    greedy_evaluation = evaluate(env, grid_dimensions)
    
    step_count = np.empty((num_episodes, 1))
    Q_store = np.empty((num_episodes, env.grid_height, env.grid_width, env.nA))
    greedy_step_count = np.empty((int(num_episodes/greedy_interval), greedy_evaluation.num_episodes, 1))
    avg_greedy_step_count = np.empty(int(num_episodes/greedy_interval))
    
    cmap = plt.get_cmap('viridis')
    u_plt = PlotUtil(cmap, grid_dimensions)
    
    eps = epsilon
    sampling_counter = 0
    
    # Loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        # if i_episode % 100 == 0:
        print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
        sys.stdout.flush()
        
        if i_episode % greedy_interval == 0:
            greedy_step_count[sampling_counter], avg_greedy_step_count[sampling_counter],\
                _ = greedy_evaluation.run_algo(alpha, Q)
            sampling_counter += 1

        # Initialize score
        score = 0
        
        # Percent of experiment completed - use for exponential decay
        percent_completion = i_episode/num_episodes
        
        # epsilon annealing - exponential decay
        eps = math.exp(-2*math.pow(percent_completion, 3.5)/0.4)

        # Observe S_0: the initial state
        state = env.reset()

        # Choose action A_0 using policy derived from Q (e.g., eps-greedy)
        action = epsilon_greedy(Q, state, nA, eps)
        count = 0
        
        while True:
            count += 1
            next_state, reward, done, info = env.step(action)  # take action A, observe R', S'
            score += reward  # add reward to agent's score
            if not done:
                # Choose action A_{t+1} using policy derived fromQ (e.g., eps-greedy)
                next_action = epsilon_greedy(Q, state, nA, eps)
                
                # Update Q
                Q[state[0]][state[1]][action] = update_Q_sarsa(alpha, gamma, Q, state, action, reward,
                                                               next_state, next_action)

                # Update state and action
                state = next_state  # S_t <- S_{t+1}
                action = next_action  # A_t <- A_{t+1}

            if done:
                Q[state[0]][state[1]][action] = update_Q_sarsa(alpha, gamma, Q, state, action, reward)
                Q_store[i_episode-1] = Q
                
                # For printing Q_table with arrows
                # u_plt.showArrowsQ(Q, cmap, i_episode)
                
                # For printing Q_table with min max Q_values per state
                # u_plt.showMinMaxQ(Q, cmap, i_episode)
                
                # For printing Q_table with arrows and min, max Q_values per state
                u_plt.showArrowsMinMaxQ(Q, cmap, i_episode)
                
                step_count[i_episode - 1][0] = count
                break

    return Q, step_count, greedy_step_count, avg_greedy_step_count, Q_store

def moving_average(step_count, n = 300):
    running_average = np.cumsum(step_count, dtype=float)
    running_average[n:] = running_average[n:] - running_average[:-n]
    return running_average[n - 1:] / n

# UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3
alpha = [0.1, 0.25, 0.3, 0.4, 0.5]
num_episodes = [10000, 20000, 40000]
starting_epsilon = 1.0
greedy_interval = [1000, 2000, 4000]
experiment_number = 1

for i in range(0, len(alpha)):
    for j in range(0, len(num_episodes)):
        Q_table, step_count, greedy_step_count, avg_greedy_step_count, Q_store = sarsa(env, num_episodes[j], alpha[i], starting_epsilon, greedy_interval[j])
        running_average = moving_average(step_count)
        
        np.save("SARSA-King-Wind-GW-Q-val-greedy_eval-{}.npy".format(experiment_number), Q_store)
        np.save("SARSA-King-Wind-GW-Step_count-greedy_eval-{}.npy".format(experiment_number), step_count)
        np.save("SARSA-Windy-GW-Greedy_Step_count-greedy_eval-{}.npy".format(experiment_number), greedy_step_count)        
        
        spacer1 = np.arange(1, len(running_average)+1)
        spacer2 = np.arange(1, num_episodes[j], greedy_interval[j])
        
        print("\n Minimum Steps:", min(step_count))
        
        
        #%%
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
        ax2.plot(spacer2, avg_greedy_step_count, color=color,  label="Greedy Eval.")
        for x,y in zip(spacer2, avg_greedy_step_count):
            ax2.annotate('%s' % y, xy=(x,y), textcoords = 'data')
        ax2.tick_params(axis='y', labelcolor=color)
        plt.title('SARSA-King-Wind-GW alp=%f' % alpha[i])
        plt.legend(loc="upper right")
        plt.savefig('SARSA-King-Wind-GW-new_{}.png'.format(experiment_number), dpi = 1000)
        
        #%%
        plt.figure()
        plt.title("Greedy Evaluation Batches")
        plt.xlabel("Greedy Episodes")
        plt.ylabel("Greedy Steps")
        for k in range(0, np.shape(greedy_step_count)[0]):
            running_avg_greedy_step_count = moving_average(greedy_step_count[k,:,0], n = 15)
            spacer3 = np.arange(0, len(running_avg_greedy_step_count))
            plt.plot(spacer3, running_avg_greedy_step_count, label = "Batch={}".format(k))
        plt.legend()
        plt.savefig("SARSA-King-Windy-GW-greedy_episodes.png", dpi = 1000)
        
        experiment_number += 1