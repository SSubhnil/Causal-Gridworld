# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:37:51 2023

@author: SSubhnil
@details: DQN code for Windy GridWorld_v3
"""
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

from util.util import PlotUtil
from util.util import RepresentationTools as rpt
from util.static_wind_greedy_evaluations import DQN_GreedyEvaluation as evaluate
from envs.windy_gridworld_env import WindyGridWorldEnv

import wandb
wandb.login(key="576d985d69bfd39f567224809a6a3dd329326993")
wandb.init(
    project="4A-Windy-GW-Vanilla")

env = WindyGridWorldEnv()

np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using device:", device)

# num_gpus = torch.cuda.device_count()
# print(f"Number of available GPUs: {num_gpus}")

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
grid_dimensions = (env.grid_height, env.grid_width)

# Define the DQN class
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_size
        self.fci = nn.Linear(state_size, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fcf = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        if len(x.size()) == 2:
            x = torch.relu(self.fci(x))
            x = torch.relu(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            # x = torch.tanh(self.fc3(x))
            # x = torch.relu(self.fc4(x))
            x = self.fcf(x)
        else:
            x = torch.relu(self.fci(x.view(-1, self.state_size)))
            torch.relu(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            # x = torch.tanh(self.fc3(x))
            # x = torch.relu(self.fc4(x))
            x = self.fcf(x)        
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.cat(states).to(device),
            torch.tensor(actions).to(device),
            torch.tensor(rewards).to(device),
            torch.cat(next_states).to(device),
            torch.tensor(dones).to(device)
        )
    
    def unpack(self):
        batch = self.memory
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.cat(states).to(device),
            torch.tensor(actions).to(device),
            torch.tensor(np.mean(rewards)/(np.std(rewards)+1e-5)).to(device),
            torch.cat(next_states).to(device),
            torch.tensor(dones).to(device)
        )

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_dim, learning_rate, discount_rate, epsilon, epsilon_decay, greedy_interval):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.greedy_interval = greedy_interval
        self.epsilon_min = 0.01
        self.final_lr = 1e-6
        self.replay_memory = ReplayMemory(512)
        self.model = DQN(state_size, action_size, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.Q_table = torch.empty((env.grid_height, env.grid_width, env.nA))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = self.model(state)
            # print(q_values)
            # self.Q_table[enco.OneHotDecoder(state)] = q_values # q_value requires grad(). That's why use detach()
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float32).to(device)#.reshape(1, -1), dtype=torch.float32).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).to(device)#.reshape(1, -1), dtype=torch.float32).to(device)
        self.replay_memory.push(state, action, reward, next_state, done)

    def replay(self, batch_size, percent_completion):
        if len(self.replay_memory) < batch_size:
            return
        
        # random.shuffle(self.replay_memory.memory)
        states, actions, rewards, next_states, dones = self.replay_memory.sample(batch_size)
        

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1).to(device)
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + self.discount_rate * next_q_values * (1 - dones.float()).to(device)
        
        loss = self.loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()        

    def train(self, env, episodes, batch_size):
        # Initialize the greedy evaluation
        # state = enco.OneHotEncoder(env.reset())
        state = env.reset()
        
        greedy_evaluation = evaluate(env, grid_dimensions, device)
        greedy_step_count = np.empty((int(episodes/self.greedy_interval), greedy_evaluation.num_episodes, 1))
        avg_greedy_step_count = np.empty(int(episodes/self.greedy_interval))
        step_count = np.empty((episodes, 1))
        
        sampling_counter = 0
        lr_decay = (self.learning_rate - self.final_lr) / episodes
        for episode in range(episodes):
            
            # Greedy evaluation
            if episode+1 % greedy_interval == 0:
                greedy_step_count[sampling_counter], avg_greedy_step_count[sampling_counter]\
                    = greedy_evaluation.run_algo(self.learning_rate, self.model)
                sampling_counter += 1
            
            percent_completion = (episode+1)/episodes
            
            state = env.reset()
            # state = env.reset()
            done = False
            
            # Epsilon annealing - exponential decay
            if self.epsilon > self.epsilon_min:
                # self.epsilon -= self.epsilon_decay
                self.epsilon = math.exp(-2*math.pow(percent_completion,3.5)/0.4)
            
            # Learning rate annealing - linear decay
            if self.learning_rate >= self.final_lr:
                self.learning_rate -= lr_decay
            episode_reward = 0
            step_counter = 0
            while not done:
                action = self.choose_action(state)
                
                next_state, reward, done, _ = env.step(action)
                # next_state = enco.OneHotEncoder(next_state)
                
                self.remember(state, action, reward, next_state, done)
                episode_reward += reward
                step_counter += 1
                state = next_state
                self.replay(batch_size, percent_completion)
            
            wandb.log({'Reward':episode_reward,'Steps/episode':step_counter,'Epsilon':self.epsilon,\
                      'Learning rate':self.learning_rate})
            step_count[episode, 0] = step_counter
            print("Episode: {}/{}, Reward: {}, Epsilon: {:.2f}, Learning Rate: {}".format(episode+1, episodes, episode_reward, self.epsilon, self.learning_rate))
        
        return self.Q_table, step_count, greedy_step_count, avg_greedy_step_count


def moving_average(step_count, n = 300):
    running_average = np.cumsum(step_count, dtype=float)
    running_average[n:] = running_average[n:] - running_average[:-n]
    return running_average[n - 1:] / n

state_size = 2
print(state_size)
action_size = env.nA
batch_size = 256
num_episodes = 5000
alpha = 1e-3
discount_rate = 0.98
greedy_interval = 1000
epsilon_start = 0.9
epsilon_decay = epsilon_start/num_episodes
hidden_dim = 32
agent = DQNAgent(state_size, action_size, hidden_dim, alpha, discount_rate,\
                 epsilon_start, epsilon_decay, greedy_interval)

Q_table, step_count, greedy_step_count, avg_greedy_step_count = agent.train(env, num_episodes, batch_size)

running_average = moving_average(step_count)

experiment_number = 101

np.save("DQN-Vanilla-GW-Step_count-greedy_eval_h256p_{}.npy".format(experiment_number), step_count)
np.save("DQN-Vanilla-GW-Greedy_Step_count-greedy_eval_h256_{}.npy".format(experiment_number), avg_greedy_step_count)
    
#avg_step_count = np.average(mega_step_count, axis=0)
spacer1 = np.arange(1, len(running_average)+1)
spacer2 = np.arange(1, num_episodes, greedy_interval)

print("\n 0: Up, 1: Right, 2: Down, 3: Left")
print("\n Minimum: Steps:", min(step_count))

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
ax2.plot(spacer2, avg_greedy_step_count, color=color, label="Greedy Eval.")
for x,y in zip(spacer2, avg_greedy_step_count):
    ax2.annotate('%s' % y, xy=(x,y), textcoords = 'data')
ax2.tick_params(axis='y', labelcolor=color)

# plt.xlabel('Episodes')
# plt.ylabel('Running Average (steps/episode)')
# plt.legend('Min_step', min(step_count))
plt.title('DQN-Vanilla-GW alp=%f' % alpha)
plt.legend(loc="upper right")
plt.savefig('DQN-Vanilla-GW-test_h256_{}.png'.format(experiment_number), dpi=600)

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
plt.savefig("DQN-Vanilla-GW-greedy_episodes_h256_{}.png".format(experiment_number), dpi = 600)

