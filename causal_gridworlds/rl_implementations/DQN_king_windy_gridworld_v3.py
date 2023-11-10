# -*- coding: utf-8 -*-
"""
Created on Mon Nov 8 15:48:36 2023

@author: SSubhnil
@details: DQN code for King Windy GridWorld_v4 - Trying simpler data handling
"""
import math
import matplotlib



import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from custom_envs.king_windy_gridworld_env import KingWindyGridWorldEnv
import gym
import wandb

wandb.login(key="576d985d69bfd39f567224809a6a3dd329326993")
wandb.init(project="Sweep-DQN-King-Windy-GW-2x32", mode="disabled")
env = KingWindyGridWorldEnv()

np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using device:", device)

num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# grid_dimensions = (env.grid_height, env.grid_width)


# Define the DQN class
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim):
        super(DQN, self).__init__()
        self.fci = nn.Linear(state_size, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc34 = nn.Linear(hidden_dim, hidden_dim)
        self.fcf = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = torch.relu(self.fci(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.tanh(self.fc3(x))
        x = torch.relu(self.fcf(x))
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
        return states, actions, rewards, next_states, dones

    def unpack(self):
        batch = self.memory
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.cat(states).to(device),
            torch.tensor(actions).to(device),
            torch.tensor(np.mean(rewards) / (np.std(rewards) + 1e-5)).to(device),
            torch.cat(next_states).to(device),
            torch.tensor(dones).to(device)
        )

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, state_size, action_size, hidden_dim, learning_rate, discount_rate, epsilon, epsilon_decay,
                 buffer_size, batch_size, greedy_interval):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.final_lr = 1e-5
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(buffer_size)
        self.policy_net = DQN(state_size, action_size, hidden_dim).to(device)
        self.target_net = DQN(state_size, action_size, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.total_reward_per_param = 0
        self.greedy_interval = greedy_interval

    def choose_greedy_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        q_values = self.policy_net(state)
        return torch.argmax(q_values).item()

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.push(state, action, reward, next_state, done)

    def replay(self, percent_completion):
        if len(self.replay_memory) == self.buffer_size:
            for i in range(int(self.buffer_size / self.batch_size)):
                states, actions, rewards, next_states, dones = self.replay_memory.sample(self.batch_size)

                states = torch.FloatTensor(states).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                actions = torch.FloatTensor(actions).to(device=device, dtype=int)
                rewards = torch.FloatTensor(rewards).to(device)  #
                dones = torch.FloatTensor(dones).to(device)  # shape = [512]

                current_q_values = (self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)).to(device)
                next_q_values = self.target_net(next_states).max(1)[0].detach()
                target_q_values = (rewards + self.discount_rate * next_q_values * (1 - dones.float())).to(device)

                loss = self.loss(current_q_values, target_q_values)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.replay_memory.memory = self.replay_memory.memory[100:self.buffer_size]

    def train(self, episodes):
        # Initialize the greedy evaluation
        state = env.reset()

        for episode in range(episodes):

            percent_completion = (episode + 1) / episodes

            state = env.reset()
            state = state[0]
            # state = env.reset()
            done = False

            # Epsilon annealing - exponential decay
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
                # self.epsilon = math.exp(-2 * math.pow(percent_completion, 3.5) / 0.4)

            episode_reward = 0
            greedy_episode_reward = 0

            "Greedy Evaluation Episodes"
            if episode % self.greedy_interval == 0:
                while not done:
                    action = self.choose_action(state)

                    next_state, reward, done, _= env.step(action)


                    greedy_episode_reward += reward
                    state = next_state

                wandb.log({'Evaluation reward': greedy_episode_reward})
                print("Episode: {}/{}, Eval. Reward: {}, Epsilon: {:.2f}, Learning Rate: {}".format(episode + 1, episodes,
                                                                                              greedy_episode_reward,
                                                                                              self.epsilon,
                                                                                              self.learning_rate))

            while not done:
                action = self.choose_action(state)

                next_state, reward, done, _ = env.step(action)

                self.remember(state, action, reward, next_state, done)

                episode_reward += reward
                state = next_state
                self.replay(percent_completion)

            wandb.log({'Reward': episode_reward, 'Epsilon': self.epsilon, \
                       'Learning rate': self.learning_rate})


            self.total_reward_per_param += episode_reward
        return self.total_reward_per_param

# def train_params(config):

state_size = 2
action_size = env.nA
batch_size = 512
buffer_size = 4096
num_episodes = 40000
alpha = 3e-4
discount_rate = 0.98
greedy_interval = 500
epsilon_start = 1.0
epsilon_decay = epsilon_start / num_episodes
hidden_dim = 32

agent = DQNAgent(state_size, action_size, hidden_dim, alpha, discount_rate, \
             epsilon_start, epsilon_decay, buffer_size, batch_size, greedy_interval)

total_reward_per_param = agent.train(num_episodes)

# def main():
#     wandb.init(project="Sweep-DQN-King-Windy-GW-2x32", mode="disabled")
#     total_reward_per_param = train_params(wandb.config)
#     wandb.log({'Total Reward per param': total_reward_per_param})
#
# sweep_configuration = {
#     "method": "random",
#     "metric": {"goal": "maximize", "name": "total_reward_per_param"},
#     "parameters": {
#         "alpha": {"values": [1e-3, 7e-4, 3e-4, 1e-4, 7e-5, 3e-5]}}}
#
# sweep_id = wandb.sweep(sweep=sweep_configuration, project="Sweep-DQN-King-Windy-GW-2x32")
#
# wandb.agent(sweep_id, function = main, count=6)
wandb.finish()
