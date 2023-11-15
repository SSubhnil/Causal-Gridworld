# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:50:24 2023

@author: SSubhnil
@details: Advantage actor-critic for the Gridworld environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

from util.util import PlotUtil
from util.util import RepresentationTools as rpt
from util.static_wind_greedy_evaluations import DQN_GreedyEvaluation as evaluate
from envs.windy_gridworld_env import WindyGridWorldEnv

env = WindyGridWorldEnv()
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
import wandb
wandb.login(key="576d985d69bfd39f567224809a6a3dd329326993")
run = wandb.init(project="A2C-4A-Windy-GW_2x64")

grid_dimensions = (env.grid_height, env.grid_width)

# Define the actor network
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
        x = F.tanh(self.fc2(x))
        x = F.softmax(self.fcf(x), dim=-1)
        return x
    
    # Temperature for softmax output scaling.
    # Closer to 10.0 makes the logits uniform
    # Closer to 0.0 makes logits biased to one value
    def temperature_scaled_softmax(self, logits, temperature = 5.0):
        logits = logits / temperature
        return torch.softmax(logits, dim=0)
    
# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fci = nn.Linear(state_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fcf = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fci(state))
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        value = self.fcf(x)
        return value

# Define a named tuple for experiences
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = list()
        self.buffer_size = buffer_size
        self.position = 0
    
    def add_experience(self, experience):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.buffer_size

    
    def sample_batch(self, batch_size):
     
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

class A2C:
    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, lr_decay, buffer_size, batch_size, gamma=0.98):
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.current_lr_actor = lr_actor
        self.current_lr_critic = lr_critic
        self.lr_decay = lr_decay
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.action_dim = action_dim
        
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action_probs = self.actor(state)
        action = torch.multinomial(action_probs, 1).item()
        # print(q_values)
        # self.Q_table[enco.OneHotDecoder(state)] = q_values # q_value requires grad(). That's why use detach()
        return action
    
    def train(self, state, action, reward, next_state, done):
        # Add experience to the replay buffer
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.add_experience(experience)
        
        # Check if the buffer is filled
        if len(self.replay_buffer.buffer) == self.buffer_size:
            for i in range(int(self.buffer_size/self.batch_size)):
            
                # Sample a batch of experience from the replay buffer
                states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)
                
                states = torch.FloatTensor(states).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                actions = torch.FloatTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device) #
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device) #  shape = [512]
                # print("Shapes: States->{}, N_States->{}, Actions->{}, Rewards->{}, Dones->{}".\
                #       format(states.shape, next_states.shape, actions.shape, rewards.shape, dones.shape))
                
                
                #Calculate advantages
                values = self.critic(states) # shape = [512, 1]
                next_values = self.critic(next_states) # shape = [512, 1]
                delta = rewards + (self.gamma * next_values * (1 - dones)) - values
                advantage = delta.detach()
                
                # Actor loss
                action_probs = self.actor(states)
                # print("Action probs:", action_probs)
                actions = actions.view(-1, 1).long() # Convert actions into int64
                
                log_probs = -torch.log(action_probs.gather(1, actions))
                actor_loss = log_probs * advantage.view(-1, 1)
                # print("Actor loss shape:", actor_loss.shape)
                actor_loss = actor_loss.mean()
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step() 
    
                # Critic loss
                critic_loss = delta.pow(2).mean()
                # print("critic loss:", critic_loss.shape)
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()
    
            # Clear the replay buffer after updating
            self.replay_buffer.buffer.clear()
                
            # try:
            #     wandb.log({'Critic_loss': critic_loss, 'Actor_loss': actor_loss})
            # except:
            #     print("WandB loss log failed")
    
        # wandb.watch(self.actor, self.critic, criterion = critic_loss, log = "all", log_freq = 1000, idx = [0, 1], log_graph = True)
            
# Example usage;
if __name__ == "__main__":
    # Define environment dimensions and action space
    state_dim = 2
    action_dim = env.nA
    hidden_dim = 64
    num_episodes = 30000
    greedy_interval = 5000
    learning_rate_actor = 1e-4
    learning_rate_critic = 1e-3
    lr_actor_final = 1e-7
    lr_decay = abs(learning_rate_actor - lr_actor_final)/num_episodes
    current_lr_actor = learning_rate_actor
    current_lr_critic = learning_rate_critic
    batch_size = 1024
    buffer_size = 1024
    
    greedy_evaluation = evaluate(env, grid_dimensions, device)
    greedy_step_count = np.empty((int(num_episodes/greedy_interval), greedy_evaluation.num_episodes, 1))
    avg_greedy_step_count = np.empty(int(num_episodes/greedy_interval))
    step_count = np.empty((num_episodes, 1))
    
    # Create teh A2C agent
    agent = A2C(state_dim, action_dim, hidden_dim, learning_rate_actor, learning_rate_critic, lr_decay, buffer_size, batch_size)
    
    wandb.watch(agent.actor, log='gradients', log_freq = 500, idx = 1, log_graph = True)
    wandb.watch(agent.critic, log='gradients', log_freq = 500, idx = 2, log_graph = True)
    
    # Training loop (replace this with environment and data)
    for episode in range(num_episodes):
        # Greedy evaluation
        # if episode+1 % greedy_interval == 0:
        #     greedy_step_count[sampling_counter], avg_greedy_step_count[sampling_counter]\
        #         = greedy_evaluation.run_algo(self.learning_rate, self.model)
        #     sampling_counter += 1
        
        state = env.reset()
        # state = env.reset()
        done = False
        
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.train(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
        
        if episode % 500 == 0:
            print("Episode: {}/{}, Reward: {}".format(episode+1, num_episodes, episode_reward))
        step_count[episode, 0] = episode_reward
        
        wandb.log({'Reward':episode_reward, 'Learning rate':current_lr_actor})
        
        # Learning Rate decay -> uncomment to implement
        # for param_group in agent.optimizer_actor.param_groups:
        #     param_group['lr'] = current_lr_actor                   
        # current_lr_actor = current_lr_actor - lr_decay
        
    wandb.finish()
    
    def moving_average(step_count, n = 300):
        running_average = np.cumsum(step_count, dtype=float)
        running_average[n:] = running_average[n:] - running_average[:-n]
        return running_average[n - 1:] / n
        
    running_average = moving_average(step_count)

    experiment_number = 1
    #%%
    # np.save("A2C-GW-Step_count-greedy_eval_h{}_{}.npy".format(hidden_dim, experiment_number), step_count)
    # np.save("A2C-GW-Greedy_Step_count-greedy_eval_h{}_{}.npy".format(hidden_dim, experiment_number), avg_greedy_step_count)
        
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
    plt.title('A2C-4A-GW_2x64 alp=%f' % learning_rate_actor)
    plt.legend(loc="upper right")
    plt.savefig('A2C-4A-GW-2x64_h{}_{}.png'.format(hidden_dim, experiment_number), dpi=600)
    
    # run.log({"A2C-Vanilla-GW-h{}".format(hidden_dim):fig})

    #%%
    # plt.figure()
    # plt.title("Greedy Evaluation Batches")
    # plt.xlabel("Greedy Episodes")
    # plt.ylabel("Greedy Steps")
    # for k in range(0, np.shape(greedy_step_count)[0]):
    #     running_avg_greedy_step_count = moving_average(greedy_step_count[k,:,0], n = 15)
    #     spacer3 = np.arange(0, len(running_avg_greedy_step_count))
    #     plt.plot(spacer3, running_avg_greedy_step_count, label = "Batch={}".format(k))
    # plt.legend()
    # plt.savefig("A2C-4A-GW-greedy_episodes_h{}_{}.png".format(hidden_dim, experiment_number), dpi = 600)
    
    