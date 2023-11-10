import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random

# if GPU is available, use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the named tuple for experiences
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, inputs, outputs, hidden):
        super(DQN, self).__init__()
        self.fci = nn.Linear(inputs, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fco = nn.Linear(hidden, outputs)

    def forward(self, x):
        x = F.relu(self.fci(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fco(x)

# Hyperparameters
BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 0.90
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
LEARNING_RATE = 7e-4
MEMORY_SIZE = 4096
NUM_EPISODES = 20000
HIDDEN_NODES = 32
EVALUATION_REWARD = []
env = gym.make('CartPole-v1').unwrapped
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

policy_net = DQN(n_states, n_actions, HIDDEN_NODES).to(device)
target_net = DQN(n_states, n_actions, HIDDEN_NODES).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)

steps_done = 0

def select_greedy_action(state):
    global steps_done
    with torch.no_grad():
        steps_done += 1
        return policy_net(state).max(-1)[1].view(1, 1).to(device)

def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            return policy_net(state).max(-1)[1].view(1, 1).to(device)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).reshape(-1, 4).to(device)
    state_batch = torch.cat(batch.state).reshape(-1, 4).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    done_batch = torch.cat(batch.done).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    
    expected_state_action_values = (next_state_values * GAMMA * (1 - done_batch.float())) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

episode_rewards = np.zeros(NUM_EPISODES)

for i_episode in range(NUM_EPISODES):
    # Initialize the environment and state
    state = env.reset()
    state = torch.FloatTensor(state[0]).to(device)
    done = False
    t = 0
    episode_reward = 0
    evaluation_reward = 0

    "Evaluation segment"
    if i_episode % 500 == 0:
        while not done:
            # Select and perform an action
            action = select_greedy_action(state)
            next_state, reward, done, _, _ = env.step(action.item())
            next_state = torch.FloatTensor(next_state).to(device)
            reward = torch.tensor([reward], device=device)

            evaluation_reward += reward.item()
            # Move to the next state
            state = next_state
        EVALUATION_REWARD.append(evaluation_reward)
        print("Episode:", i_episode, "; Evaluation Reward:", evaluation_reward)

    "Training segment"
    while not done:
        # Select and perform an action
        action = select_action(state)
        next_state, reward, done, _, _ = env.step(action.item())
        next_state = torch.FloatTensor(next_state).to(device)
        reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        memory.push(state, action, next_state, reward, torch.tensor([done], device=device))

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        episode_reward += reward

    episode_rewards[i_episode] = episode_reward





