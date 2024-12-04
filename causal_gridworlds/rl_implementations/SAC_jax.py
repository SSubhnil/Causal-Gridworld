# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:23:19 2023

@author: shubh
@details: SAC for Stochastic Windy Gridworld using JAX and TensorFlow Probability.
"""

import os
import sys
import random
import numpy as np
from collections import namedtuple, deque
import argparse

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# JAX and related libraries
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from flax import linen as nn
from flax.training import train_state
import optax

# TensorFlow Probability
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

# WandB for logging
import wandb

from util.wind_greedy_evaluations import SAC_GreedyEvaluation_JAX
from custom_envs.stoch_windy_gridworld_env_v3 import StochWindyGridWorldEnv_V3
from custom_envs.stoch_king_windy_gridworld_env import StochKingWindyGridWorldEnv

np.random.seed(42)

# Set up WandB
wandb.login(key="576d985d69bfd39f567224809a6a3dd329326993")  # Replace with your actual WandB API key


# Define the Actor network
class Actor(nn.Module):
    action_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, state):
        x = nn.relu(nn.Dense(self.hidden_dim)(state))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.leaky_relu(nn.Dense(self.hidden_dim)(x))
        logits = nn.Dense(self.action_dim)(x)
        return logits  # Return logits; softmax will be applied later


# Define the Critic network
class Critic(nn.Module):
    action_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, state):
        x = nn.relu(nn.Dense(self.hidden_dim)(state))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.leaky_relu(nn.Dense(self.hidden_dim)(x))
        q_values = nn.Dense(self.action_dim)(x)
        return q_values  # Output Q-values for all actions


# Define a named tuple for experiences
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


class SACAgent:
    def __init__(self, env, state_dim, action_dim, hidden_dim,
                 lr_actor, lr_critic, buffer_size, batch_size, entropy_weight,
                 wind_distribution, gamma=0.98, tau=0.005):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau  # For soft update of target networks
        self.batch_size = batch_size
        self.entropy_weight = entropy_weight
        self.wind_distribution_ok = wind_distribution

        # Initialize networks
        self.key = jax.random.PRNGKey(42)
        self.actor = Actor(action_dim, hidden_dim)
        self.critic_1 = Critic(action_dim, hidden_dim)
        self.critic_2 = Critic(action_dim, hidden_dim)
        self.target_critic_1 = Critic(action_dim, hidden_dim)
        self.target_critic_2 = Critic(action_dim, hidden_dim)

        # Initialize parameters
        dummy_state = jnp.ones((1, state_dim))
        self.actor_params = self.actor.init(self.key, dummy_state)
        self.critic_1_params = self.critic_1.init(self.key, dummy_state)
        self.critic_2_params = self.critic_2.init(self.key, dummy_state)
        self.target_critic_1_params = self.critic_1_params
        self.target_critic_2_params = self.critic_2_params

        # Optimizers
        self.actor_tx = optax.adam(lr_actor)
        self.critic_tx = optax.adam(lr_critic)
        self.actor_state = train_state.TrainState.create(apply_fn=self.actor.apply,
                                                         params=self.actor_params,
                                                         tx=self.actor_tx)
        self.critic_1_state = train_state.TrainState.create(apply_fn=self.critic_1.apply,
                                                            params=self.critic_1_params,
                                                            tx=self.critic_tx)
        self.critic_2_state = train_state.TrainState.create(apply_fn=self.critic_2.apply,
                                                            params=self.critic_2_params,
                                                            tx=self.critic_tx)

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state):
        state = jnp.array(state).reshape(1, -1)
        logits = self.actor.apply(self.actor_state.params, state)
        action_probs = nn.softmax(logits)
        action_dist = tfd.Categorical(probs=action_probs)
        action = action_dist.sample(seed=self.key)
        return int(action[0])

    def store_experience(self, experience):
        self.replay_buffer.add_experience(experience)

    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)

        # Convert to JAX arrays
        states = jnp.array(states)
        next_states = jnp.array(next_states)
        actions = jnp.array(actions).reshape(-1, 1)
        rewards = jnp.array(rewards).reshape(-1, 1)
        dones = jnp.array(dones).reshape(-1, 1)

        # Critic update
        key1, key2 = jax.random.split(self.key)
        self.key = key1

        @jit
        def critic_loss_fn(critic_params, target_critic_params, actor_params, states, actions, rewards, next_states,
                           dones):
            def compute_q_target(next_states):
                logits = self.actor.apply(actor_params, next_states)
                action_probs = nn.softmax(logits)
                log_action_probs = nn.log_softmax(logits)
                q1_next = self.target_critic_1.apply(target_critic_params[0], next_states)
                q2_next = self.target_critic_2.apply(target_critic_params[1], next_states)
                q_next = jnp.minimum(q1_next, q2_next)
                q_target = action_probs * (q_next - self.entropy_weight * log_action_probs)
                q_target = q_target.sum(axis=1, keepdims=True)
                return q_target

            if self.wind_distribution_ok:
                expected_q_targets = []
                for wind_effect, prob in zip(
                        np.arange(-self.env.range_random_wind, self.env.range_random_wind + 1),
                        self.env.probabilities
                ):
                    # Adjust next_states for wind effect
                    next_state_wind = np.array([
                        self.env.clamp_to_grid((state[0] - wind_effect, state[1]))
                        for state in next_states
                    ])
                    next_state_wind = jnp.array(next_state_wind)
                    q_target_wind = compute_q_target(next_state_wind)
                    expected_q_targets.append(prob * q_target_wind)
                q_target = sum(expected_q_targets)
                q_target = rewards + (1 - dones) * self.gamma * q_target
            else:
                q_target = compute_q_target(next_states)
                q_target = rewards + (1 - dones) * self.gamma * q_target

            q1 = self.critic_1.apply(critic_params[0], states)
            q2 = self.critic_2.apply(critic_params[1], states)

            q1_pred = jnp.take_along_axis(q1, actions, axis=1)
            q2_pred = jnp.take_along_axis(q2, actions, axis=1)

            critic_1_loss = jnp.mean((q1_pred - q_target) ** 2)
            critic_2_loss = jnp.mean((q2_pred - q_target) ** 2)
            return critic_1_loss + critic_2_loss

        # Update critics
        critic_params = [self.critic_1_state.params, self.critic_2_state.params]
        target_critic_params = [self.target_critic_1_params, self.target_critic_2_params]
        grads = grad(critic_loss_fn)(critic_params, target_critic_params, self.actor_state.params,
                                     states, actions, rewards, next_states, dones)
        self.critic_1_state = self.critic_1_state.apply_gradients(grads=grads[0])
        self.critic_2_state = self.critic_2_state.apply_gradients(grads=grads[1])

        # Actor update
        @jit
        def actor_loss_fn(actor_params, critic_params, states):
            logits = self.actor.apply(actor_params, states)
            action_probs = nn.softmax(logits)
            log_action_probs = nn.log_softmax(logits)
            q1 = self.critic_1.apply(critic_params[0], states)
            q2 = self.critic_2.apply(critic_params[1], states)
            q_min = jnp.minimum(q1, q2)
            actor_loss = (action_probs * (self.entropy_weight * log_action_probs - q_min)).sum(axis=1).mean()
            return actor_loss

        # Update actor
        actor_grads = grad(actor_loss_fn)(self.actor_state.params,
                                          [self.critic_1_state.params, self.critic_2_state.params], states)
        self.actor_state = self.actor_state.apply_gradients(grads=actor_grads)

        # Soft update target critics
        def soft_update(target_params, params):
            return jax.tree_util.tree_map(lambda tp, p: tp * (1 - self.tau) + p * self.tau, target_params, params)

        self.target_critic_1_params = soft_update(self.target_critic_1_params, self.critic_1_state.params)
        self.target_critic_2_params = soft_update(self.target_critic_2_params, self.critic_2_state.params)


def train_params(config):
    # Define environment dimensions and action space
    env = StochKingWindyGridWorldEnv()
    grid_dimensions = (env.grid_height, env.grid_width)
    env.seed(42)
    state_dim = 2
    action_dim = env.nA
    hidden_dim = config["hidden_dim"]
    num_episodes = 40000
    batch_size = config["batch_size"]
    buffer_size = config["buffer_size"]
    learning_rate_actor = config["lr_actor"]
    learning_rate_critic = config["lr_critic"]
    wind_distribution_ok = config["wind_distribution_ok"]
    total_reward_per_param = 0
    entropy_weight = config["entropy_weight"]
    tau = config["tau"]
    gamma = config["gamma"]

    # Create the SAC agent
    agent = SACAgent(env, state_dim, action_dim, hidden_dim, learning_rate_actor,
                     learning_rate_critic, buffer_size, batch_size, entropy_weight,
                     wind_distribution_ok, gamma=gamma, tau=tau)
    # Initialize the greedy evaluation
    greedy_evaluation = SAC_GreedyEvaluation_JAX(env, grid_dimensions)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Select an action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store experience
            experience = Experience(state, action, reward, next_state, done)
            agent.store_experience(experience)

            # Train the agent
            agent.train()
            episode_reward += reward
            state = next_state

        # Log reward for each episode
        wandb.log({"Reward": episode_reward})

        # Periodic logging
        if episode % 500 == 0:
            print(f"Episode: {episode + 1}/{num_episodes}, Reward: {episode_reward}")
            # Runs greedy evaluation
            greedy_rewards, avg_evaluation_reward = greedy_evaluation.evaluate(agent.actor_state.params, agent.actor.apply)
            wandb.log({"Avg. Evaluation Reward": avg_evaluation_reward})

        total_reward_per_param += episode_reward

    return total_reward_per_param

def main(single_run=False):
    if single_run:
        # Perform a single run
        wandb.init(project="SAC-JAX-Stoch-GW-Wind_seen", config={
            "lr_actor": 3e-4,
            "lr_critic": 3e-3,
            "buffer_size": 1024,
            "batch_size": 512,
            "wind_distribution_ok": False,
            "entropy_weight": 0.2,
            "hidden_dim": 128,
            "tau": 0.005,
            "gamma": 0.98,
        }, mode="online")

        config = wandb.config
        total_reward_per_param = train_params(config)
        wandb.log({"Total_Reward": total_reward_per_param})
        wandb.finish()
    else:
        # Sweep logic
        def sweep_main():
            wandb.init()
            config = wandb.config  # Automatically set for each sweep run
            total_reward_per_param = train_params(config)
            wandb.log({"Total_Reward": total_reward_per_param})
            wandb.finish()
        # Perform a sweep
        sweep_configuration = {
            "method": "grid",  # Exhaustive search over all parameter combinations
            "metric": {"goal": "maximize", "name": "Total_Reward"},  # Optimize for total reward
            "parameters": {
                "lr_actor": {"values": [1e-4, 3e-4]},
                "lr_critic": {"values": [1e-3, 3e-3]},
                "buffer_size": {"values": [1024, 2048]},
                "batch_size": {"values": [512, 1024]},
                "wind_distribution_ok": {"values": [False, True]},
                "entropy_weight": {"values": [0.1, 0.2]},
                "hidden_dim": {"values": [128]},
                "tau": {"values": [0.005]},
                "gamma": {"values": [0.98]},
            },
        }

        # Initialize the sweep
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="SAC-JAX-Stoch-GW-Wind_seen")
        wandb.agent(sweep_id, function=sweep_main)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAC training.")
    parser.add_argument("--single_run", action="store_true",
                        help="Run a single training instead of a sweep.")
    args = parser.parse_args()

    main(single_run=args.single_run)