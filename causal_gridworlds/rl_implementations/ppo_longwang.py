from torch import nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical, Normal
import gymnasium as gym
from tqdm.notebook import tnrange
import numpy as np
import scipy
import wandb
from gymnasium.spaces import Box, Discrete
import os
import random
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers import NormalizeObservation
from causal_gridworlds.custom_envs. PPO_stoch_windy_gridworld_env import StochWindyGridWorldEnv_V3
from causal_gridworlds.util.wind_greedy_evaluations import A2C_GreedyEvaluation


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def combined_shape(length, shape=None):
    """
    Helper function makes sure the shape of experience is correct for the buffer

    Args:
        length (int): _description_
        shape (tuple[int,int], optional): _description_. Defaults to None.

    Returns:
        tuple[int,int]: correct shape
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


# TODO: This buffer cannot recompute GAE. Maybe change it in the future
class PPOBuffer():
    """
    A buffer to store the rollout experience from OpenAI spinningup
    """

    def __init__(self, observation_dim, action_dim, capacity, gamma, lam):
        self.obs_buf = np.zeros(combined_shape(capacity, observation_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(capacity, action_dim), dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.rtg_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32)
        self.logp_buf = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.idx = 0
        self.path_idx = 0
        self.gamma = gamma
        self.lam = lam

    def push(self, obs, act, rew, val, logp):
        assert self.idx < self.capacity
        self.obs_buf[self.idx] = obs
        self.act_buf[self.idx] = act
        self.rew_buf[self.idx] = rew
        self.val_buf[self.idx] = val
        self.logp_buf[self.idx] = logp

        self.idx += 1

    def GAE_cal(self, last_val):
        """Calculate the GAE when an episode is ended

        Args:
            last_val (int): last state value, it is zero when the episode is terminated.
            it's v(s_{t+1}) when the state truncate at t.
        """
        path_slice = slice(self.path_idx, self.idx)
        # to make the deltas the same dim
        rewards = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rewards[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        ### OpenAI spinning up implemetation comment: No ideal, big value loss when episode rewards are large
        # self.rtg_buf[path_slice] = discount_cumsum(rewards, self.gamma)[:-1]

        ### OpenAI stable_baseline3 implementation
        ### in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        ### TD(lambda) estimator, see "Telescoping in TD(lambda)"
        self.rtg_buf[path_slice] = self.adv_buf[path_slice] + self.val_buf[path_slice]

        self.path_idx = self.idx

    def sample(self, minibatch_size, device):
        """This method sample a list of minibatches from the memory

        Args:
            minibatch_size (int): size of minibatch, usually 2^n
            device (object): CPU or GPU

        Returns:
            list: a list of minibatches
        """
        assert self.idx == self.capacity, f'The buffer is not full, \
              self.idx:{self.idx} and self.capacity:{self.capacity}'
        # normalise advantage
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / (np.std(self.adv_buf) + 1e-8)

        inds = np.arange(self.capacity)

        np.random.shuffle(inds)

        data = []
        for start in range(0, self.capacity, minibatch_size):
            end = start + minibatch_size
            minibatch_inds = inds[start:end]
            minibatch = dict(obs=self.obs_buf[minibatch_inds], act=self.act_buf[minibatch_inds], \
                             rtg=self.rtg_buf[minibatch_inds], adv=self.adv_buf[minibatch_inds], \
                             logp=self.logp_buf[minibatch_inds])
            data.append({k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in minibatch.items()})

        return data

    def reset(self):
        # reset the index
        self.idx, self.path_idx = 0, 0


def layer_init(layer, std=np.sqrt(2)):
    """Init the weights as the stable baseline3 so the performance is comparable.
       But it is not the ideal way to initialise the weights.

    Args:
        layer (_type_): layers
        std (_type_, optional): standard deviation. Defaults to np.sqrt(2).

    Returns:
        _type_: layers after init
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class Actor_Net(nn.Module):
    def __init__(self, n_observations, n_actions, num_cells, continous_action, log_std_init=0.0, layer_num=3):
        super(Actor_Net, self).__init__()

        self.continous_action = continous_action
        self.action_dim = n_actions

        layers = []
        input_dim = n_observations
        for _ in range(layer_num):
            layers.append(layer_init(nn.Linear(input_dim, num_cells)))
            layers.append(nn.LayerNorm(num_cells))
            layers.append(nn.Tanh())
            input_dim = num_cells
        layers.append(layer_init(nn.Linear(num_cells, n_actions), std=0.01))

        self.network = nn.Sequential(*layers)

        if self.continous_action:
            log_std = log_std_init * np.ones(self.action_dim, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=True)

    def forward(self, x):
        return self.network(x)

    def act(self, x):
        if self.continous_action:
            mu = self.forward(x)
            std = torch.exp(self.log_std)
            dist = Normal(mu, std)
        else:
            log_probs = F.log_softmax(self.forward(x), dim=1)
            dist = Categorical(log_probs)

        action = dist.sample()
        if self.continous_action:
            action_logprob = dist.log_prob(action).sum(axis=-1)
        else:
            action_logprob = dist.log_prob(action)

        return action, action_logprob

    def logprob_ent_from_state_acton(self, x, act):
        if self.continous_action:
            mu = self.forward(x)
            std = torch.exp(self.log_std)
            dist = Normal(mu, std)
            act_logp = dist.log_prob(act).sum(axis=-1)
        else:
            dist = Categorical(F.softmax(self.forward(x), dim=1))
            act_logp = dist.log_prob(act)
        entropy = dist.entropy()

        return entropy, act_logp


class Critic_Net(nn.Module):
    def __init__(self, n_observations, num_cells, layer_num=3):
        super(Critic_Net, self).__init__()

        layers = []
        input_dim = n_observations
        for _ in range(layer_num):
            layers.append(layer_init(nn.Linear(input_dim, num_cells)))
            layers.append(nn.LayerNorm(num_cells))
            layers.append(nn.Tanh())
            input_dim = num_cells
        layers.append(layer_init(nn.Linear(num_cells, 1), std=1.0))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Actor_Critic_net(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, continous_action, continous_observation, parameters_hardshare,
                 log_std_init=0.0, layer_num=3):
        super(Actor_Critic_net, self).__init__()

        self.parameters_hardshare = parameters_hardshare
        self.continous_action = continous_action
        self.continous_observation = continous_observation
        self.action_dim = act_dim
        self.obs_dim = obs_dim

        if self.parameters_hardshare:
            layers = []
            input_dim = obs_dim
            for _ in range(layer_num):
                layers.append(layer_init(nn.Linear(input_dim, hidden_dim)))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.Tanh())
                input_dim = hidden_dim

            self.shared_network = nn.Sequential(*layers)
            self.actor_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
            self.critic_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

            if self.continous_action:
                log_std = log_std_init * np.ones(self.action_dim, dtype=np.float32)
                self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=True)
        else:
            self.actor = Actor_Net(obs_dim, act_dim, hidden_dim, continous_action, log_std_init, layer_num)
            self.critic = Critic_Net(obs_dim, hidden_dim, layer_num)

    def forward(self, x):
        if not self.continous_observation:
            x = F.one_hot(x.long(), num_classes=self.obs_dim).float()
        if self.parameters_hardshare:
            x = self.shared_network(x)
            actor_logits = self.actor_head(x)
            value = self.critic_head(x)
        else:
            actor_logits = self.actor(x)
            value = self.critic(x)
        return actor_logits, value

    def get_value(self, x):
        if not self.continous_observation:
            x = F.one_hot(x.long(), num_classes=self.obs_dim).float()
        return self.critic(x).item()

    def act(self, x):
        if self.continous_action:
            mu, value = self.forward(x)
            log_std = self.log_std if self.parameters_hardshare else self.actor.log_std
            std = torch.exp(log_std)
            dist = Normal(mu, std)
        else:
            actor_logits, value = self.forward(x)
            log_probs = F.log_softmax(actor_logits, dim=-1)
            dist = Categorical(log_probs)

        action = dist.sample()
        if self.continous_action:
            action_logprob = dist.log_prob(action).sum(axis=-1)
        else:
            action_logprob = dist.log_prob(action)

        return action, action_logprob, value

    def logprob_ent_from_state_acton(self, x, action):
        if self.continous_action:
            mu, value = self.forward(x)
            log_std = self.log_std if self.parameters_hardshare else self.actor.log_std
            std = torch.exp(log_std)
            dist = Normal(mu, std)
            action_logp = dist.log_prob(action).sum(axis=-1)
        else:
            actor_logits, value = self.forward(x)
            log_probs = F.log_softmax(actor_logits, dim=-1)
            dist = Categorical(log_probs)
            action_logp = dist.log_prob(action)
        entropy = dist.entropy().sum(axis=-1)

        return entropy, action_logp, value

    def marginalize_wind(self, env, state, device):
        """ Compute expected state value under wind distribution """
        expected_value = 0
        for wind_effect, prob in zip(
                range(-env.range_random_wind, env.range_random_wind + 1), env.probablities
        ):
            # Simulate next state under wind effect
            next_state = env.clamp_to_grid((state[0] - wind_effect, state[1]))
            next_state_tensor = torch.FloatTensor([next_state]).to(device)
            expected_value += prob * self.critic(next_state_tensor).item()
        return expected_value


class PPO():
    def __init__(self, gamma, lamb, eps_clip, K_epochs, \
                 observation_space, action_space, num_cells, layer_num, \
                 actor_lr, critic_lr, memory_size, minibatch_size, \
                 max_training_iter, cal_total_loss, c1, c2, \
                 early_stop, kl_threshold, parameters_hardshare, \
                 max_grad_norm, device, wind_distribution
                 ):
        """Init

        Args:
            gamma (float): discount factor of future value
            lamb (float): lambda factor from GAE from 0 to 1
            eps_clip (float): clip range, usually 0.2
            K_epochs (in): how many times learn from one batch
            action_space (tuple[int, int]): action space of environment
            num_cells (int): how many cells per hidden layer
            critic_lr (float): learning rate of the critic
            memory_size (int): the size of rollout buffer
            minibatch_size (int): minibatch size
            cal_total_loss (bool): add entropy loss to the actor loss or not
            c1 (float): coefficient for value loss
            c2 (float): coefficient for entropy loss
            kl_threshold (float): approx kl divergence, use for early stop
            parameters_hardshare (bool): whether to share the first two layers of actor and critic
            device (_type_): tf device

        """
        self.gamma = gamma
        self.lamb = lamb
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.max_training_iter = max_training_iter

        self.observation_space = observation_space
        self.action_space = action_space
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size

        self.cal_total_loss = cal_total_loss
        self.c1 = c1
        self.c2 = c2
        self.early_stop = early_stop
        self.kl_threshold = kl_threshold

        self.parameters_hardshare = parameters_hardshare
        self.episode_count = 1
        self.max_grad_norm = max_grad_norm
        self.global_step = 0

        self.wind_distribution_ok = wind_distribution

        self._last_obs = None
        self._episode_reward = 0
        self._early_stop_count = 0

        if isinstance(action_space, Box):
            self.continous_action = True
        elif isinstance(action_space, Discrete):
            self.continous_action = False
        else:
            raise AssertionError(f"action space is not valid {action_space}")

        if isinstance(observation_space, Box):
            self.continous_observation = True
        elif isinstance(observation_space, Discrete):
            self.continous_observation = False
        else:
            raise AssertionError(f"observation space is not valid {observation_space}")

        self.actor_critic = Actor_Critic_net(
            observation_space.shape[0] if self.continous_observation else observation_space.n,
            action_space.shape[0] if self.continous_action else action_space.n,
            num_cells, self.continous_action, self.continous_observation, parameters_hardshare, layer_num=layer_num).to(
            device)

        if parameters_hardshare:
            ### eps=1e-5 follows stable-baseline3
            self.actor_critic_opt = torch.optim.Adam(self.actor_critic.parameters(), lr=actor_lr, eps=1e-5)

        else:
            self.actor_critic_opt = torch.optim.Adam([
                {'params': self.actor_critic.actor.parameters(), 'lr': actor_lr, 'eps': 1e-5},
                {'params': self.actor_critic.critic.parameters(), 'lr': critic_lr, 'eps': 1e-5}
            ])

        self.memory = PPOBuffer(observation_space.shape, action_space.shape, memory_size, gamma, lamb)

        self.device = device

        # These two lines monitor the weights and gradients
        wandb.watch(self.actor_critic.actor, log='all', log_freq=1000, idx=1)
        wandb.watch(self.actor_critic.critic, log='all', log_freq=1000, idx=2)
        # wandb.watch(self.actor_critic, log='all', log_freq=1000)

    def roll_out(self, env):
        """rollout for experience

        Args:
            env (gymnasium.Env): environment from gymnasium
        """

        assert self._last_obs is not None, "No previous observation"

        action_shape = env.action_space.shape
        # Run the policy for T timestep
        for i in range(self.memory_size):
            with torch.no_grad():
                obs_tensor = torch.tensor(self._last_obs, \
                                          dtype=torch.float32, device=self.device).unsqueeze(0)

                action, action_logprob, value = self.actor_critic.act(obs_tensor)
            if self.continous_action:
                action = action.cpu().numpy().reshape(action_shape)
            else:
                action = action.item()

            action_logprob = action_logprob.item()

            # Added for wind_distribution
            if hasattr(env, 'wind_distribution_ok') and env.wind_distribution_ok:
                value = self.actor_critic.marginalize_wind(env, self._last_obs, self.device)
            else:
                value = value.item()

            ### Clipping actions when they are reals is important
            clipped_action = action

            if self.continous_action:
                clipped_action = np.clip(action, self.action_space.low, self.action_space.high)

            next_obs, reward, terminated, truncated, info = env.step(clipped_action)

            self.global_step += 1

            self.memory.push(self._last_obs, action, reward, value, action_logprob)

            self._last_obs = next_obs

            self._episode_reward += reward

            if terminated or truncated:
                if truncated:
                    with torch.no_grad():
                        input_tensor = torch.tensor(next_obs, dtype=torch.float32,
                                                    device=self.device) if self.continous_observation else torch.tensor(
                            next_obs, dtype=torch.long, device=self.device)
                        last_value = self.actor_critic.get_value(input_tensor)
                else:
                    last_value = 0

                self.memory.GAE_cal(last_value)

                self._last_obs, _ = env.reset()

                self.episode_count += 1

                wandb.log({'episode_reward': self._episode_reward}, step=self.global_step)

                self._episode_reward = 0

        with torch.no_grad():
            input_tensor = torch.tensor(next_obs, dtype=torch.float32,
                                        device=self.device) if self.continous_observation else torch.tensor(next_obs,
                                                                                                            dtype=torch.long,
                                                                                                            device=self.device)
            last_value = self.actor_critic.get_value(input_tensor)
        self.memory.GAE_cal(last_value)

    def evaluate_recording(self, env):
        self.actor_critic.eval()

        env_name = env.spec.id

        video_folder = os.path.join(wandb.run.dir, 'videos')

        env = RecordVideo(env, video_folder, name_prefix=env_name)

        obs, _ = env.reset()

        done = False

        action_shape = env.action_space.shape

        while not done:
            obs_tensor = torch.tensor(obs, \
                                      dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = self.actor_critic.act(obs_tensor)

            if self.continous_action:
                action = action.cpu().numpy().reshape(action_shape)
            else:
                action = action.item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs = next_obs

        mp4_files = [file for file in os.listdir(video_folder) if file.endswith(".mp4")]

        for mp4_file in mp4_files:
            wandb.log({'Episode_recording': wandb.Video(os.path.join(video_folder, mp4_file))})

        env.close()

    def compute_loss(self, data):
        """compute the loss of state value, policy and entropy

        Args:
            data (List[Dict]): minibatch with experience

        Returns:
            actor_loss : policy loss
            critic_loss : value loss
            entropy_loss : mean entropy of action distribution
        """
        observations, actions, logp_old = data['obs'], data['act'], data['logp']
        advs, rtgs = data['adv'], data['rtg']

        # Calculate the pi_theta (a_t|s_t)
        entropy, logp, values = self.actor_critic.logprob_ent_from_state_acton(observations, actions)
        ratio = torch.exp(logp - logp_old)
        # Kl approx according to http://joschu.net/blog/kl-approx.html
        kl_apx = ((ratio - 1) - (logp - logp_old)).mean()

        clip_advs = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advs
        # Torch Adam implement tation mius the gradient, to plus the gradient, we need make the loss negative
        actor_loss = -(torch.min(ratio * advs, clip_advs)).mean()

        values = values.flatten()  # I used squeeze before, maybe a mistake

        if self.wind_distribution_ok:
            values = torch.FloatTensor([
                self.actor_critic.marginalize_wind(env, state, self.device) for state in data['obs']
            44444]).to(self.device)

        critic_loss = F.mse_loss(values, rtgs)
        # critic_loss = ((values - rtgs) ** 2).mean()

        entropy_loss = entropy.mean()

        return actor_loss, critic_loss, entropy_loss, kl_apx

    def optimise(self):

        entropy_loss_list = []
        actor_loss_list = []
        critic_loss_list = []
        kl_approx_list = []

        # for _ in tnrange(self.K_epochs, desc=f"epochs", position=1, leave=False):
        for _ in range(self.K_epochs):

            # resample the minibatch every epochs
            data = self.memory.sample(self.minibatch_size, self.device)

            for minibatch in data:

                actor_loss, critic_loss, entropy_loss, kl_apx = self.compute_loss(minibatch)

                entropy_loss_list.append(-entropy_loss.item())
                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                kl_approx_list.append(kl_apx.item())

                if self.cal_total_loss:
                    total_loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy_loss

                ### If this update is too big, early stop and try next minibatch
                if self.early_stop and kl_apx > self.kl_threshold:
                    self._early_stop_count += 1
                    ### OpenAI spinning up uses break as they use fullbatch instead of minibatch
                    ### Stable-baseline3 uses break, which is questionable as they drop the rest
                    ### of minibatches.
                    continue

                self.actor_critic_opt.zero_grad()
                if self.cal_total_loss:
                    total_loss.backward()
                    # Used by stable-baseline3, maybe more important for RNN
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.actor_critic_opt.step()

                else:
                    actor_loss.backward()
                    critic_loss.backward()
                    # Used by stable-baseline3, maybe more important for RNN
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.actor_critic_opt.step()

        self.memory.reset()
        # Logging, use the same metric as stable-baselines3 to compare performance
        with torch.no_grad():
            if self.continous_action:
                mean_std = np.exp(self.actor_critic.actor.log_std.mean().item())
                wandb.log({'mean_std': mean_std})

        wandb.log(
            {
                'actor_loss': np.mean(actor_loss_list),
                'critic_loss': np.mean(critic_loss_list),
                'entropy_loss': np.mean(entropy_loss_list),
                'KL_approx': np.mean(kl_approx_list)
            }, step=self.global_step
        )
        if self.early_stop:
            wandb.run.summary['early_stop_count'] = self._early_stop_count

    def train(self, env):
        self.actor_critic.train()

        self._last_obs, _ = env.reset()

        for i in tnrange(self.max_training_iter // self.memory_size):
            self.roll_out(env)

            self.optimise()

        # save the model to the wandb run folder
        # PATH = os.path.join(wandb.run.dir, "actor_critic.pt")
        # torch.save(self.actor_critic.state_dict(), PATH)

        wandb.run.summary['total_episode'] = self.episode_count

sweep_configuration = {
    'method': 'bayes',
    'metric': {'goal': 'maximize', 'name': 'episode_reward'},
    'parameters': {
        'actor_lr': {'values': [1e-4, 3e-4, 1e-3]},
        'critic_lr': {'values': [1e-4, 3e-4, 1e-3]},
        'memory_size': {'values': [1024, 2048, 4096]},
        'k_epochs': {'values': [5, 10, 20]},
        'gamma': {'values': [0.95, 0.99]},
        'lam': {'values': [0.90, 0.95]},
        'early_stop': {'value': False},
        'cal_total_loss': {'value': True},
        'parameters_hardshare': {'value': False},
        'seed': {'value': 43201},
        'c1': {'values': [0.1, 0.5, 1.0]},
        'c2': {'values': [0, 0.01, 0.1]},
        'minibatch_size': {'values': [32, 64, 128]},
        'kl_threshold': {'value': 0.15},
        'max_grad_norm': {'values': [0.5, 1.0]},
        'eps_clip': {'values': [0.1, 0.2, 0.3]},
        'hidden_dim': {'values': [64, 128, 256]},
        'layer_num': {'values': [2, 3, 4]},
        'max_iterations': {'distribution': 'int_uniform', 'min': 50_000, 'max': 300_000},
        # 'max_iterations': {'value': 50000},
        'env_name': {'value': 'Taxi-v3'},
    },

    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 20,
        'max_iter': 100,
        's': 2
    }
}


def main(debug=False, env_name='Walker2d-v4'):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    if debug:
        run = wandb.init(
            project='PPO-test',
            mode='disabled',
            # config = sweep_configuration
        )
        gamma = 0.99
        lamb = 0.95
        eps_clip = 0.2
        max_training_iter = 1_000_000
        k_epochs = 10
        num_cells = 64
        layer_num = 3
        actor_lr = 3e-4
        critic_lr = actor_lr
        memory_size = 2048
        minibatch_size = 64
        c1 = 0.5
        c2 = 0
        kl_threshold = 0.15
        env_name = env_name
        parameters_hardshare = False
        early_stop = False
        cal_total_loss = False
        max_grad_norm = 0.5
        wind_distribution_ok = False
        seed = 123456

        wandb.config.update(
            {
                'actor_lr': actor_lr,
                'critic_lr': critic_lr,
                'gamma': gamma,
                'lambda': lamb,
                'eps_clip': eps_clip,
                'max_training_iter': max_training_iter,
                'k_epochs': k_epochs,
                'hidden_cell_dim': num_cells,
                'layer_num': layer_num,
                'memory_size': memory_size,
                'minibatch_size': minibatch_size,
                'c1': c1,
                'c2': c2,
                'kl_threshold': kl_threshold,
                'env_name': env_name,
                'early_stop': early_stop,
                'parameters_hardshare': parameters_hardshare,
                'early_stop': early_stop,
                'cal_total_loss': cal_total_loss,
                'max_grad_norm': max_grad_norm,
                'wind_distribution_ok': wind_distribution_ok,
                'seed': seed
            }, allow_val_change=True
        )
    else:
        run = wandb.init()
        gamma = wandb.config.gamma
        lamb = wandb.config.lam
        k_epochs = wandb.config.k_epochs
        actor_lr = wandb.config.actor_lr
        critic_lr = wandb.config.critic_lr
        memory_size = wandb.config.memory_size
        minibatch_size = wandb.config.minibatch_size
        c1 = wandb.config.c1
        c2 = wandb.config.c2
        kl_threshold = wandb.config.kl_threshold
        env_name = wandb.config.env_name
        parameters_hardshare = wandb.config.parameters_hardshare
        early_stop = wandb.config.early_stop
        cal_total_loss = wandb.config.cal_total_loss
        max_grad_norm = wandb.config.max_grad_norm
        wind_distribution_ok = wandb.config.wind_distribution_ok
        seed = wandb.config.seed
        eps_clip = wandb.config.eps_clip
        num_cells = wandb.config.hidden_dim
        layer_num = wandb.config.layer_num
        max_training_iter = wandb.config.max_iterations
        seed = wandb.config.seed

    wandb.config.update(
        {
            'implementation': 'my_ppo'
        }
    )
    # wandb.define_metric("episode_reward", summary="mean")
    # wandb.define_metric("KL_approx", summary="mean")

    # Using render_mode slow the training process down
    # env = gym.make(env_name)
    # recording_env = gym.make(env_name, render_mode='rgb_array_list')

    # Windy Gridworld Custom Environment

    # Seeding for evaluation purpose
    env.np_random = np.random.default_rng(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    recording_env.np_random = np.random.default_rng(seed)
    recording_env.action_space.seed(seed)
    recording_env.observation_space.seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Deterministic operations for CuDNN, it may impact performances
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    my_ppo = PPO(gamma, lamb, eps_clip, k_epochs, env.observation_space, env.action_space, num_cells, layer_num, \
                 actor_lr, critic_lr, memory_size, minibatch_size, max_training_iter, \
                 cal_total_loss, c1, c2, early_stop, kl_threshold, parameters_hardshare, max_grad_norm, device,
                 wind_distribution_ok)

    my_ppo.train(env)
    my_ppo.evaluate_recording(recording_env)

    env.close()
    recording_env.close()
    run.finish()


# %env "WANDB_NOTEBOOK_NAME" "PPO_GYM"
sweep_id = wandb.sweep(sweep=sweep_configuration, project='test')
wandb.agent(sweep_id, function=main, count=100)