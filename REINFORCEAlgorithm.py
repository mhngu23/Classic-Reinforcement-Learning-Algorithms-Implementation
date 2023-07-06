import random
from collections import deque
import itertools

import torch
import numpy as np
from tqdm import trange

from torch.distributions import Categorical
from torch import nn
import torch.nn.functional as F

import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA=0.99

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        observation_space = env.observation_space
		# Retrieve the number of observations
        num_observations = observation_space.shape[0] 
        n_actions = env.action_space.n
        self.layer1 = nn.Linear(num_observations, 128)
        self.layer2 = nn.Linear(128, n_actions)
	
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return F.softmax(x, dim=1)
	
    def act(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device) # Adding another dimension to the torch array with unsqueeze
        action_probs = self.forward(state).cpu()
        m = Categorical(action_probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
        

class REINFORCEAlgorithmTester:
	def __init__(self, env, train_seed, training_time):
		"""
		Parameters
		----------
			env: Testing env object
		Self
		----------
			self.rew_buffer: A buffer to record all rewards per episode.
			self.episode_reward: Current episode reward.
			self.policy: Main training network.
			self.optimizer: Optimization method.
		"""
		if torch.cuda.is_available() and train_seed is None:
			self.num_episodes = 500
		elif torch.cuda.is_available() and train_seed is not None:
			self.num_episodes = len(train_seed)
		else:
			self.num_episodes = 50
		self.episode_duration_list = []

		for _ in range(training_time):		
			self.episode_reward = 0
			self.reward_by_step = []
			self.train_reward = []
			self.episode_duration = []

			# Create the network/policy
			self.policy = Network(env).to(device)
			
			self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=5e-4)

			self.main_iteration(env)
			self.episode_duration_list.append(self.episode_duration)
			self.episode_duration = []

		# utils.show_result(change_in_training=self.train_reward, algo_name="RF")
		utils.show_result(change_in_training=self.episode_duration_list, algo_name="RF")

	
	def __calculate_discounted_returns__(self, rewards):
		"""
		Function to calculate the discounted return associated with each state
		"""
		returns = deque(maxlen=len(rewards))
		for t in range(len(rewards))[::-1]:
			# If it is the first state then discounted_return_current_state is 0 otherwise it will equal return of previous state * GAMMA
			discounted_return_current_state = returns[0] if len(returns) > 0 else 0
			returns.appendleft(rewards[t] + GAMMA * discounted_return_current_state)
		return returns

	def main_iteration(self, env):
			

		saved_log_probs = []
		rewards = []
		seed = 0
		state, _ = env.reset(seed = seed)

		for step in range(1, 30000):
			action, log_prob = self.policy.act(state)
			saved_log_probs.append(log_prob)

			# Execute action and observe result
			new_states, rew, done, _, _ = env.step(action)
			rewards.append(rew)

			# Update state with new_state
			state = new_states
			self.episode_reward += rew

			if done: 
				print(self.episode_reward+1)
				self.episode_duration.append(self.episode_reward+1)
				self.episode_reward = 0

				# Calculate loss
				discounted_returns = self.__calculate_discounted_returns__(rewards)
				discounted_returns_t = torch.tensor(discounted_returns)

				loss = []
				for log_prob, discounted_return in zip(saved_log_probs, discounted_returns_t):
					loss.append(-log_prob.to(device) * discounted_return.to(device))
				loss = torch.cat(loss).sum()

				# Gradient Descent
				self.optimizer.zero_grad()
				loss.backward(retain_graph=True)
				self.optimizer.step()
				
				saved_log_probs = []
				rewards = []

				seed += 1
				state, _ = env.reset(seed = seed)

		
