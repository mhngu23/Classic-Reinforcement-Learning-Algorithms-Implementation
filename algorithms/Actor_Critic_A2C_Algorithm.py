import random
from collections import deque
import itertools
import warnings

import torch
import numpy as np
from tqdm import trange

from torch.distributions import Categorical
from torch import nn
import torch.nn.functional as F

import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# warnings.filterwarnings("ignore", category=UserWarning)

GAMMA=0.99

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        observation_space = env.observation_space
		# Retrieve the number of observations
        num_observations = observation_space.shape[0] 
        n_actions = env.action_space.n
        self.layer1 = nn.Linear(num_observations, 128)
        self.action_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)
	
    def forward(self, x):
        x = F.relu(self.layer1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)	
        return action_prob, state_value
	
    def act(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device) # Adding another dimension to the torch array with unsqueeze
        action_probs, state_value = self.forward(state)
        action_probs = action_probs.cpu()
        m = Categorical(action_probs)
        action = m.sample()
        return action.item(), m.log_prob(action), state_value    

class ActorCriticAlgorithmTester:
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
		# if torch.cuda.is_available() and train_seed is None:
		# 	self.num_episodes = 500
		# elif torch.cuda.is_available() and train_seed is not None:
		# 	self.num_episodes = len(train_seed)
		# else:
		# 	self.num_episodes = 50
		
		self.episode_duration_list = []

		for training in range(training_time):		
			self.episode_reward = 0
			self.reward_by_step = []
			self.train_reward = []
			self.episode_duration = []
	
			# Create the network/policy
			self.policy = Network(env).to(device)
			
			self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
		# self.state_value_optimizer = torch.optim.Adam(self.state_value.parameters(), lr=5e-4)
			self.main_iteration(env, training)
			self.episode_duration_list.append(self.episode_duration)
			self.episode_duration = []

			torch.save(self.policy, f"A2C_model_{training}.pt")


		# utils.show_result(change_in_training=self.train_reward, algo_name = "A2C")
		utils.show_result(change_in_training=self.episode_duration_list, algo_name = "A2C")

	
	def 	__calculate_discounted_returns__(self, rewards):
		"""
		Function to calculate the discounted return associated with each state
		"""
		returns = deque(maxlen=len(rewards))
		for t in range(len(rewards))[::-1]:
			# If it is the first state then discounted_return_current_state is 0 otherwise it will equal return of previous state * GAMMA
			discounted_return_current_state = returns[0] if len(returns) > 0 else 0
			returns.appendleft(rewards[t] + GAMMA * discounted_return_current_state)
		return returns

	def main_iteration(self, env, training):
		seed = 0
		rewards = []
		saved_actions = []
		value_losses = []
		policy_losses = []
		state, _ = env.reset(seed = seed)

		for step in range(1, 35000):
			print(f"Training number {training}, Step number {step}")
			action, log_prob, state_value = self.policy.act(state)

			# Execute action and observe result
			new_state, rew, done, _, _ = env.step(action)
			rewards.append(rew)
			saved_actions.append((log_prob, state_value))
			
			# Update state with new_state
			state = new_state	
			
			self.episode_reward += rew
			self.episode_duration.append(self.episode_reward)
			if done:
				# self.episode_duration.append(self.episode_reward+1)
				self.episode_reward = 0	
				
				# Calculate loss
				discounted_returns = self.__calculate_discounted_returns__(rewards)
				discounted_returns_t = torch.tensor(discounted_returns)

				for (log_prob, value), R in zip(saved_actions, discounted_returns_t):
					advantage = R - value.item()

					# calculate actor (policy) loss
					policy_losses.append(-log_prob * advantage)

					# calculate critic (value) loss using L1 smooth loss
					value_losses.append(F.smooth_l1_loss(value.to(device), torch.tensor([[R]]).to(device)))
				
				# sum up all the values of policy_losses and value_losses	
				self.policy_optimizer.zero_grad()
	
				loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()


				loss.backward(retain_graph=True)
				self.policy_optimizer.step()

				saved_actions = []
				rewards = []
				
				seed += 1
				state, _ = env.reset(seed = seed)





	
				
