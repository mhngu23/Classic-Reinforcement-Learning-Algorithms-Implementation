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

class Network1(nn.Module):
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

class Network2(nn.Module):
    def __init__(self, env):
        super().__init__()
        observation_space = env.observation_space
		# Retrieve the number of observations
        num_observations = observation_space.shape[0] 
        n_actions = env.action_space.n
        self.layer1 = nn.Linear(num_observations, 128)
        self.layer2 = nn.Tanh()
        self.layer3 = nn.Linear(128, 1)
	
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.layer3(x)
        return x
        

class ActorCriticAlgorithmTester:
	def __init__(self, env):
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

		self.rew_buffer = deque([0, 0], maxlen=100)
		self.episode_reward = 0
		
		# Create the network/policy
		self.policy = Network1(env).to(device)
		self.state_value = Network2(env).to(device)

		
		self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=5e-4)
		self.state_value_optimizer = torch.optim.Adam(self.state_value.parameters(), lr=5e-4)
						
		self.main_iteration(env)
	
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
		if torch.cuda.is_available():
			num_episodes = 10000
		else:
			num_episodes = 50
			
		mean_reward_list = []
		t = trange(num_episodes)

		for i_episode in t:
			rewards = []
			I = 1
			state, _ = env.reset()
			for step in itertools.count():
				action, log_prob = self.policy.act(state)

				# Execute action and observe result
				new_state, rew, done, _, _ = env.step(action)
				rewards.append(rew)
				
                # Get the value of current state and next state
				state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
				new_state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)

				state_value = self.state_value(state_t)
				new_state_value = self.state_value(new_state_t)
				
				self.episode_reward += rew
				if done:
					# state, _ = env.reset() 
					
					break
				
				val_loss = F.mse_loss(rew + GAMMA * new_state_value, state_value) * I
       
				advantage_function = rew + GAMMA * new_state_value.item() - state_value.item()
				policy_loss = -log_prob.to(device) * advantage_function * I
				# policy_loss = policy_loss.sum()


				# Gradient Descent
				self.state_value_optimizer.zero_grad()
				val_loss.backward()
				self.state_value_optimizer.step()	

				self.policy_optimizer.zero_grad()
				policy_loss.backward(retain_graph=True)
				self.policy_optimizer.step()
				
							
                # Update state with new_state
				state = new_state	
				I *= GAMMA
			self.rew_buffer.append(self.episode_reward)
			# Logging
			print(f"\nEpisode reward is {self.episode_reward} and Average episode reward is {np.mean(self.rew_buffer)}") 
			mean_reward_list.append(np.mean(self.rew_buffer))	
			self.episode_reward = 0			
				
		utils.plot_change_in_each_step(None, mean_reward_list)