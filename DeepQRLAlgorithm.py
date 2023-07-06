import random
from collections import deque
import itertools
import math

import torch
import numpy as np
from tqdm import trange

from torch import nn
import torch.nn.functional as F

import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA=0.99
BATCH_SIZE = 32
BUFFER_SIZE = 1000
MIN_REPLAY_SIZE = 100
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 1000
TARGET_UPDATE_FREQ = 1000

	
class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
		# Retrieve the number of observations
        num_observations = env.observation_space.shape[0] 
        self.n_actions = env.action_space.n
        self.layer1 = nn.Linear(num_observations, 128)
        self.layer2 = nn.Tanh()
        self.layer3 = nn.Linear(128, self.n_actions)
	
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device) # Adding another dimension to the torch array with unsqueeze
            Q = self.forward(state).cpu() 
            max_q_index = torch.argmax(Q, dim=1)[0]
            action = max_q_index.detach().item()
        else:
            action = random.randrange(self.n_actions)
        return action
    
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)


class DeepQRLAlgorithmTester:
    
	def __init__(self, env, train_seed = None, training_time = None):
		"""
		Parameters
		----------
			env: Testing env object
		Self
		----------
			self.replay_memory_D: A dataset to store experiences at each time step, e_t = (s_t, a_t, r_t, s_t+1), D_t = {e_1, ..., e_t}
			self.episode_reward: Current episode reward.
			self.online_net: Main training network.
			self.target_net: To generate y_i 
			self.optimizer: Optimization method
		"""
		if torch.cuda.is_available() and train_seed is None:
			self.num_episodes = 500
		elif torch.cuda.is_available() and train_seed is not None:
			self.num_episodes = len(train_seed)
		else:
			self.num_episodes = 50
		self.epsilon_by_frame = lambda frame_idx: EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * frame_idx / EPSILON_DECAY)

		self.episode_duration_list = []
	
		for _ in range(training_time):		

			self.replay_memory_D = ReplayBuffer(1000)
			self.episode_reward = 0
			self.episode_duration = []

			# Create the two networks
			self.online_net = Network(env).to(device)
			self.target_net = Network(env).to(device)
			self.target_net.load_state_dict(self.online_net.state_dict())	# Load saved state of online network
			
			self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=5e-4)

			self.main_iteration(env)
			self.episode_duration_list.append(self.episode_duration)
			self.episode_duration = []

		utils.show_result(change_in_training=self.episode_duration_list, algo_name = "dqn")

	
	def __sample_from_replay_memory__(self):
		state, action, reward, done, next_state = self.replay_memory_D.sample(batch_size=BATCH_SIZE)

		# states = np.asarray([t[0] for t in transitions])
		# actions = np.asarray([t[1] for t in transitions])
		# rewards = np.asarray([t[2] for t in transitions])
		# dones = np.asarray([t[3] for t in transitions])
		# new_states = np.asarray([t[4] for t in transitions])

		# states_t = torch.as_tensor(states, dtype=torch.float32).to(device)
		# actions_t = torch.LongTensor(actions).to(device)
		# rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
		# dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)
		# new_states_t = torch.as_tensor(new_states, dtype=torch.float32).to(device)

		state_t      = torch.FloatTensor(np.float32(state)).to(device)
		next_state_t = torch.FloatTensor(np.float32(next_state)).to(device)
		action_t     = torch.LongTensor(action).to(device)
		reward_t     = torch.FloatTensor(reward).to(device)
		done_t       = torch.FloatTensor(done).to(device)

		return state_t, action_t, reward_t, done_t, next_state_t


	def main_iteration(self, env):
		
		# t = trange(self.num_episodes)
		# for i_episode in t:
		# 	state, _ = env.reset(seed = i_episode)
		seed = 0
		state, _ = env.reset(seed = seed)
		for step in range(1, 30000):
			epsilon = self.epsilon_by_frame(step)
			# With prob epsilon select a action
			action = self.online_net.act(state, epsilon)
			
			# Execute action and observe result
			new_states, rew, done, _, _ = env.step(action)
			
			# Store transition in replay_memory
			self.replay_memory_D.push(state, action, rew, done, new_states)
			
			state = new_states
			self.episode_reward += rew

			if done:
				print(self.episode_reward+1)
				self.episode_duration.append(self.episode_reward+1)
				self.episode_reward = 0	
								
			if len(self.replay_memory_D) > BATCH_SIZE:

				# Sample random minibatch equals to BATCH_SIZE
				states_t, actions_t, rewards_t, dones_t, new_states_t = self.__sample_from_replay_memory__()

				q_values      = self.online_net.forward(states_t)
				next_q_values = self.target_net.forward(new_states_t)

				q_value          = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
				next_q_value     = next_q_values.max(1)[0]
				expected_q_value = rewards_t + GAMMA * next_q_value * (1 - dones_t)
				
				# loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
				loss = nn.functional.smooth_l1_loss(q_value, expected_q_value) # Similar to MSE Loss

				# Gradient Descent
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				# Update target network
				if step % TARGET_UPDATE_FREQ == 0:
					self.target_net.load_state_dict(self.online_net.state_dict())

			if done:
				seed += 1
				state, _ = env.reset(seed = seed)

		

	
					
				


				


			

	

	