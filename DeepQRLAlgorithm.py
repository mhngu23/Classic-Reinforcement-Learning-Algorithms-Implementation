import random
from collections import deque
import itertools

import torch
import numpy as np
from tqdm import trange

from torch import nn
import torch.nn.functional as F

import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA=0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON = 0.1
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
NUM_STEPS = 1000

	
class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        observation_space = env.observation_space
		# Retrieve the number of observations
        num_observations = observation_space.shape[0] 
        n_actions = env.action_space.n
        self.layer1 = nn.Linear(num_observations, 128)
        self.layer2 = nn.Tanh()
        self.layer3 = nn.Linear(128, n_actions)
	
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.layer3(x)
        return x
	
    def act(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device) # Adding another dimension to the torch array with unsqueeze
        Q = self.forward(state).cpu() 
        max_q_index = torch.argmax(Q, dim=1)[0]
        action = max_q_index.detach().item()
        return action 

class DeepQRLAlgorithmTester:
    
	def __init__(self, env):
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

		self.replay_memory_D = deque(maxlen=BUFFER_SIZE)
		self.rew_buffer = deque([0, 0], maxlen=100)
		self.episode_reward = 0
		
		# Create the two networks
		self.online_net = Network(env).to(device)
		self.target_net = Network(env).to(device)
		self.target_net.load_state_dict(self.online_net.state_dict())	# Load saved state of online network
		
		self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=5e-4)
		

		self.__init_replay_memory__(env)
				
		self.main_iteration(env)
		
		
	def __init_replay_memory__(self, env):
		"""
		Initialize self.replay_memory_D memory dataset consists of MIN_REPLAY_SIZE numbers of sample.
		Example: 
			if MIN_REPLAY_SIZE = 1000, self.replay_memory_D_memory consists of 1000 tuple (state, action, rew, done, new_states)

		"""
		state, _ = env.reset()
		for _ in range(MIN_REPLAY_SIZE):
			action = env.action_space.sample()	# Return a random action
			new_states, rew, done, _, _ = env.step(action)
			transition = (state, action, rew, done, new_states)
			self.replay_memory_D.append(transition)
			state = new_states

			if done:
				# If done before MIN_REPLAY_SIZE then restart
				state, _ = env.reset()  
	
	def __epsilon_greedy_policy__(self, env, state, step):
		"""
		Epsilon greedy policy function.
		"""
		random_prob = np.random.random()

		epsilon =np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
		# epsilon = EPSILON
		     
		if random_prob <= epsilon:
			# Explore
			action = env.action_space.sample()
		else:
			# Exploit
			action = self.online_net.act(state)
		return action
	
	def __sample_from_replay_memory__(self):
		transitions = random.sample(self.replay_memory_D, BATCH_SIZE)

		states = np.asarray([t[0] for t in transitions])
		actions = np.asarray([t[1] for t in transitions])
		rewards = np.asarray([t[2] for t in transitions])
		dones = np.asarray([t[3] for t in transitions])
		new_states = np.asarray([t[4] for t in transitions])

		states_t = torch.as_tensor(states, dtype=torch.float32).to(device)
		actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).to(device)
		rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
		dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)
		new_states_t = torch.as_tensor(new_states, dtype=torch.float32).to(device)

		return states_t, actions_t, rewards_t, dones_t, new_states_t


	def main_iteration(self, env):
		if torch.cuda.is_available():
			num_episodes = 500
		else:
			num_episodes = 50
		mean_reward_list = []
		
		t = trange(num_episodes)

		for i_episode in t:
			state, _ = env.reset()
			for step in itertools.count():
				# With prob epsilon select a action
				action = self.__epsilon_greedy_policy__(env, state, step)
				
				# Execute action and observe result
				new_states, rew, done, _, _ = env.step(action)
				transition = (state, action, rew, done, new_states)
				
				# Store transition in replay_memory
				self.replay_memory_D.append(transition)
				state = new_states
				self.episode_reward += rew

				if done:
					break

				# Sample random minibatch equals to BATCH_SIZE
				states_t, actions_t, rewards_t, dones_t, new_states_t = self.__sample_from_replay_memory__()

				# Compute Targets
				target_q_values = self.target_net.forward(new_states_t) # Calculate the Q value using the network used to calculate target value
				max_target_q_values = target_q_values.max(dim=1, keepdim = True)[0] # Only keep the maximum value between actions

				targets = rewards_t + GAMMA * (1-dones_t) * max_target_q_values

				# Compute Currents
				q_values = self.online_net.forward(states_t)
				action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

				# Compute Loss
				loss = nn.functional.smooth_l1_loss(action_q_values, targets) # Similar to MSE Loss

				# Gradient Descent
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				# Update target network
				if step % TARGET_UPDATE_FREQ == 0:
					self.target_net.load_state_dict(self.online_net.state_dict())

				self.rew_buffer.append(self.episode_reward)
				# Logging
				print(f"\nEpisode reward is {self.episode_reward} and Average episode reward is {np.mean(self.rew_buffer)}") 
				mean_reward_list.append(np.mean(self.rew_buffer))
				self.episode_reward = 0	
			

		utils.plot_change_in_each_step(None, mean_reward_list)

				


			

	

	