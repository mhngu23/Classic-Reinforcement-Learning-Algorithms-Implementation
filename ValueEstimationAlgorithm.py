import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from tqdm import trange

import utils

class ValueEstimationAlgorithmTester:
	
	def __init__(self, env, env_name):
		"""
		Parameters
		----------
			env: Testing env object
		"""
		self.env = env
		if env_name not in  ("TaxiEnv-v3", "CartPoleEnv-v0", "Pendulum-v1"): 
			self.nS = self.env.nrow * self.env.ncol
			self.nA = 4
			self.P = self.env.P
		elif self.env in ("Pendulum-v1", "CartPoleEnv-v0"):
			print(self.env)
		elif env_name == "TaxiEnv-v3":
			self.nS = 500
			self.nA = 6
			self.P = self.env.P


	def policy_evaluation(self, policy, change_in_value_function_list, gamma=0.95, tol=1e-3):
		"""Evaluate the value function from a given policy.
		Parameters
		----------
		gamma: float
			Discount factor. Number in range [0, 1)
		policy: np.array[nS]
			The policy to evaluate. Maps states to actions.
		tol: float
			Terminate policy evaluation when
				max |value_function(s) - prev_value_function(s)| < tol
		Returns
		-------
		value_function: np.ndarray[nS]
			The value function of the given policy, where value_function[s] is
			the value of state s
		"""

		value_function = np.zeros(self.nS)
		while True:
			delta = 0
			new_value_function = np.copy(value_function)

			for state in range(self.nS):
				action = policy[state]

				new_value_function[state] = sum([rew + prob * gamma * value_function[next_s] for (prob, next_s, rew, _) in self.P[state][action]])

			delta = np.max(np.abs(new_value_function - value_function))

			change_in_value_function_list.append(delta)

			value_function = new_value_function

			if delta < tol:
				break
				
		return value_function, change_in_value_function_list

	

	def policy_improvement(self, value_from_policy, policy, gamma=0.9):
		"""Given the value function from policy improve the policy.

		Parameters
		----------
		gamma: float
			Discount factor. Number in range [0, 1)
		value_from_policy: np.ndarray
			The value calculated from the policy
		policy: np.array
			The previous policy.

		Returns
		-------
		new_policy: np.ndarray[nS]
			An array of integers. Each integer is the optimal action to take
			in that state according to the environment dynamics and the
			given value function.
		"""

		new_policy = np.copy(policy)

		for state in range(self.nS):

			actions_reward = []

			for action in range(self.nA):

				actions_reward.append(sum([rew + prob * gamma * value_from_policy[next_s] for (prob, next_s, rew, _) in self.P[state][action]]))

			new_policy[state] = np.argmax(actions_reward)

		return new_policy


	def policy_iteration(self, gamma=0.9, tol=1e-3):
		"""
		Parameters
		----------
		gamma: float
			Discount factor. Number in range [0, 1)
		tol: float
			tol parameter used in policy_evaluation()
		Returns:
		----------
		value_function: np.ndarray[nS]
		policy: np.ndarray[nS]
		"""

		value_function = np.zeros(self.nS)
		policy = np.zeros(self.nS, dtype=int)

		change_in_value_function_list = []

		while True:
			# Prediction Step
			value_function, change_in_value_function_list = self.policy_evaluation(policy, change_in_value_function_list , gamma, tol)
			# Control Step
			new_policy = self.policy_improvement(value_function, policy, gamma)
			policy_change = (new_policy != policy).sum()
			if policy_change != 0:
				policy = new_policy
			else:
				break

		utils.show_result(change_in_value_function_list, None)
		
		return value_function, policy

	def value_iteration(self, gamma=0.95, tol=1e-3):
		"""
		Parameters:
		----------
		gamma: float
			Discount factor. Number in range [0, 1)
		tol: float
			Terminate value iteration when
				max |value_function(s) - prev_value_function(s)| < tol
		Returns:
		----------
		value_function: np.ndarray[nS]
		policy: np.ndarray[nS]
		"""

		value_function = np.zeros(self.nS)
		policy = np.zeros(self.nS, dtype=int)
		change_in_value_function_list = []

		# Prediction Step
		while True:
			new_value_function = np.copy(value_function)

			for state in range(self.nS):
				actions_reward = []
				
				for action in range(self.nA):

					actions_reward.append(sum([rew + prob * gamma * value_function[next_state] for (prob, next_state, rew, _) in self.P[state][action]]))

				# Get the value of each update   
				new_value_function[state] = np.max(actions_reward)
			
			
			delta = np.max(np.abs(new_value_function - value_function))

			change_in_value_function_list.append(delta)

			value_function = new_value_function

			if delta < tol:
				break
			
		# Control Step
		for state in range(self.nS):
			actions_reward = []
			for action in range(self.nA):
					actions_reward.append(sum([rew + prob * gamma * value_function[next_state] for (prob, next_state, rew, _) in self.P[state][action]]))

			policy[state] = np.argmax(actions_reward)

		utils.show_result(change_in_value_function_list, None)
		
		return value_function, policy

	def SARSA(self, alpha=0.1, gamma=0.95, epsilon=0.5, n_episodes=100):
		"""
		Parameters:
		----------
		alpha: float
			Learning rate factor. Number in range [0, 1]
		gamma: float
			Discount factor. Number in range [0, 1]
		epsilon: float
			Greedy rate. Number in range [0, 1]
		n_episodes: int
			number of training episodes.
		Returns:
		----------
		value_function: np.ndarray[nS]
		policy: np.ndarray[nS]
		"""
		value_function = np.zeros((self.nS, self.nA))
		policy = np.zeros(self.nS, dtype=int)
		t = trange(n_episodes)
		episode_reward_list = []
		
		for i in t:
			new_value_function = np.copy(value_function)

			# Prediction Step
			ob, _ = self.env.reset()                 # Init first state randomly
			action = utils.epsilon_greedy(new_value_function, epsilon, self.nA, ob)
			# Updating dictionary
			done = False
			episode_reward = 0        
			while not done:
				
				next_ob, rew, done, _, _ = self.env.step(action)
				next_action = utils.epsilon_greedy(new_value_function, epsilon, self.nA, next_ob)
				new_value_function[ob, action] += alpha *  (rew + gamma * new_value_function[next_ob, next_action] - new_value_function[ob, action])
				episode_reward += rew
				if done is True:
					t.set_description(f'Training Episode {i} Reward {episode_reward}')
					# Update the list of reward receive at the end of each episode
					episode_reward_list.append(episode_reward)   
					t.refresh()
					break

				ob, action = next_ob, next_action		

			value_function = new_value_function

		self.env.close() 
		
		# Control Step
		for s in range(self.nS):     
			policy[s] = np.argmax(new_value_function[s, :])

		utils.show_result(None, episode_reward_list)
		
		
		return value_function, policy, episode_reward_list

	def Q_Learning(self, alpha=0.1, gamma=0.95, epsilon=0.9, n_episodes=100):
		"""
		Parameters:
		----------
		alpha: float
			Learning rate factor. Number in range [0, 1]
		gamma: float
			Discount factor. Number in range [0, 1]
		epsilon: float
			Greedy rate. Number in range [0, 1]
		n_episodes: int
			number of training episodes.
		Returns:
		----------
		value_function: np.ndarray[nS]
		policy: np.ndarray[nS]
		"""
		value_function = np.zeros((self.nS, self.nA))
		policy = np.zeros(self.nS, dtype=int)
		t = trange(n_episodes)
		episode_reward_list = []
		
		for i in t:
			new_value_function = np.copy(value_function)

			# Prediction Step
			ob, _ = self.env.reset()                 # Init first state randomly
		
			done = False
			episode_reward = 0
			
			while not done:
				action = utils.epsilon_greedy(new_value_function, epsilon, self.nA, ob)
				next_ob, rew, done, _, _ = self.env.step(action)
				new_value_function[ob, action] += alpha *  (rew + gamma * np.max(new_value_function[next_ob]) - new_value_function[ob, action])
				episode_reward += rew
				if done is True:
					t.set_description(f'Training Episode {i} Reward {episode_reward}')
					# Update the list of reward receive at the end of each episode
					episode_reward_list.append(episode_reward)   
					t.refresh()
					break
				ob = next_ob
			value_function = new_value_function

		self.env.close() 
		# Control Step
		for s in range(self.nS):     
			policy[s] = np.argmax(new_value_function[s, :])

		utils.show_result(None, episode_reward_list)
		
		return value_function, policy, episode_reward_list
	

	


