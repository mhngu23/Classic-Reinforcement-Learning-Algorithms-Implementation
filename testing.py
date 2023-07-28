import argparse
import utils 
import warnings
import csv

import torch
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt 

from testing_envs import *
warnings.filterwarnings("ignore", category=DeprecationWarning)


np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(
    description="A program to run assignment 1 implementations.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--env",
    type=str,
    help="The name of the environment to run your algorithm on.",
    choices=["Deterministic-4x4-FrozenLake-v0", "Stochastic-4x4-FrozenLake-v0", "TaxiEnv-v3", ],
    default="CartPoleEnv-v0",
)

parser.add_argument(
    "--number_testing_sample",
    "-n",
    type=int,
    help="The number of testing samples.",
    default=100,
)

def render_testing(env, policy = None, DQN = None, REINFORCE = None, A2C = None, max_steps=10000):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
    """
    episode_reward_list_testing = []
    env = gym.make(args.env, render_mode="rgb_array")
    
    for seed in range(10000, 10000+args.number_testing_sample):
        episode_reward = 0
        ob, _ = env.reset(seed = seed)

        for t in range(max_steps):
            env.render()
            # time.sleep(0.01)
            if DQN is not None:
                a = DQN.act(ob,  0.05)
                name = "dqn_result.csv"
            elif REINFORCE is not None:
                a, _ = REINFORCE.act(ob)
                name = "RF_result.csv"        
            elif A2C is not None:
                a, _, _ = A2C.act(ob)    
                name = "A2C_result.csv"             
            else:
                a = policy[ob]
            ob, rew, done, _, _ = env.step(a)
            episode_reward += rew
            if done:
                break

        env.render()
        if not done:
            print(
                "The agent didn't reach a terminal state in {} steps.".format(
                    max_steps
                )
            )
            episode_reward_list_testing.append(episode_reward)
        else:
            print("Testing episode reward: %f" % episode_reward)
            episode_reward_list_testing.append(episode_reward)
    
    mean_reward = np.mean(episode_reward_list_testing)
    std_reward = np.std(episode_reward_list_testing)

    return mean_reward, std_reward, episode_reward_list_testing

    
if __name__ == "__main__":
    args = parser.parse_args()

    render_testing(env, DQN=torch.load("DQN_model_0.pt"))






