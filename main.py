import argparse
import utils 

import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt 

from testing_envs import *
from ValueEstimationAlgorithm import *
from DeepQRLAlgorithm import *
from REINFORCEAlgorithm import *
from Actor_Critic_A2C_Algorithm import *
import warnings

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
    choices=["Deterministic-4x4-FrozenLake-v0", "Stochastic-4x4-FrozenLake-v0", "TaxiEnv-v3", "CartPoleEnv-v0"],
    default="CartPoleEnv-v0",
)

parser.add_argument(
    "--render-mode",
    "-r",
    type=str,
    help="The render mode for the environment. 'human' opens a window to render. 'ansi' does not render anything.",
    choices=["human", "ansi", "rgb_array"],
    default="rgb_array",
)

parser.add_argument(
    "--algorithm",
    "-a",
    type=str,
    help="The type of algorithm that you will be using. classic_RL includes PI, VI, SARSA, Q_learning",
    choices=["classic_RL", "policy_iteration", "value_iteration", "Comparing_Classic", "SARSA",
              "Q_learning", "DQN", "REINFORCE", "A2C"],
    default="A2C",
)

parser.add_argument(
    "--number_testing_sample",
    "-n",
    type=int,
    help="The number of testing samples.",
    default=100,
)



def render_testing(env, policy, DQN = None, REINFORCE = None, A2C = None, max_steps=1000):
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
    env = gym.make(args.env, render_mode="human")

    for seed in range(args.number_testing_sample):
        episode_reward = 0
        ob, _ = env.reset(seed = seed)

        for t in range(max_steps):
            env.render()
            # time.sleep(0.01)
            if DQN is not None:
                a = DQN.online_net.act(ob)
            elif REINFORCE is not None:
                a, _ = REINFORCE.policy.act(ob)        
            elif A2C is not None:
                a, _, _ = A2C.policy.act(ob)                 
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
        else:
            print("Testing episode reward: %f" % episode_reward)
            episode_reward_list_testing.append(episode_reward)
    
    mean_reward = np.mean(episode_reward_list_testing)
    std_reward = np.std(episode_reward_list_testing)
    return mean_reward, std_reward, episode_reward_list_testing

def call_function(args_algorithm = "classic_RL"):
    
    value_estimation_algorithm_tester = ValueEstimationAlgorithmTester(env, args.env)

    if args_algorithm == "policy_iteration":
        print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)
        V_pi, p_pi= value_estimation_algorithm_tester.policy_iteration(gamma=0.95, tol=1e-3)
        mean_reward, std_reward, episode_reward_list_testing = render_testing(env, p_pi)
        utils.plot_evaluating_result(mean_reward, std_reward, episode_reward_list_testing)
    
    # elif args_algorithm == "value_iteration": 
    #     print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    #     V_pi, p_pi = value_estimation_algorithm_tester.value_iteration(gamma=0.95, tol=1e-3)
    #     mean_reward, std_reward, episode_reward_list_testing = render_testing(env, p_pi)
    #     utils.plot_evaluating_result(mean_reward, std_reward, episode_reward_list_testing)
    
    # elif args_algorithm == "SARSA":
    #     print("\n" + "-" * 25 + "\nBeginning SARSA\n" + "-" * 25)
    #     Q_pi, p_pi, _ = value_estimation_algorithm_tester.SARSA(alpha=0.5, gamma=0.95, epsilon=0.1, n_episodes=1000)
    #     mean_reward, std_reward, episode_reward_list_testing = render_testing(env, p_pi)
    #     utils.plot_evaluating_result(mean_reward, std_reward, episode_reward_list_testing)   

    # elif args_algorithm == "Q_learning":
    #     print("\n" + "-" * 25 + "\nBeginning Q_learning\n" + "-" * 25)
    #     Q_pi, p_pi, _ = value_estimation_algorithm_tester.Q_Learning(alpha=0.5, gamma=0.95, epsilon=0.1, n_episodes=1000)
    #     mean_reward, std_reward, episode_reward_list_testing = render_testing(env, p_pi)
    #     utils.plot_evaluating_result(mean_reward, std_reward, episode_reward_list_testing)

    elif args_algorithm == "DQN":
        print("\n" + "-" * 25 + "\nBeginning DQN\n" + "-" * 25)
        deep_q_rl_algorithm_tester = DeepQRLAlgorithmTester(env)
        mean_reward, std_reward, episode_reward_list_testing = render_testing(env, None, deep_q_rl_algorithm_tester)
    
    elif args_algorithm == "REINFORCE":
        print("\n" + "-" * 25 + "\nBeginning REINFORCE\n" + "-" * 25)
        REINFORCE_algorithm_tester = REINFORCEAlgorithmTester(env)
        mean_reward, std_reward, episode_reward_list_testing = render_testing(env, None, None, REINFORCE_algorithm_tester)

    elif args_algorithm == "A2C":
        print("\n" + "-" * 25 + "\nBeginning A2C\n" + "-" * 25)
        Actor_Critic_algorithm_tester = ActorCriticAlgorithmTester(env)
        mean_reward, std_reward, episode_reward_list_testing = render_testing(env, None, None, None, Actor_Critic_algorithm_tester)    
    
    # Testing classic_RL algorithms at once.
    elif args_algorithm == "Comparing_Classic":
        print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)
        V_pi, p_pi = value_estimation_algorithm_tester.policy_iteration(gamma=0.95, tol=1e-3)
        mean_reward, std_reward, episode_reward_list_testing = render_testing(env, p_pi)
        utils.plot_evaluating_result(mean_reward, std_reward, episode_reward_list_testing)

        print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
        V_pi, p_pi = value_estimation_algorithm_tester.value_iteration(gamma=0.95, tol=1e-3) 
        mean_reward, std_reward, episode_reward_list_s = render_testing(env, p_pi)
        utils.plot_evaluating_result(mean_reward, std_reward, episode_reward_list_testing)

        print("\n" + "-" * 25 + "\nBeginning SARSA\n" + "-" * 25)
        Q_pi_SARSA, p_pi, episode_reward_list_s = value_estimation_algorithm_tester.SARSA(alpha=0.1, gamma=0.95, epsilon=0.1, n_episodes=10000)
        mean_reward, std_reward, episode_reward_list_testing = render_testing(env, p_pi)
        utils.plot_evaluating_result(mean_reward, std_reward, episode_reward_list_testing)

        print("\n" + "-" * 25 + "\nBeginning Q_learning\n" + "-" * 25)
        Q_pi_Q_Learning, p_pi, episode_reward_list_q = value_estimation_algorithm_tester.Q_Learning(alpha=0.1, gamma=0.95, epsilon=0.1, n_episodes=10000)
        mean_reward, std_reward, episode_reward_list_testing = render_testing(env, p_pi)
        utils.plot_evaluating_result(mean_reward, std_reward, episode_reward_list_testing) 

        utils.compare_training_sarsa_q_learning(episode_reward_list_s, episode_reward_list_q, Q_pi_SARSA, Q_pi_Q_Learning)

if __name__ == "__main__":
    # read in script argument
    args = parser.parse_args()

    # Make gym environment

    env = gym.make(args.env, render_mode=args.render_mode)

    # Print action spaces and observation spaces
    print("The numbers of action space are: ", env.action_space)
    print("The numbers of observation space are: ", env.observation_space)
    # print(env.P)


    call_function(args.algorithm)






