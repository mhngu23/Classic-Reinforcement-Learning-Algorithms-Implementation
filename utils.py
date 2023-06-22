import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def epsilon_greedy(value_function, epsilon, n_actions, s):
    """
    value_function: Q Table
    epsilon: exploration parameter
    n_actions: number of actions
    s: state
    """
    # selects a random action with probability epsilon
    random_prob = np.random.random()
    # Explore
    if random_prob <= epsilon:
        return np.random.randint(n_actions)
    # Exploit
    else:
        return np.argmax(value_function[s, :])
    
def updating_dictionary(value, dictionary):
    if value in dictionary.keys():
                dictionary[value] += 1
    else:
        dictionary[value] = 1
    return dictionary

def plot_change_in_each_step(change_in_value_function = None, reward = None):
    if change_in_value_function is not None:
        df = pd.DataFrame({'step': [i for i in range(len(change_in_value_function))],
                    'change_in_value_function': change_in_value_function})
        plt.figure(1)
        plt.plot(df.step, df.change_in_value_function, c = "orange", label='Change in value function after each step', linewidth=3)
        #add title and axis labels
        plt.title('Change in Value Function after Update Step')
        plt.xlabel('Improvement Step')
        plt.ylabel('Change in value function')
        plt.legend()

    if reward is not None:
        df = pd.DataFrame({'step': [i for i in range(len(reward))],
                    'reward': reward})
        plt.figure(2)
        plt.plot(df.step, df.reward, label='Reward after each step')
        #add title and axis labels
        plt.title('Reward after Update Step')
        plt.xlabel('Improvement Step')
        plt.ylabel('Training Reward')
        plt.legend()

    plt.show()
    return

def compare_training_sarsa_q_learning_through_plotting_training_reward(episode_reward_list_s, episode_reward_list_q):
    if episode_reward_list_s is not None:
        df = pd.DataFrame({'step': [i for i in range(len(episode_reward_list_s))],
                    'episode_reward_list_s': episode_reward_list_s})
        plt.figure(2)
        plt.plot(df.step, df.episode_reward_list_s, label='Reward after each step SARSA')
        #add title and axis labels
        plt.title('Reward after Update Step')
        plt.xlabel('Improvement Step')
        plt.ylabel('Training Reward')
        plt.legend()

    if episode_reward_list_q is not None:
        df = pd.DataFrame({'step': [i for i in range(len(episode_reward_list_q))],
                    'episode_reward_list_q': episode_reward_list_q})
        plt.figure(2)
        plt.plot(df.step, df.episode_reward_list_q, label='Reward after each step Q_Learning')
        #add title and axis labels
        plt.title('Reward after Update Step')
        plt.xlabel('Improvement Step')
        plt.ylabel('Training Reward')
        plt.legend()

    return

def compare_training_sarsa_q_learning_through_heatmap_q_value(Q_pi_SARSA, Q_pi_Q_Learning):
    fig, (ax1, ax2) = plt.subplots(1,2)
    s1 = sns.heatmap(Q_pi_SARSA, ax=ax1)
    s2 = sns.heatmap(Q_pi_Q_Learning, ax=ax2)
    s1.set_title("SARSA Q Value")
    s2.set_title("Q_learning Q Value")
    s1.set(xlabel='Actions', ylabel='States')
    s2.set(xlabel='Actions', ylabel='States')

    sns.set(rc={'figure.figsize':(30,30)})

    return

def compare_training_sarsa_q_learning(episode_reward_list_s = None, episode_reward_list_q = None, Q_pi_SARSA = None, Q_pi_Q_Learning = None):
    print(f"Average reward SARSA is {np.mean(episode_reward_list_s)} and average reward q_learning is {np.mean(episode_reward_list_q)}")
    compare_training_sarsa_q_learning_through_plotting_training_reward(episode_reward_list_s, episode_reward_list_q)
    compare_training_sarsa_q_learning_through_heatmap_q_value(Q_pi_SARSA, Q_pi_Q_Learning)
    plt.show()

    return

def plot_evaluating_result(mean_reward, std_reward, episode_rewards):
    # Print evaluation results
    print(f"Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Min = {min(episode_rewards):.1f} and Max {max(episode_rewards):.1f}")

    # Show the distribution of rewards obtained from evaluation
    plt.figure(figsize=(5,5))
    plt.title(label='Rewards distribution from evaluation', loc='center')
    plt.hist(episode_rewards, bins=25, color='#00000f')
    plt.show()







