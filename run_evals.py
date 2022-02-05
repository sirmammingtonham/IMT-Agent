import os
import gym
import numpy as np
import matplotlib.pyplot as plt

import argparse

from stable_baselines3 import DQN, DDPG, A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter

from agents.imt_agent import IMTAgent
from agents.qtable_agent import QAgent

str_to_model = {
    'dqn': DQN,
    'ddpg': DDPG,
    'a2c': A2C,
    'ppo': PPO
}


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()


def evaluate_baseline(modelstr, policy, environment, num_episodes=1000, **kwargs):
    log_dir = f'./logs/{modelstr}_{environment}/'
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(gym.make(environment), f'{log_dir}/{num_episodes}_episodes')
    model = str_to_model[modelstr](
        policy, env, verbose=1, **kwargs).learn(num_episodes)

    results_plotter.plot_results(
        [log_dir], None, results_plotter.X_EPISODES, environment)
    plot_results(log_dir, title=f'{modelstr}_{environment} learning curve')

    eval_env = gym.make(environment)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

def evaluate_qtable(environment, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=1000):
    log_dir = f'./logs/imt_{environment}/'
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(gym.make(environment), f'{log_dir}/{num_episodes}_episodes')
    agent = QAgent(env, alpha, gamma, epsilon)
    agent.run(num_episodes)

    results_plotter.plot_results(
        [log_dir], None, results_plotter.X_EPISODES, environment)
    plot_results(log_dir, title=f'imt_{environment} learning curve')

    agent.env = gym.make(environment)
    eval_rewards = agent.run(num_episodes=100)
    print(f"mean_reward:{np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}")

def evaluate_imt(environment, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=1000):
    log_dir = f'./logs/imt_{environment}/'
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(gym.make(environment), f'{log_dir}/{num_episodes}_episodes')
    agent = IMTAgent(env, alpha, gamma, epsilon)
    agent.run(num_episodes)

    results_plotter.plot_results(
        [log_dir], None, results_plotter.X_EPISODES, environment)
    plot_results(log_dir, title=f'imt_{environment} learning curve')

    agent.env = gym.make(environment)
    eval_rewards = agent.run(num_episodes=100)
    print(f"mean_reward:{np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}")
    

def main(args):
    np.random.seed(args.seed)
    baselines = [
        (('dqn', args.policy, args.environment, args.episodes), {}),
        # (('ddpg', args.policy, args.environment),{}),
        (('a2c', args.policy, args.environment, args.episodes), {}),
        (('ppo', args.policy, args.environment, args.episodes), {}),
    ]

    for args, kwargs in baselines:
        evaluate_baseline(*args, **kwargs)

    evaluate_qtable(args.environment, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=args.episodes)
    evaluate_imt(args.environment, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=args.episodes)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=69420, type=int, help='Seed')
parser.add_argument('--environment', default='CartPole-v1', type=str, help='OpenAI Gym Environment')
parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy Type')
parser.add_argument('--episodes', default=1000, type=int, help='Number of episodes to train for')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
