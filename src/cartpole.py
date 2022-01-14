import gym
import random
import logging
import numpy as np
from tqdm import trange
from imt_agent import IMTAgent
from qtable_agent import QAgent


class CartpoleWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(CartpoleWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Tuple(
            spaces=(gym.spaces.Discrete(20),
                    gym.spaces.Discrete(20),
                    gym.spaces.Discrete(50),
                    gym.spaces.Discrete(50)))

        # bin values from https://github.com/JackFurby/CartPole-v0/blob/master/cartPole.py
        self.bins = [
            np.linspace(-4.8, 4.8, 20),
            np.linspace(-4, 4, 20),
            np.linspace(-.418, .418, 50),
            np.linspace(-4, 4, 50)
        ]

    def observation(self, obs):
        state = [np.digitize(state_value, self.bins[i]) -
                 1 for i, state_value in enumerate(obs)]
        return np.array(state)

    def uniform_random_action(self):
        return self.action_space.sample()

    def valid_moves(self):
        return [True]*self.env.action_space.n

def run(Agent, SEED = 42069, EPISODES = 1000, MAX_STEPS = 5000, RENDER = False):
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    np.random.seed(SEED)

    env = CartpoleWrapper(gym.make('CartPole-v0'))
    env.seed(SEED)
    env.action_space.seed(SEED)

    obs_size = len(env.observation_space)
    action_size = env.action_space.n

    agent = Agent(obs_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.1)
    rewards = []
    max_time = 0

    progress_bar = trange(EPISODES)
    for i in progress_bar:
        state = env.reset()
        done = False
        timer = 0
        while not done or i < MAX_STEPS:
            state, reward, done, _ = agent.step(env, state)
            rewards.append(reward)

            if RENDER:
                env.render()

            if done:
                break

            timer += 1
        
        progress_bar.set_description(f'Survived for {timer} steps')
        if max_time < timer:
            max_time = timer
    print(f'Longest time: {max_time}')
    return rewards

if __name__ == '__main__':
    rewards = run(IMTAgent, RENDER=True)
