import os, sys
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/GymGo')

import gym
import random
import logging
import numpy as np
from tqdm import trange
from imt_agent import IMTAgent
from qtable_agent import QAgent

class GoWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(GoWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Tuple(
            spaces=(gym.spaces.Discrete(3),)*25)

    def observation(self, obs):
        new_obs = np.zeros(25)
        # print(obs.flatten())
        for ind, ob in enumerate(obs.flatten()[:25]):
            if new_obs[ind] == 0 and ob == 1:
                new_obs[ind] = 1

        for ind, ob in enumerate(obs.flatten()[25:50]):
            if new_obs[ind] == 0 and ob == 1:
                new_obs[ind] = -1

        return new_obs

def run(black, white, SEED = 42069):
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    # Go test with qAgents running against each other
    EPISODES = 500
    SIZE = 5
    RENDER = False
    np.random.seed(SEED)
    

    env = GoWrapper(gym.make('gym_go:go-v0', size=SIZE,
                             komi=0, reward_method='heuristic'))
    env.seed(SEED)
    env.action_space.seed(SEED)

    obs_size = len(env.observation_space)
    action_size = env.action_space.n

    black = black(obs_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.1)
    black_rewards = []

    white = white(obs_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.1)
    white_rewards = []

    game_status = []

    progress_bar = trange(EPISODES)
    for _ in progress_bar:
        state = env.reset()
        done = False
        while not done:
            state, reward, done, _ = black.step(env, state)
            black_rewards.append(reward)

            if done:
                break

            state, reward, done, _ = white.step(env, state)
            white_rewards.append(reward)

        game_status.append(reward)
        win = [x for x in game_status if x > 0]
        tie = [x for x in game_status if x == 0]
        loss = [x for x in game_status if x < 0]
        progress_bar.set_description(f'WIN%: {len(win)/len(game_status):.2f}, TIE%: {len(tie)/len(game_status):.2f}, LOSS%: {len(loss)/len(game_status):.2f}')
        if RENDER:
            env.render('terminal')

    win = [x for x in game_status if x > 0]
    tie = [x for x in game_status if x == 0]
    loss = [x for x in game_status if x < 0]

    return win, tie, loss

if __name__ == '__main__':
    win_list, tie_list, loss_list = [], [], []

    for _ in range(50):
        SEED = random.randint(0, 2**32 - 1)

        win, tie, loss = run(IMTAgent, QAgent, SEED)
        
        win_list.append(win)
        tie_list.append(tie)
        loss_list.append(loss)

    print(f'WIN%: {sum(win_list)/len(win_list):.2f}')
    print(f'TIE%: {sum(tie_list)/len(tie_list):.2f}')
    print(f'LOSS%: {sum(loss_list)/len(loss_list):.2f}')