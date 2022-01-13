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

def run(black, white, SEED = 42069, EPISODES = 10000):
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

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
        black_win = [x for x in game_status if x > 0]
        tie = [x for x in game_status if x == 0]
        white_win = [x for x in game_status if x < 0]
        progress_bar.set_description(f'BLACK WIN%: {len(black_win)/len(game_status):.2f}, TIE%: {len(tie)/len(game_status):.2f}, WHITE WIN%: {len(white_win)/len(game_status):.2f}')
        if RENDER:
            env.render('terminal')

    # if isinstance(black, IMTAgent):
    #     imt = black
    #     qtable = white
    # else:
    #     qtable = black
    #     imt = white
    # print('IMT REVISIT %:', imt.experiential_model.revisit_counter / imt.experiential_model.update_counter)
    # print('QTABLE REVISIT %:', qtable.model.revisit_counter / qtable.model.update_counter)

    print('BLACK QTABLE REVISIT %:', black.model.revisit_counter / black.model.update_counter)
    print('WHITE QTABLE REVISIT %:', white.model.revisit_counter / white.model.update_counter)

    black_win = [x for x in game_status if x > 0]
    tie = [x for x in game_status if x == 0]
    white_win = [x for x in game_status if x < 0]

    return len(black_win), len(tie), len(white_win)

if __name__ == '__main__':
    wins, ties, losses = 0, 0, 0
    ITERATIONS = 2
    EPISODES = 1000000
    SEED = 240

    # run with imt agent first
    black_wins, tie, white_wins = run(QAgent, QAgent, SEED, EPISODES)

    wins += black_wins
    ties += tie
    losses += white_wins

    # run with imt agent second
    # black_wins, tie, white_wins = run(QAgent, IMTAgent, SEED, EPISODES)

    # wins += white_wins
    # ties += tie
    # losses += black_wins

    total = EPISODES*1
    print()
    print(f'IMT_AGENT WIN%: {wins/total:.2f}')
    print(f'TIE%: {ties/total:.2f}')
    print(f'IMT_AGENT LOSS%: {losses/total:.2f}')