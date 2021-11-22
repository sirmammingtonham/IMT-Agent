import gym
import numpy as np
from imt_agent import IMTAgent
from qtable_agent import QAgent
from util.bitmask import BitMask


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


if __name__ == '__main__':
    # Go test with qAgents running against each other
    EPISODES = 500
    SIZE = 5
    RENDER = False

    env = GoWrapper(gym.make('gym_go:go-v0', size=SIZE,
                             komi=0, reward_method='heuristic'))
    obs_size = len(env.observation_space)
    action_size = env.action_space.n

    # setup imt agent
    imt_agent = IMTAgent(obs_size, action_size, alpha=0.1,
                         gamma=0.9, epsilon=0.1)
    imt_rewards = []

    # setup q agent
    q_agent = QAgent(obs_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.1)
    q_rewards = []

    game_status = []
    for episode in range(EPISODES):
        state = env.reset()
        if RENDER:
            env.render()

        done = False
        iterations = 0
        while not done:
            state, reward, done, _ = imt_agent.step(env, state)
            imt_rewards.append(reward)

            if done:
                print('reward:', reward)
                break

            state, reward, done, _ = q_agent.step(env, state)
            q_rewards.append(reward)

            if RENDER:
                env.render()

        game_status.append(reward)
        print(f'finished episode {episode}, result={reward}')

    win = [x for x in game_status if x > 0]
    tie = [x for x in game_status if x == 0]
    loss = [x for x in game_status if x < 0]
    print(f'WIN%: {len(win)/len(game_status)}')
    print(f'TIE%: {len(tie)/len(game_status)}')
    print(f'LOSS%: {len(loss)/len(game_status)}')
    print(f'AVERAGE REWARD: {np.mean(game_status)}')
