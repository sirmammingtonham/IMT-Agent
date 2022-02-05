import gym
import numpy as np
import gym_mtsim
import random
import logging
from tqdm import trange
# from imt_agent import IMTAgent
# from qtable_agent import QAgent
from stable_baselines3 import PPO


class TradeWrapper(gym.ObservationWrapper):
    class TradeObservation():
        def __init__(self, obs):
            self.obs = obs

        def tobytes(self):
            return b'|'.join([value.tobytes() for value in self.obs])

    def __init__(self, env=None):
        super(TradeWrapper, self).__init__(env)
        self.bins = {
            'balance': np.linspace(0, 1000000, num=10000),
            'equity': np.linspace(0, 1000000, num=10000),
            'margin': np.linspace(0, 1000000, num=10000),
            'features': np.linspace(0, 1000, num=10000),
            'orders': np.linspace(0, 1000, num=10000)
        }

    def observation(self, obs):
        return self.TradeObservation((np.digitize(value, bins=self.bins[key])
                              for key, value in obs.items()))

    def uniform_random_action(self):
        return np.array([np.random.choice((-1.0, 0.0, 1.0)) for _ in range(self.env.action_space.shape[0])])

    def valid_moves(self):
        return [True]*self.env.action_space.shape[0]

# broken
def run(Agent, SEED = 42069, EPISODES = 1000, MAX_STEPS = 5000, RENDER = False):
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    np.random.seed(SEED)

    env = TradeWrapper(gym.make('stocks-hedge-v0'))
    env.seed(SEED)

    obs_size = len(env.observation_space)
    action_size = 1

    agent = Agent(obs_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.1)
    rewards = []

    progress_bar = trange(EPISODES)
    for i in progress_bar:
        state = env.reset()
        done = False
        while not done or i < MAX_STEPS:
            state, reward, done, _ = agent.step(env, state)
            rewards.append(reward)

            if RENDER:
                env.render()

            if done:
                break
        
        progress_bar.set_description(f'Reward: {reward}')
    return rewards

def run_baseline(policy='MultiInputPolicy', RENDER=False):
    env = gym.make('stocks-hedge-v0')
    model = PPO(policy, env, verbose=1)
    model.learn(total_timesteps=1000)

    observation = env.reset()    
    if RENDER:
        print(env.render())

    rewards = []

    while True:
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)

        if RENDER:
            print(env.render())

        if done:
            break

    return rewards


if __name__ == '__main__':
    rewards = run_baseline('MultiInputPolicy', RENDER=True)
    print(rewards)

    rewards_lstm = run_baseline('MultiInputPolicy', RENDER=True)
    print(rewards)