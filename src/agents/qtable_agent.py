import gym
import numpy as np
from models.q_table import Q_Table
from observation_wrapper import ObservationWrapper

# todo convert all of this to pytorch so we can use autograd :)
class QAgent():
    def __init__(self, obs_size: int, action_size: int, alpha: float, gamma: float, epsilon: float):
        self.obs_size = obs_size
        self.action_size = action_size

        self.model = Q_Table(self.action_size, alpha, gamma, epsilon)

    def step(self, env: gym.Env, state: np.array):
        action = self.model.get_action(state, env.uniform_random_action())

        while not env.valid_moves()[action]:
            action = env.uniform_random_action() # force random action if action is illegal
            if isinstance(action, np.ndarray): # break if action is an array( i.e. trading environment, there are no invalid actions)
                break

        next_state, reward, done, info = env.step(action)

        self.model.update_q(state, action, reward, next_state)

        return next_state, reward, done, info