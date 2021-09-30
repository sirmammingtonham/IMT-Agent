import numpy as np


class Q_Table():
    def __init__(self, state_space, alpha, gamma, epsilon):
        self.Q = np.zeros(state_space)
        self.alpha = alpha
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon

    def get_alpha(self, timestep=None):
        if timestep is not None:
            return max(self.alpha, min(1.0, 1.0 - np.log10((timestep + 1) / 25)))
        return self.alpha

    def get_epsilon(self, timestep=None):
        if timestep is not None:
            return max(self.epsilon, min(1.0, 1.0 - np.log10((timestep + 1) / 25)))
        return self.epsilon

    def get_action(self, state, random_action):
        if np.random.rand() < self.get_epsilon():
            return random_action # env.action_space.sample() when interacting with gym, but sometimes we want to pass our own actions
        else:
            try:
                return np.argmax(self.Q[state])
            except:
                print(state)
                raise

    def calculate_q(self, state, action, reward, next_state):
        try:
            return self.Q[state][action] + self.get_alpha() * (
                reward + self.gamma * np.max(self.Q[next_state])
                - self.Q[state][action]
            )
        except:
            print(state)
            print(action)
            print(self.Q.shape)
            print(len(self.Q[state]))
            print(self.Q[4,0,0,3])
            print(type(state))
            print(self.Q[tuple(state)][3])

            print(type(action))
            print(action == 3)
            # print(self.Q[state, action])
            print(self.Q[state][action])
            assert(False)