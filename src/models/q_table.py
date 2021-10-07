import numpy as np


class Q_Table():
    def __init__(self, state_space, alpha, gamma, epsilon):
        self.Q = np.zeros(state_space)
        self.alpha = alpha
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon

    def get_alpha(self, timestep=None):
        ## in case we want to implement decay
        # if timestep is not None:
        #     return max(self.alpha, min(1.0, 1.0 - np.log10((timestep + 1) / 25)))
        return self.alpha

    def get_epsilon(self, timestep=None):
        ## in case we want to implement decay
        # if timestep is not None:
        #     return max(self.epsilon, min(1.0, 1.0 - np.log10((timestep + 1) / 25)))
        return self.epsilon

    def get_action(self, state, random_action):
        if np.random.rand() < self.get_epsilon():
            return random_action
        else:
            return np.argmax(self.Q[state])

    def calculate_q(self, state, action, reward, next_state):
        return self.Q[state][action] + self.get_alpha() * (
            reward + self.gamma * np.max(self.Q[next_state])
            - self.Q[state][action]
        )

    def cosine_between_vectors(a , b):
        anorm = np.linalg.norm(x=a)
        bnorm = np.linalg.norm(x=b)
        cos = np.matmul(np.transpose(a/anorm),(b/bnorm))
        return cos
