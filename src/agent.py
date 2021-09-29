import gym
import torch
import numpy as np
from models.q_table import Q_Table
import collections


class Agent():
    def __init__(self, env: gym.Env, alpha: float, gamma: float, epsilon: float, window_size: int=20):
        self.env = env
        self.num_episodes = 3
        
        self.obs_size = np.prod(env.observation_space.shape)
        self.action_size = np.prod(env.action_space.shape)

        self.max_attentional_state = bin('1' * self.obs_size)
        self.max_affordance_state = bin('1' * self.action_size)

        self.past_qs = collections.deque(maxlen=window_size)

        self.goal_model = Q_Table(self.obs_size+2, 2, alpha, gamma, epsilon)
        self.attentional_model = Q_Table(self.obs_size, 2, alpha, gamma, epsilon)
        self.affordance_model = Q_Table(self.obs_size+2, 2, alpha, gamma, epsilon)
        # self.agent_q_table = np.zeros(
        #     (np.prod(env.observation_space.shape) + 2,
        #      np.prod(env.action_space.shape))
        # )

    # function for running through the episodes
    def run(self):
        # for episode in range(self.num_episodes):
        current_state = self.env.reset()
        current_state.flatten()
        attentional_state = np.random.randint(0, self.max_attentional_state)
        affordance_state = np.random.randint(0, self.max_affordance_state)

        while True:
            # affective system
            if len(self.past_qs) == self.past_qs.maxlen:
                first_derivative = torch.autograd.grad(
                    list(self.past_qs), 'idk what variable they take the derivative wrt')
                second_derivative = torch.autograd.grad(first_derivative, )
                emotive_state = first_derivative / self.past_qs.maxlen
                arousal_state = second_derivative / self.past_qs.maxlen
                emotive_state = np.digitize(emotive_state, 100)
                arousal_state = np.digitize(arousal_state, 100)

            # goal model
            if len(self.past_qs) == self.past_qs.maxlen:
                subgoal_action = self.goal_model.get_action(current_state + [emotive_state, arousal_state], np.random.choice([-1, 1]))
                subgoal_state += subgoal_action # (should be -1 or +1)
            else:
                subgoal_state = np.random.choice(np.arange(self.obs_size))
            
            # attention model
            attentional_action = self.attentional_model.get_action(current_state, np.random.choice([-1, 1]))
            attentional_state += attentional_action
            attentional_state = max(attentional_state, self.max_attentional_state)
            attentional_mask = [int(i) for i in bin(attentional_state)[2:]] # convert to decimal to binary bitmask (there might be a faster way to do this)

            # affordance model
            affordance_action = self.affordance_model.get_action(current_state + [emotive_state, arousal_state], np.random.choice([-1, 1]))
            affordance_state += affordance_action
            affordance_state = max(affordance_state, self.max_affordance_state)
            affordance_mask = [int(i) for i in bin(affordance_state)[2:]]

            # experiential model
            experiential_states = current_state * attentional_mask
            # action ← Q Learning with subgoal and Affordance mask for tie breaking
            experiential_action = None
            observation, reward, done, info = self.env.step(experiential_action)



