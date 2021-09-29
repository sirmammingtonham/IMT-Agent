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
        self.experiential_model = Q_Table(self.obs_size+2, self.action_size, alpha, gamma, epsilon)

    # function for running through the episodes
    def run(self):
        # for episode in range(self.num_episodes):
        environmental_state = self.env.reset()
        environmental_state = environmental_state.flatten()

        emotive_state = 0
        arousal_state = 0

        states = environmental_state + [emotive_state, arousal_state]

        attentional_state = np.random.randint(0, self.max_attentional_state)
        affordance_state = np.random.randint(0, self.max_affordance_state)
        subgoal_state = np.random.choice(np.arange(self.obs_size))

        while True:
            # experiential model
            attentional_mask = [int(i) for i in bin(attentional_state)[2:]] # convert to decimal to binary bitmask (there might be a faster way to do this)
            affordance_mask = [int(i) for i in bin(affordance_state)[2:]]

            experiential_state = states * attentional_mask
            experiental_action = self.experiential_model.get_action(experiential_state, self.env.action_space.sample()) # dont know how to incorporate subgoal, gonna ignore for now
            experiental_action *= affordance_mask

            next_environmental_state, environmental_reward, done, info = self.env.step(experiental_action)
            next_environmental_state = next_environmental_state.flatten()

            next_states = next_environmental_state + [emotive_state, arousal_state]
            
            # update qs
            self.experiential_model.Q[states, experiental_action] = self.experiential_model.calculate_q(states, experiental_action, subgoal_reward, next_states)
            self.past_qs.appendleft(self.experiential_model.Q[states, experiental_action])

            # affective system
            if len(self.past_qs) == self.past_qs.maxlen:
                first_derivative = torch.autograd.grad(list(self.past_qs), environmental_reward)
                second_derivative = torch.autograd.grad(first_derivative, environmental_reward)
                emotive_state = first_derivative / self.past_qs.maxlen
                arousal_state = second_derivative / self.past_qs.maxlen
                emotive_state = np.digitize(emotive_state, 100)
                arousal_state = np.digitize(arousal_state, 100)

            # goal model
            subgoal_action = self.goal_model.get_action(next_states, np.random.choice([-1, 1]))
            subgoal_state += subgoal_action # (should be -1 or +1)
            subgoal_reward = 0 # probably need some goal state reward thing
            self.goal_model.Q[states, subgoal_action] = self.experiential_model.calculate_q(states, subgoal_action, environmental_reward, next_states)

            # attention model
            attentional_action = self.attentional_model.get_action(next_environmental_state, np.random.choice([-1, 1]))
            attentional_state += attentional_action
            attentional_state = max(attentional_state, self.max_attentional_state)
            self.attentional_model.Q[environmental_state, attentional_action] = self.experiential_model.calculate_q(states, attentional_action, environmental_reward, next_environmental_state)

            # affordance model
            affordance_action = self.affordance_model.get_action(next_states, np.random.choice([-1, 1]))
            affordance_state += affordance_action
            affordance_state = max(affordance_state, self.max_affordance_state)
            self.affordance_model.Q[states, affordance_action] = self.experiential_model.calculate_q(states, affordance_action, environmental_reward, next_states)

