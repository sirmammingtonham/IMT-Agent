import gym
import collections
import numpy as np
from models.q_table import Q_Table
from util.bitmask import BitMask
import torch

# todo convert all of this to pytorch so we can use autograd :)
class Agent():
    def __init__(self, env: gym.Env, alpha: float, gamma: float, epsilon: float, window_size: int=20, num_episodes: int=3):
        self.env = env
        self.num_episodes = num_episodes
        
        self.obs_size = np.prod(env.observation_space.shape).astype(int)
        self.action_size = np.prod(env.action_space.shape).astype(int)

        # self.max_attentional_state = int('1' * self.obs_size, base=2)
        # self.max_affordance_state = int('1' * self.action_size, base=2)

        self.past_qs = collections.deque(maxlen=window_size)

        self.goal_model = Q_Table((self.obs_size, 100,100, 2), alpha, gamma, epsilon)
        self.attentional_model = Q_Table((self.obs_size, 2), alpha, gamma, epsilon)
        self.affordance_model = Q_Table((self.obs_size, 100,100, 2), alpha, gamma, epsilon)
        self.experiential_model = Q_Table((self.obs_size, 100,100, self.action_size), alpha, gamma, epsilon)


    # function for running through the episodes
    def run(self):
        rewards = []
        for episode in range(self.num_episodes):
            print(f'starting episode {episode}')

            environmental_state = self.env.reset()
            environmental_state = environmental_state.flatten().astype(int)

            subgoal_state = np.random.choice(np.arange(self.obs_size))
            attentional_state = BitMask(self.obs_size, random=True)
            affordance_state = BitMask(self.action_size, random=True)

            affective_states = np.zeros(2) # emotive state, arousal state
            states = np.append( environmental_state, affective_states).astype(int)

            done = False
            iterations = 0
            total_reward = 0
            while not done:
                # affective system
                if len(self.past_qs) == self.past_qs.maxlen:
                    assert(environmental_reward is not None)
                    first_derivative = np.diff(list(self.past_qs)) # first derivative: change in q
                    second_derivative = np.diff(first_derivative) # second derivative: change in first derivative
                    emotive_state = sum(first_derivative)
                    arousal_state = sum(second_derivative)
                    affective_states[0] = round(emotive_state, 2) * 100 # discretize by rounding, then mult by 100 to be index for state
                    affective_states[1] = round(arousal_state, 2) * 100

                # goal model
                subgoal_action = self.goal_model.get_action(states, np.random.choice([-1, 1]))
                subgoal_state += subgoal_action # (should be -1 or +1)

                # attentional model
                attentional_action = self.attentional_model.get_action(environmental_state, np.random.choice([0, 1]))
                attentional_state.add_one() if attentional_action else attentional_state.subtract_one()

                # affordance model
                affordance_action = self.affordance_model.get_action(states, np.random.choice([0, 1]))
                affordance_state.add_one() if attentional_action else affordance_state.subtract_one()

                # experiential model
                experiential_state = np.append(environmental_state * attentional_state.bits, affective_states).astype(int) # * attentional_mask
                experiental_action = self.experiential_model.get_action(experiential_state, self.env.action_space.sample()) # dont know how to incorporate subgoal, gonna ignore for now
                # experiental_action *= affordance_mask

                next_environmental_state, environmental_reward, done, _ = self.env.step(experiental_action)
                next_environmental_state = next_environmental_state.flatten().astype(int)

                next_states = np.append(next_environmental_state, affective_states).astype(int)

                # update q values
                self.experiential_model.Q[states, experiental_action] = self.experiential_model.calculate_q(states, experiental_action, environmental_reward, next_states)
                self.goal_model.Q[states, subgoal_action] = self.goal_model.calculate_q(states, subgoal_action, environmental_reward, next_states)
                self.attentional_model.Q[environmental_state, attentional_action] = self.attentional_model.calculate_q(environmental_state, attentional_action, environmental_reward, next_environmental_state)
                self.affordance_model.Q[states, affordance_action] = self.affordance_model.calculate_q(states, affordance_action, environmental_reward, next_states)
                self.past_qs.appendleft(self.experiential_model.Q[states, experiental_action])

                environmental_state = next_environmental_state
                states = next_states

                iterations += 1
                total_reward += environmental_reward
                print(f'finished iteration {iterations}, reward={total_reward}')

            rewards.append(total_reward)

if __name__ == '__main__':
    env = gym.make("CartPole-v1").env
    agent = Agent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    agent.run()