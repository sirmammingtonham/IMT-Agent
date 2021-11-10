import gym
import collections
import numpy as np
from .models.q_table import Q_Table
from .util.bitmask import BitMask
import matplotlib.pyplot as plt


class IMTAgent():
    def __init__(self, env: gym.Env, alpha: float, gamma: float, epsilon: float, target_value=0.1, window_size: int = 20):
        self.env = env

        self.obs_size = env.observation_space
        self.action_size = env.action_space.n
        # self.max_attentional_state = int('1' * self.obs_size, base=2)
        # self.max_affordance_state = int('1' * self.action_size, base=2)

        self.past_qs = collections.deque(maxlen=window_size)

        self.arousal_model = Q_Table(100, alpha, gamma, epsilon)
        self.emotive_model = Q_Table(100, alpha, gamma, epsilon)
        self.goal_model = Q_Table(2, alpha, gamma, epsilon)
        self.attentional_model = Q_Table(2, alpha, gamma, epsilon)
        self.affordance_model = Q_Table(2, alpha, gamma, epsilon)
        self.experiential_model = Q_Table(
            self.action_size, alpha, gamma, epsilon)

        self.visited_states = {}

        self.target_value = target_value
        self.affective_values = np.arange(0, 1, 0.1)

    @staticmethod
    def create_circular_mask(h, w, center, radius):
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask

    @staticmethod
    def get_distance(first_state, second_state):
        first_state = np.array(first_state)
        second_state = np.array(second_state)
        return np.linalg.norm(first_state - second_state)
# stochastic switch bits

    # function for running through the episodes
    def run(self, num_episodes=1000, render=False):
        rewards = []
        for episode in range(num_episodes):
            # print(f'starting episode {episode}')

            environmental_state = self.env.reset()

            if render:
                self.env.render()

            subgoal_idx = 0  # np.random.choice(np.arange(self.obs_size))
            attentional_mask = BitMask(self.obs_size, random=True)
            affordance_mask = BitMask(self.action_size, random=True)
            arousal_state, emotive_state = None, None
            arousal_action, emotive_action = (0, 0)

            done = False
            iterations = 0
            total_reward = 0
            while not done:
                self.visited_states.append(environmental_state)

                # affective system
                if len(self.past_qs) == self.past_qs.maxlen:
                    first_derivative = np.gradient(self.past_qs)
                    second_derivative = np.gradient(first_derivative)
                    first_derivative = np.average(first_derivative)
                    second_derivative = np.average(second_derivative)

                    if arousal_state is not None:
                        next_arousal_state = (environmental_state, second_derivative)
                        next_emotive_state = (environmental_state, arousal_action, first_derivative)
                        self.arousal_model.Q[arousal_state][arousal_action] = self.arousal_model.calculate_q(arousal_state, arousal_action, arousal_reward, next_arousal_state)
                        self.emotive_model.Q[emotive_state][emotive_action] = self.emotive_model.calculate_q(emotive_state, emotive_action, emotive_reward, next_emotive_state)

                    arousal_state = (environmental_state, second_derivative)
                    emotive_state = (environmental_state, arousal_action, first_derivative)

                    arousal_action = self.arousal_model.get_action(arousal_state, np.random.choice(self.affective_values))
                    arousal_reward = 1 - np.abs(arousal_action - self.target_value) # arousal reward is distance between action and target
                    
                    emotive_action = self.arousal_model.get_action(emotive_state, np.random.choice(self.affective_values))
                    emotive_reward = 1 - np.abs(emotive_action - arousal_action)

                    arousal_action = int(arousal_action*100)
                    emotive_action = int(emotive_action*100)

                # goal model
                subgoal_state = (environmental_state, emotive_action, arousal_action)
                subgoal_action = self.goal_model.get_action(subgoal_state, np.random.choice([-1, 0, 1]))
                subgoal_idx += subgoal_action  # (should be -1 or +1)
                if subgoal_idx < 0:
                    subgoal_idx = 0
                elif subgoal_idx == len(self.visited_states):
                    subgoal_idx -= 1
# influence map
# distance
# internal r (new state close to the subgoal)

                # attentional model
                attentional_action = self.attentional_model.get_action(environmental_state, np.random.choice([0, 1]))
                attentional_mask.add_one() if attentional_action else attentional_mask.subtract_one()
# flip bits

#instead of addone, 
                # affordance model
                affordance_state = (environmental_state, emotive_action, arousal_action, subgoal_idx, ) # attentional_mask?)
                affordance_action = self.affordance_model.get_action(affordance_state, np.random.choice([0, 1]))
                affordance_mask.add_one() if attentional_action else affordance_mask.subtract_one()

                # experiential model
                experiential_state = (environmental_state, emotive_action, arousal_action, subgoal_idx)
                experiental_action = self.experiential_model.get_action(experiential_state, self.env.action_space.sample())
                if not affordance_mask.bits[experiental_action]:
                    experiental_action = 0
                    # turn action into pass

                next_environmental_state, environmental_reward, done, _ = self.env.step(experiental_action)
                if render:
                    self.env.render()

                next_experiental_state = (next_environmental_state, emotive_action, arousal_action, subgoal_idx,)
                next_subgoal_state = (next_environmental_state, emotive_action, arousal_action,)
                next_affordance_state = (next_environmental_state, emotive_action, arousal_action, subgoal_idx,)

                # update q values
                self.experiential_model.Q[experiential_state][experiental_action] = self.experiential_model.calculate_q(experiential_state, experiental_action, environmental_reward, next_experiental_state)

                subgoal_reward = IMTAgent.get_distance(environmental_state, self.visited_states[subgoal_idx]) - environmental_reward
                self.goal_model.Q[subgoal_state][subgoal_action] = self.goal_model.calculate_q(subgoal_state, subgoal_action, subgoal_reward, next_subgoal_state)

                self.attentional_model.Q[environmental_state][attentional_action] = self.attentional_model.calculate_q(environmental_state, attentional_action, environmental_reward, next_environmental_state)
                self.affordance_model.Q[affordance_state][affordance_action] = self.affordance_model.calculate_q(affordance_state, affordance_action, environmental_reward, next_affordance_state)

                self.past_qs.appendleft(
                    self.experiential_model.Q[experiential_state][experiental_action])

                environmental_state = next_environmental_state
                # states = next_states

                iterations += 1
                total_reward += environmental_reward
                # print(f'finished iteration {iterations}, reward={environmental_reward}')

            rewards.append(total_reward)
            print(f'finished episode {episode}, total reward={total_reward}')
        # print(self.experiential_model.Q)
        win = [x for x in rewards if x == 1]
        tie = [x for x in rewards if x == 0]
        loss = [x for x in rewards if x == -1]
        print(f'WIN%: {len(win)/len(rewards)}')
        print(f'TIE%: {len(tie)/len(rewards)}')
        print(f'LOSS%: {len(loss)/len(rewards)}')
        print(f'AVERAGE REWARD: {np.mean(rewards)}')
        # plt.plot(rewards)
        # plt.show()
        return rewards


if __name__ == '__main__':
    np.random.seed(42069)
    env = gym.make("Blackjack-v1")
    agent = IMTAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    agent.run(num_episodes=100000)


# bucket space in 1000s
