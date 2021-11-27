import gym
import collections
import numpy as np
from models.q_table import Q_Table
from util.bitmask import BitMask
import matplotlib.pyplot as plt

class IMTAgent():
    def __init__(self, obs_size: int, action_size: int, alpha: float, gamma: float, epsilon: float, target_value=0.1, window_size: int = 20):

        self.obs_size = obs_size
        self.action_size = action_size

        self.past_qs = collections.deque(maxlen=window_size)

        self.arousal_model = Q_Table(100, alpha, gamma, epsilon)
        self.emotive_model = Q_Table(100, alpha, gamma, epsilon)

        self.goal_model = Q_Table(3, alpha, gamma, epsilon)
        self.subgoal_idx = 0

        self.attentional_model = Q_Table(3, alpha, gamma, epsilon)
        self.attentional_mask = BitMask(obs_size, random=True)

        self.affordance_model = Q_Table(3, alpha, gamma, epsilon)
        self.affordance_mask = BitMask(action_size, random=True)

        self.experiential_model = Q_Table(
            self.action_size, alpha, gamma, epsilon)

        self.visited_states = []

        self.target_value = target_value
        self.affective_values = np.arange(0, 100, 1)

    @staticmethod
    def create_circular_mask(h, w, center, radius):
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask

    def get_distance(self, first_state, second_state):
        first_state = np.array(first_state)
        second_state = np.array(second_state)
        return np.linalg.norm(first_state - second_state)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)

    def concatenate_state(self, *args):
        return np.append(args[0], args[1:])

    def trim_environmental_state(self, env_state, atten_mask):
        return np.asarray([val for ind, val in enumerate(env_state) if not atten_mask.bits[ind]])

# stochastic switch bits
    def step(self, env: gym.Env, environmental_state: np.array):
        self.visited_states.append(environmental_state)

        # affective system
        if len(self.past_qs) == self.past_qs.maxlen:
            first_derivative = np.gradient(self.past_qs)
            second_derivative = np.gradient(first_derivative)
            first_derivative = np.average(first_derivative)
            second_derivative = np.average(second_derivative)

            arousal_state = self.concatenate_state(environmental_state, second_derivative,)
            arousal_action = self.arousal_model.get_action(arousal_state, np.random.choice(self.affective_values))

            emotive_state = self.concatenate_state(environmental_state, arousal_action, first_derivative,)
            emotive_action = self.emotive_model.get_action(emotive_state, np.random.choice(self.affective_values))
        else:
            arousal_state, emotive_state = None, None
            arousal_action, emotive_action = 0, 0

        # goal model
        subgoal_state = self.concatenate_state(environmental_state, emotive_action, arousal_action,)
        subgoal_action = self.goal_model.get_action(subgoal_state, np.random.choice([-1, 0, 1]))
        self.subgoal_idx += subgoal_action
        self.subgoal_idx = self.clamp(self.subgoal_idx, 0, len(self.visited_states))

        # attentional model
        attentional_action = self.attentional_model.get_action(environmental_state, np.random.choice([-1, 0, 1]))
        if attentional_action == 1:
            self.attentional_mask.add_one()
        elif attentional_action == -1:
            self.attentional_mask.subtract_one()

        # affordance model
        affordance_state = self.concatenate_state(environmental_state, emotive_action, arousal_action, self.subgoal_idx,)
        affordance_action = self.affordance_model.get_action(affordance_state, np.random.choice([-1, 0, 1]))
        if affordance_action == 1:
            self.affordance_mask.add_one()
        elif affordance_action == -1:
            self.affordance_mask.subtract_one()

        # experiential model
        # mask state attentional here
        environmental_state = self.trim_environmental_state(environmental_state, self.attentional_mask)

        experiential_state = self.concatenate_state(environmental_state, emotive_action, arousal_action, self.subgoal_idx,)
        experiental_action = self.experiential_model.get_action(experiential_state, env.action_space.sample())
        # mask action here

        try:
            next_environmental_state, environmental_reward, done, info = env.step(experiental_action)
        except AssertionError: # invalid move
            # do you change experiential action or keep it and have the rewards be from passing?
            next_environmental_state, environmental_reward, done, info = env.step(self.action_size - 1)

        # todo find out how the next state is updated
        next_subgoal_state = self.concatenate_state(next_environmental_state, emotive_action, arousal_action,)
        next_affordance_state = self.concatenate_state(next_environmental_state, emotive_action, arousal_action, self.subgoal_idx,)
        next_experiental_state = self.concatenate_state(next_environmental_state, emotive_action, arousal_action, self.subgoal_idx,)

        if arousal_state is not None:
            next_arousal_state = self.concatenate_state(next_environmental_state, second_derivative,)
            next_emotive_state = self.concatenate_state(next_environmental_state, arousal_action, first_derivative,)
            arousal_reward = (1  - (self.target_value -  second_derivative)) * environmental_reward
            emotive_reward =(1  - (self.target_value -  first_derivative)) * environmental_reward
            #  update q values
            self.arousal_model.update_q(arousal_state, arousal_action, arousal_reward, next_arousal_state)
            self.emotive_model.update_q(emotive_state, emotive_action, emotive_reward, next_emotive_state)

        subgoal_reward = self.get_distance(environmental_state, self.visited_states[self.subgoal_idx]) - environmental_reward
        self.goal_model.update_q(subgoal_state, subgoal_action, subgoal_reward, next_subgoal_state)

        self.attentional_model.update_q(environmental_state, attentional_action, environmental_reward, next_environmental_state)
        self.affordance_model.update_q(affordance_state, affordance_action, environmental_reward, next_affordance_state)

        self.experiential_model.update_q(experiential_state, experiental_action, environmental_reward, next_experiental_state)
        self.past_qs.appendleft(self.experiential_model.Q[experiential_state.tobytes()][experiental_action])


        return next_environmental_state, environmental_reward, done, info


    # function for running through the episodes
#     def run(self, env, num_episodes=1000, render=False):
#         rewards = []
#         for episode in range(num_episodes):
#             # print(f'starting episode {episode}')

#             environmental_state = env.reset()

#             if render:
#                 env.render()

#             subgoal_idx = 0  # np.random.choice(np.arange(self.obs_size))
#             attentional_mask = BitMask(self.obs_size, random=True)
#             affordance_mask = BitMask(self.action_size, random=True)
#             arousal_state, emotive_state = None, None
#             arousal_action, emotive_action = (0, 0)
#             environmental_reward = None

#             done = False
#             iterations = 0
#             total_reward = 0
#             while not done:
#                 environmental_reward, environmental_state, arousal_state, arousal_action, emotive_state, emotive_action, subgoal_idx = self.agent_step(environmental_state, arousal_state, arousal_action, emotive_state, emotive_action, subgoal_idx, render)
#                 total_reward += environmental_reward
#                 iterations += 1
                

#             rewards.append(total_reward)
#             print(f'finished episode {episode}, total reward={total_reward}')

#         win = [x for x in rewards if x == 1]
#         tie = [x for x in rewards if x == 0]
#         loss = [x for x in rewards if x == -1]
#         print(f'WIN%: {len(win)/len(rewards)}')
#         print(f'TIE%: {len(tie)/len(rewards)}')
#         print(f'LOSS%: {len(loss)/len(rewards)}')
#         print(f'AVERAGE REWARD: {np.mean(rewards)}')
#         # plt.plot(rewards)
#         # plt.show()
#         return rewards


# if __name__ == '__main__':
#     np.random.seed(42069)
#     # env = gym.make("Blackjack-v1")
#     env = gym.make('gym_go:go-v0', size=7, komi=0)
#     agent = IMTAgent(25,26, alpha=0.1, gamma=0.9, epsilon=0.1)
#     agent.run(env, num_episodes=100000)


# # bucket space in 1000s
# # influence map
# # distance
# # internal r (new state close to the subgoal)