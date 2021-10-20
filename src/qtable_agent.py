import gym
import numpy as np
from models.q_table import Q_Table
import matplotlib.pyplot as plt

# todo convert all of this to pytorch so we can use autograd :)
class Agent():
    def __init__(self, env: gym.Env, alpha: float, gamma: float, epsilon: float, window_size: int=20, num_episodes: int=3):
        self.env = env
        self.num_episodes = num_episodes
        
        # self.obs_size = env.observation_space.n
        self.action_size = env.action_space.n
        self.model = Q_Table(self.action_size, alpha, gamma, epsilon)


    # function for running through the episodes
    def run(self, render=False):
        rewards = []
        for episode in range(self.num_episodes):
            state = self.env.reset()
            if render:
                self.env.render()

            done = False
            iterations = 0
            total_reward = 0
            while not done:
                action = self.model.get_action(state, self.env.action_space.sample()) # dont know how to incorporate subgoal, gonna ignore for now
                next_state, reward, done, _ = self.env.step(action)

                # update q values
                self.model.Q[state][action] = self.model.calculate_q(state, action, reward, next_state)
                state = next_state

                iterations += 1
                total_reward += reward
                # print(f'finished iteration {iterations}, reward={reward}')

            rewards.append(total_reward)
            print(f'finished episode {episode}, total reward={total_reward}')

        if render:
            self.env.render()

        win = [x for x in rewards if x == 1]
        tie = [x for x in rewards if x == 0]
        loss = [x for x in rewards if x == -1]
        print(f'WIN%: {len(win)/len(rewards)}')
        print(f'TIE%: {len(tie)/len(rewards)}')
        print(f'LOSS%: {len(loss)/len(rewards)}')
        print(f'AVERAGE REWARD: {np.mean(rewards)}')
        # plt.plot(rewards)
        # plt.show()

if __name__ == '__main__':
    np.random.seed(42069)
    env = gym.make("Blackjack-v1")
    agent = Agent(env, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=10000)
    agent.run()