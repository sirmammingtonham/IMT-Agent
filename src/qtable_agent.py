import gym
import numpy as np
from models.q_table import Q_Table
import matplotlib.pyplot as plt
from observation_wrapper import ObservationWrapper

# todo convert all of this to pytorch so we can use autograd :)
class QAgent():
    def __init__(self, env: gym.Env, alpha: float, gamma: float, epsilon: float):
        self.env = env
        
        # self.obs_size = env.observation_space.n
        self.action_size = env.action_space.n
        self.model = Q_Table(self.action_size, alpha, gamma, epsilon)


    # function for running through the episodes
    def run(self, num_episodes=1000, render=False):
        rewards = []
        for episode in range(num_episodes):
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
    # Go test with qAgents running against each other
    env = ObservationWrapper(gym.make('gym_go:go-v0', size=5,
                   komi=0, reward_method='real'))
    agent1 = QAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    agent2 = QAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)

    rewards = []
    for episode in range(1):
        state = env.reset()

        done = False
        iterations = 0
        total_reward = 0
        while not done:
            action = agent1.model.get_action(state.tobytes(), env.uniform_random_action())
            # print(f'agent1: {action}')
            try:
                next_state, reward, done, _ = env.step(action)
            except:
                pass

            # print(state.shape)
            agent1.model.Q[state.tobytes()][action] = agent1.model.calculate_q(state.tobytes(), action, reward, next_state.tobytes())
            state = next_state

            if done:
                state = env.reset()
                break

            action = agent2.model.get_action(state.tobytes(), env.uniform_random_action())
            # print(f'agent2: {action}')
            try:
                next_state, reward, done, _ = env.step(action)
            except:
                pass
            
            agent2.model.Q[state.tobytes()][action] = agent2.model.calculate_q(state.tobytes(), action, reward, next_state.tobytes())
            state = next_state

            iterations += 1
            total_reward += reward

            if done:
                state = env.reset()
                break
            
            # uncomment to see visual steps
            # env.render('terminal')
        rewards.append(total_reward)
        print(f'finished episode {episode}, total reward={total_reward}')
        

    win = [x for x in rewards if x == 1]
    tie = [x for x in rewards if x == 0]
    loss = [x for x in rewards if x == -1]
    print(f'WIN%: {len(win)/len(rewards)}')
    print(f'TIE%: {len(tie)/len(rewards)}')
    print(f'LOSS%: {len(loss)/len(rewards)}')
    print(f'AVERAGE REWARD: {np.mean(rewards)}')

        
