import gym
import numpy as np


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Tuple(spaces=(
            gym.spaces.Discrete(3), gym.spaces.Discrete(3), gym.spaces.Discrete(3),
            gym.spaces.Discrete(3), gym.spaces.Discrete(3), gym.spaces.Discrete(3),
            gym.spaces.Discrete(3), gym.spaces.Discrete(3), gym.spaces.Discrete(3),
            gym.spaces.Discrete(3), gym.spaces.Discrete(3), gym.spaces.Discrete(3),
            gym.spaces.Discrete(3), gym.spaces.Discrete(3), gym.spaces.Discrete(3),
            gym.spaces.Discrete(3), gym.spaces.Discrete(3), gym.spaces.Discrete(3),
            gym.spaces.Discrete(3), gym.spaces.Discrete(3), gym.spaces.Discrete(3),
            gym.spaces.Discrete(3), gym.spaces.Discrete(3), gym.spaces.Discrete(3),
            gym.spaces.Discrete(3)))
        

    def observation(self, obs):
        new_obs = np.zeros(25)
        # print(obs.flatten())
        for ind, ob in enumerate(obs.flatten()[:25]):
            if new_obs[ind] == 0 and ob == 1:
                new_obs[ind] = 1
        
        for ind, ob in enumerate(obs.flatten()[25:50]):
            if new_obs[ind] == 0 and ob == 1:
                new_obs[ind] = 2

        return new_obs

go_env = ObservationWrapper(gym.make('gym_go:go-v0', size=5,
                   komi=0, reward_method='real'))
    

action = go_env.uniform_random_action()
print(action)
state, reward, done, info = go_env.step(action)
print(state)
go_env.render('terminal')

action = go_env.uniform_random_action()
print(action)
state, reward, done, info = go_env.step(action)
print(state)
go_env.render('terminal')
