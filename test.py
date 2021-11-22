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
        return new_obs

class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env=None):
        super(ActionWrapper, self).__init__(env)
    
    def action(self, action):
        return action

go_env = ObservationWrapper(ActionWrapper(gym.make('gym_go:go-v0', size=5,
                   komi=0, reward_method='real')))
    

action = go_env.uniform_random_action()
print(action)
state, reward, done, info = go_env.step(action)
print(state)
go_env.render('terminal')
