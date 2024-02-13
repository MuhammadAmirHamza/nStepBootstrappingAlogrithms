"""
Description:
============

The purpose of this script is to get comfortable with the environment
and also to find the practical limits of the observation, this will help
in Quantization of the states.
"""


import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


env = gym.make('CartPole-v1', max_episode_steps=1000)

print('Action Space                 : ', env.action_space.n)
print('Number of Observations       : ', env.observation_space.shape)
print('Observation Space High       : ', env.observation_space.high)
print('Observation Space Low        : ', env.observation_space.low)
print('Observation Space shape      : ', env.observation_space.shape)
print('Meta data                    : ', env.metadata)
print('Specs                        : ', env.spec)
print('Doc                          : ', env.__doc__)


# finding the min. and max of the observation received practically
env.reset()
temp = np.zeros((5000, 4))
for i in range(5000):
    action = env.action_space.sample()
    nextState, _, terminated, truncated, _ = env.step(action)
    temp[i, :] = nextState
    # print(nextState)
    if terminated or truncated:
        env.reset()
        
print('max position                     : ', np.max(temp[:, 0]))
print('max linear velocity              : ', np.max(temp[:, 1]))
print('max angle                        : ', np.max(temp[:, 2]))
print('max angular velocity             : ', np.max(temp[:, 3]))
print('\n')
print('min  position                     : ', np.min(temp[:, 0]))
print('min  linear velocity              : ', np.min(temp[:, 1]))
print('min  angle                        : ', np.min(temp[:, 2]))
print('min  angular velocity             : ', np.min(temp[:, 3]))

