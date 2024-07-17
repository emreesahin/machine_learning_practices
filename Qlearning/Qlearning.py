import gym 
import random 
import numpy as np 

environment = gym.make('FrozenLake-v1', is_slippery = False, render_mode = 'ansi')
environment.reset()
nb_states = environment.observation_space.n
nb_actions = environment.action_space.n 
qtable = np.zeros((nb_states, nb_actions))

print('Q-table: ')
print(qtable)

action = environment.action_space.sample()

"""
sol: 0 
asagi: 1
sag: 2 
yukari: 3

"""

new_state, reward, done, info, _ = environment.step(action)

print(new_state)
