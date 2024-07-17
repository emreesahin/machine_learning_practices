import gym
import numpy as np
from tqdm import tqdm

# Ortamı oluşturma
environment = gym.make('FrozenLake-v1', is_slippery=False, render_mode='ansi')
environment.reset()
nb_states = environment.observation_space.n
nb_actions = environment.action_space.n 
qtable = np.zeros((nb_states, nb_actions))

print('Q-table: ')  # ajanin beyni
print(qtable)

episodes = 1000
alpha = 0.5
gamma = 0.9

outcomes = []

# Eğitim
for _ in tqdm(range(episodes)):
    state = environment.reset()[0]
    done = False  # basari durumu

    outcomes.append('Failure')

    while not done:  # basarili olana kadar state icerisinde kal 
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else: 
            action = environment.action_space.sample()
        
        new_state, reward, done, _, info = environment.step(action)

        # Q-table'ı güncelleme
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        state = new_state

        if reward: 
            outcomes[-1] = 'Success'

print(f"Final State: {state}")
print("Q-table:")
print(qtable)

# Başarı oranını hesaplama
success_rate = outcomes.count('Success') / episodes
print(f"Success Rate: {success_rate:.2f}")

# Örnek bir oyun oynama
state = environment.reset()[0]
environment.render()
done = False

while not done:
    action = np.argmax(qtable[state])
    state, reward, done, _, info = environment.step(action)
    environment.render()
