import gym
from imitation.algorithms import bc
from Wordle import WordleEnv
from itertools import count
import pandas as pd
import stable_baselines3
#make sure to run file in dedicated terminal ffs SDFJSDJFSLKDJF

gym.register(
    id='WordleGame-v0',
    entry_point='Wordle:WordleEnv',
)

size = None
env = gym.make("WordleGame-v0", subset_size=size) 
obs = env.reset()
env.render()

num_episodes = 10
average_reward = 0

with open('data/wordle_actual.txt', 'r') as f:
    words = [word.strip().upper() for word in f.readlines() if len(word.strip()) == 5]

df = pd.read_csv('dataset/normal.csv')
guesses = df.drop(columns=["hits_0","hits_1","hits_2","hits_3","hits_4"], axis=1)
guesses = guesses.replace("     ", None)
values = []
for index, row in df.iterrows():
    for i in range(0, len(row)):
        if i < len(row) and row.iloc[i] == 'GGGGG':
            values.append(row.iloc[i-1])

for episode in range(num_episodes):
    state = env.reset()
    row = guesses.iloc[episode]
    env.target_word = values[episode]
    for index, value in row.items():
        if value is not None:
            action = words.index(value.upper())
            observation, reward, done, _ = env.step(action)
            env.render()
            if done:
                next_state = None
            else:
                next_state = observation
            if done:
                break
            action = value
            state = next_state
        
    
print('Complete')

