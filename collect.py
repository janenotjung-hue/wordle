import numpy as np
import gymnasium as gym
import pandas as pd
from Wordle import WordleEnv
from stable_baselines3.common.env_checker import check_env
from collections import namedtuple
from imitation.data.rollout import flatten_trajectories_with_rew
import imitation.data.types as dt
from imitation.util import util
from typing import Mapping, Sequence, cast
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


env = WordleEnv()
#check_env(env)
num_episodes = 1
with open('data/wordle_actual.txt', 'r') as f:
    words = [word.strip().upper() for word in f.readlines() if len(word.strip()) == 5]

df = pd.read_csv('dataset/normal.csv')
guesses = df.drop(columns=["hits_0","hits_1","hits_2","hits_3","hits_4"], axis=1)
guesses = guesses.replace("     ", None)
answers = []
for index, row in df.iterrows():
    for i in range(0, len(row)):
        if i < len(row) and row.iloc[i] == 'GGGGG':
            answers.append(row.iloc[i-1])

import torch
from ReplayMemory import ReplayMemory
memory = []

for episode in range(num_episodes):
    print(episode)
    state = env.reset()
    row = guesses.iloc[episode]
    env.target_word = answers[episode].upper()
    for index, value in row.items():
        if value is not None:
            action = words.index(value.upper())
            observation, reward, done, done, _ = env.step(action)
            reward = torch.tensor([reward])
            env.render()
            if done:
                next_state = None
            else:
                next_state = observation
                next_state = torch.tensor(observation, dtype=torch.float32)
                #TODO: look at how to add trajectory? transition?
            memory.append(state, action, next_state, reward)
            if done:
                break
            state = next_state
        
rollout = flatten_trajectories_with_rew(memory)
print('Complete')



