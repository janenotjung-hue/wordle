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
from imitation.data.types import TrajectoryWithRew

num_episodes = 3
with open('data/valid_guesses.csv', 'r') as f:
    words = [word.strip().upper() for word in f.readlines() if len(word.strip()) == 5]

df = pd.read_csv('dataset/normal.csv')
guesses = df.drop(columns=["hits_0","hits_1","hits_2","hits_3","hits_4"], axis=1)
guesses = guesses.replace("     ", None)
answers = []
for index, row in df.iterrows():
    for i in range(0, len(row)):
        if i < len(row) and row.iloc[i] == 'GGGGG':
            answers.append(row.iloc[i-1])

states = []
actions = []
rewards = []
trajs = []

gym.register(
    id='WordleGame-v0',
    entry_point='Wordle:WordleEnv',
    max_episode_steps=6
)

env = WordleEnv()
#check_env(env)

for episode in range(num_episodes):
    print(episode)
    state = env.reset()
    print(state[0])
    states.append(np.array(state[0]))
    env.target_word = answers[episode].upper()
    row = guesses.iloc[episode]
    env.render()
    for index, value in row.items():
        if value is not None:
            action = words.index(value.upper())
            observation, reward, done, done, _ = env.step(action)
            states.append(np.array(observation))
            actions.append(action)
            rewards.append(reward)
            env.render()
            if done:
                next_state = None
            else:
                next_state = observation
            if done:
                break
            
            state = next_state
    print(states)
    print(np.asarray(actions, dtype=np.int64))
    print(rewards)
    trajs.append(TrajectoryWithRew(obs=states, acts=np.array(actions), infos=None, terminal=True, rews=np.array(rewards)))
    """
        TrajectoryWithRew(obs=array([[1],[2],[3],[4],[5],[6],[7]]), acts=array([10087,  5193,  8488,  4657,  9781,  5790], dtype=int64), infos=None, terminal=True, rews=array([  0.,   0.,   1.,   0.,   0., -10.]))]
    """
np.save('data/trajectories', trajs, allow_pickle=True)
print('Complete')



