import gymnasium as gym
from imitation.algorithms import bc
from Wordle import WordleEnv
from itertools import count
import pandas as pd
#make sure to run file in dedicated terminal ffs SDFJSDJFSLKDJF

from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.a2c import MlpPolicy
from imitation.util.util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import TimeLimit
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

gym.register(
    id='WordleGame-v0',
    entry_point='Wordle:WordleEnv',
    max_episode_steps=7
)

env = gym.make('WordleGame-v0')
#check_env(env)

venv = make_vec_env(
    "WordleGame-v0",
    rng=np.random.default_rng(),
    n_envs=4,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
)

from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import TimeLimit

expert = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=64,
)

reward, _ = evaluate_policy(expert, env, 10)
print(f"Reward before training: {reward}")
    
expert.learn(100)  # Note: set to 100000 to train a proficient expert
reward, _ = evaluate_policy(expert, expert.get_env(), 10)
print(f"Expert reward: {reward}")

"""
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

"""