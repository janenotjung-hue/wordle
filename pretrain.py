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

#have a look at the section about transitions
"""
https://imitation.readthedocs.io/en/latest/tutorials/1_train_bc.html
Note that the rollout function requires a vectorized environment and needs the RolloutInfoWrapper around each of the environments. This is why we passed the post_wrappers argument to make_vec_env above.
"""

reward, _ = evaluate_policy(expert, env, 10)
print(f"Reward before training: {reward}")
    
expert.learn(100)  # Note: set to 100000 to train a proficient expert

expert.save('ppo_expert.zip')
reward, _ = evaluate_policy(expert, expert.get_env(), 10)
print(f"Expert reward: {reward}")
