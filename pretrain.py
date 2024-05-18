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
# https://imitation.readthedocs.io/en/latest/_api/imitation.data.huggingface_utils.html#imitation.data.huggingface_utils.TrajectoryDatasetSequence look at this for link
gym.register(
    id='WordleGame-v0',
    entry_point='Wordle:WordleEnv',
    max_episode_steps=6
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
    env=venv,
    batch_size=6,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=6,
)

#have a look at the section about transitions
"""
https://imitation.readthedocs.io/en/latest/tutorials/1_train_bc.html
Note that the rollout function requires a vectorized environment and needs the RolloutInfoWrapper around each of the environments. This is why we passed the post_wrappers argument to make_vec_env above.
"""

reward, _ = evaluate_policy(expert, venv, 10)
print(f"Reward before training: {reward}")
    
expert.learn(100000, progress_bar=True)  # Note: set to 300000 to train a proficient expert

expert.save('ppo_expert_300k.zip')
reward, _ = evaluate_policy(expert, venv, 10)
print(f"Expert reward: {reward}")
