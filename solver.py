#loading specific checkpoint 
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from Wordle import WordleEnv
from stable_baselines3.common import monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.bc import BC
import gymnasium as gym
from gymnasium.utils.play import play
import os

def sovler_model():
    #alternately, test model on one word
    rng = np.random.default_rng()
    rollouts = np.load('data/trajectories_all.npy', allow_pickle=True)
    transitions = rollout.flatten_trajectories_with_rew(rollouts)

    env = monitor.Monitor(WordleEnv(solver=1))
    venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])  # Wrap a single environment -- only useful for simple testing like this
    venv.render_mode="human"

    checkpoint_path = "checkpoints/bc_model_epoch_200.zip"  # Example checkpoint path
    loaded_policy = ActorCriticPolicy.load(checkpoint_path)

    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        policy=loaded_policy
    )

    action = -1
    #env = WordleEnv()
    

    # Wrap the existing environment
    #env = monitor.Monitor(WordleEnv(answer=action))
    #venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])  # Wrap a single environment -- only useful for simple testing like this
    #venv.render_mode = "human"
    
    # Run policy evaluation without resetting the environment
    rew, episode_lengths = evaluate_policy(bc_trainer.policy, venv, n_eval_episodes=1, render=True, return_episode_rewards=True)
    print(f"Number of guesses via machine: {episode_lengths[0]}")
    return episode_lengths[0]

m = sovler_model()