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

def test_model():
    #alternately, test model on one word
    rng = np.random.default_rng()
    rollouts = np.load('data/trajectories_all.npy', allow_pickle=True)
    transitions = rollout.flatten_trajectories_with_rew(rollouts)

    env = monitor.Monitor(WordleEnv())
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
    while action == -1:
            answer = input('Answer: ').upper()
            action = env.get_word_index(answer)
    
    env = WordleEnv()

    # Wrap the existing environment
    env = monitor.Monitor(WordleEnv(answer=action))
    venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])  # Wrap a single environment -- only useful for simple testing like this
    venv.render_mode = "human"
    # Run policy evaluation without resetting the environment
    rew, episode_lengths = evaluate_policy(bc_trainer.policy, venv, n_eval_episodes=1, render=True, return_episode_rewards=True)
    print(f"Number of guesses via machine: {episode_lengths[0]}")
    return episode_lengths[0]

def play():
    env = WordleEnv(player_type=1)
    env.render_mode = "human"

    env.reset()
    done = False
    tries = 0
    while not done:
        action = -1
        tries = tries+1 

        while action == -1:
            guess = input('Guess: ').upper()
            if(env.guessable_words.count(guess) > 0):
                action =  env.guessable_words.index(guess)
            else:
                 print("Invalid guess")
            
        obs, r, done, done, _ = env.step(action)
        if not done:
            env.render()
    return tries