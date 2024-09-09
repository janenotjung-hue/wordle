#loading specific checkpoint 
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from Wordle import WordleEnv
from stable_baselines3.common import monitor
from Wordle import WordleEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.bc import BC
import gymnasium as gym
from gymnasium.utils.play import play

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

#test model on 10 random games
reward_after_training, episode_lengths = evaluate_policy(bc_trainer.policy, venv, n_eval_episodes=10, render=True, return_episode_rewards=True)
print(f"Reward after training: {(reward_after_training)}")
print(f"Average reward after training: {np.mean(reward_after_training)}")

print(f"Episode lengths (number of steps): {episode_lengths}")
print(f"Average number of steps per episode: {np.mean(episode_lengths)}")


