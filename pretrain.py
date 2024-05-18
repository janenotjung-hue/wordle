import gymnasium as gym
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3 import PPO
from stable_baselines3.a2c import MlpPolicy
from imitation.data import rollout
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
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
    n_envs=1,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
)
#rollouts = trajectories
rollouts = np.load('data/trajectories_all.npy', allow_pickle=True)

transitions = rollout.flatten_trajectories_with_rew(rollouts)
learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
    seed=None,
)

reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=transitions,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

# evaluate the learner before training
learner_rewards_before_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

# train the learner and evaluate again
gail_trainer.train(4000)  # Train for 800_000 steps to match expert.
learner_rewards_after_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))