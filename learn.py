"""Script demonstrating the use of `gym_pybullet_drones`' Gym interface.

Class TakeoffAviary is used as a learning env for the A2C and PPO algorithms.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
The boolean argument --rllib switches between `stable-baselines3` and `ray[rllib]`.
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning libraries `stable-baselines3` and `ray[rllib]`.
It is not meant as a good/effective learning example.

"""
import time
import argparse
import numpy as np
import gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from drone_landing.env.LandingAviary import LandingAviary
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_GUI = True
TRAIN = True

# class HParamCallback(BaseCallback):
#     def __init__(self):
#         """
#         Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
#         """
#         super().__init__()

#     def _on_training_start(self) -> None:
#         hparam_dict = {
#             "algorithm": self.model.__class__.__name__,
#             "batch_size": self.model.batch_size,
#         }
#         metric_dict = {
#             "rollout/ep_len_mean": 0,
#         }
#         self.logger.record(
#             "hparams",
#             HParam(hparam_dict, metric_dict),
#             exclude=("stdout", "log", "json", "csv"),
#         )

#     def _on_step(self) -> bool:
#         return True

def run(gui=DEFAULT_GUI):

    # env = make_vec_env("landing-aviary-v0", n_envs=5) #env_kwargs={})
    env = gym.make("landing-aviary-v0")
    # try:
        # model = SAC.load("landing-aviary-v0", env=env)
    # except:
    model = SAC("MlpPolicy",
            env,
            buffer_size = 1000,
            # batch_size=64,
            # learning_rate=0.001,
            # tensorboard_log="./tensorboard/",
            seed=0,
            verbose=1
            )
    # model = PPO(
    #     policy="MlpPolicy",
    #     env=env,
    #     # learning_rate=0.001,
    #     n_steps=64,
    #     batch_size=64,
    #     # n_epochs=15,
    #     # gamma=0.995,
    #     # gae_lambda=0.97,
    #     # ent_coef=0.01,
    #     policy_kwargs={'net_arch': [256, dict(pi=[256, 128], vf=[256, 128])]},
    #     # tensorboard_log="./ppo_tensorboard/",
    #     seed=0,
    #     verbose=1)
    if TRAIN:
        model.learn(total_timesteps=10_000) #callback=HParamCallback())
        model.save("landing-aviary-v0")
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=False)
        print("------------------------------------------------")
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        print("------------------------------------------------")

    env = LandingAviary(gui=gui, record=True)
    obs = env.reset()
    start = time.time()


    total_reward = 0
    for i in range((env.EPISODE_LEN_SEC + 10) * int(env.SIM_FREQ/env.AGGR_PHY_STEPS)):
        # action, _states = model.predict(obs)
        # Action example
        action = np.array([[-0.3]*4])

        obs, reward, done, info = env.step(action)
        print(obs)
        total_reward += reward
        if i % env.SIM_FREQ == 0:
            env.render()
        sync(i, start, env.AGGR_PHY_STEPS * env.TIMESTEP)
        if done:
            print("Episode reward", total_reward)
            total_reward = 0
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description='Single agent reinforcement learning example script using TakeoffAviary')
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,
                        help='Whether to use PyBullet GUI (default: True)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
