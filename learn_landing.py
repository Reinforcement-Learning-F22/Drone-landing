import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3 
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from drone_landing.env.BaseSingleAgentAviary import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool


class HParamCallback(BaseCallback):
    def __init__(self):
        """
        Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
        """
        super().__init__()

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "batch_size": self.model.batch_size,
        }
        metric_dict = {
            "rollout/ep_len_mean": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        dones = self.training_env.get_attr("done")
        vels = self.training_env.get_attr("VEL")
        for done, vel in zip(dones, vels):
            if done:
                self.logger.record("end velocity", vel)
        return True


TOTAL_TIMESTAMPS = 400000
OBS = ObservationType.KIN
ACT = ActionType.ONE_D_RPM

env = make_vec_env("landing-aviary-v0", n_envs=7,
                   env_kwargs={'obs': OBS, 'act': ACT})

env_td3 = make_vec_env("landing-aviary-v0", n_envs=1,
                   env_kwargs={'obs': OBS, 'act': ACT})


EXPEREMENT_PARAMS = {
    # "PPO":[ {"name":"PPO_1", "buffer_size":100000, "batch_size":64, "learning_rate":0.001, "gamma":0.99},
    #         {"name":"PPO_2", "buffer_size":100000, "batch_size":64, "learning_rate":0.0001, "gamma":0.9},
    #         {"name":"PPO_3", "buffer_size":100000, "batch_size":64, "learning_rate":0.001, "gamma":0.1},
    #         {"name":"PPO_4", "buffer_size":100000, "batch_size":32, "learning_rate":0.001, "gamma":0.7}], 
    "DDPG":[{"name":"DDPG_1", "buffer_size":100000, "batch_size":64, "learning_rate":0.001, "gamma":0.99},
            {"name":"DDPG_2", "buffer_size":100000, "batch_size":64, "learning_rate":0.0001, "gamma":0.99},
            {"name":"DDPG_3", "buffer_size":100000, "batch_size":64, "learning_rate":0.01, "gamma":0.7},
            {"name":"DDPG_4", "buffer_size":100000, "batch_size":64, "learning_rate":0.001, "gamma":0.7}], 
    # "SAC":[ {"name":"SAC_1", "buffer_size":100000, "batch_size":64, "learning_rate":0.001, "gamma":0.99}, 
    #         {"name":"SAC_2", "buffer_size":100000, "batch_size":64, "learning_rate":0.0001, "gamma":0.9}, 
    #         {"name":"SAC_3", "buffer_size":100000, "batch_size":32, "learning_rate":0.01, "gamma":0.1},
    #         {"name":"SAC_4", "buffer_size":100000, "batch_size":64, "learning_rate":0.001, "gamma":0.7}]    
}

policy = "MlpPolicy" if OBS == ObservationType.KIN else "CnnPolicy"

offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                            net_arch=[512, 512, 256, 128]
                            )

for model_type in EXPEREMENT_PARAMS:
    experements = EXPEREMENT_PARAMS[model_type]
    for exp in experements:
        model_name = "lan_{0}_kin".format(exp["name"])
        
        # # On-policy algorithms
        if model_type == "PPO":
            model = PPO(
                    policy,
                    env,
                    learning_rate=exp["learning_rate"],
                    gamma=exp["gamma"],
                    batch_size=exp["batch_size"],
                    n_epochs = 5,
                    n_steps = 64,
                    ent_coef = 0.01,
                    tensorboard_log="./tensorboard/",
                    seed=0,
                    verbose=1
                )
        # Off-policy algorithms
        elif model_type == "SAC":
            model = SAC(
                policy,
                env,
                buffer_size=exp["buffer_size"],
                batch_size=exp["batch_size"],
                learning_rate=exp["learning_rate"],
                gamma=exp["gamma"],
                tensorboard_log="./tensorboard/",
                seed=0,
                verbose=1
            )
        elif model_type == "DDPG":
            model = TD3(
                policy, 
                env_td3,
                buffer_size=exp["buffer_size"],
                batch_size=exp["batch_size"],
                learning_rate=exp["learning_rate"],
                gamma=exp["gamma"],
                tensorboard_log="./tensorboard/",
                seed=0,
                verbose=1
            )

        model.learn(total_timesteps=TOTAL_TIMESTAMPS, callback=HParamCallback())
        model_name = "landing-" + model.__class__.__name__ + \
            "_" + OBS._value_ + "_tt" + str(TOTAL_TIMESTAMPS) + model_name
        model.save(model_name)
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=False)
        print("------------------------------------------------")
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        print("------------------------------------------------")

