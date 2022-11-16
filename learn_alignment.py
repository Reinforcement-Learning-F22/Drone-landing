from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
import json

from drone_landing.env.BaseSingleAgentAviary import ObservationType


class HParamCallback(BaseCallback):
    def __init__(self):
        """
        Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
        """
        super().__init__()

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": self.model.learning_rate,
            "target_update_interval": self.model.target_update_interval,
            "batch_size": self.model.batch_size,
            "train_freq": self.model.train_freq[0],
        }
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean": 0
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )
        self.logger.record("hyper_params", json.dumps(hparam_dict))

    def _on_step(self) -> bool:
        return True


TOTAL_TIMESTAMPS = 60_000
OBS = ObservationType.RGB

env = make_vec_env("alignment-aviary-v0", n_envs=7, env_kwargs={'obs': OBS})
try:
    raise
    model_name = "alignment-DQN_rgb_tt100000"
    model = DQN.load(model_name, env=env, leaarning_rate=0.00005)
except:
    # raise
    policy = "MlpPolicy" if OBS == ObservationType.KIN else "CnnPolicy"
    # model = PPO(
    #         policy,
    #         env,
    #         learning_rate = 0.005,
    #         batch_size=32,
    #         # n_epochs = 5,
    #         n_steps = 32,
    #         # ent_coef = 0.001,
    #         tensorboard_log="./tensorboard/",
    #         seed=0,
    #         verbose=1
    #         )

    model = DQN(policy,
                env,
                learning_rate=0.00002,
                # buffer_size=300_000,
                learning_starts=300,
                batch_size=32,
                # train_freq=(8, "step"),
                target_update_interval=500,
                tensorboard_log="./tensorboard/",
                seed=0,
                verbose=1)

    # model = A2C(policy,
    #             env,
    #             learning_rate=0.01,
    #             tensorboard_log="./tensorboard/",
    #             seed = 0,
    #             verbose=1)

model.learn(total_timesteps=TOTAL_TIMESTAMPS, callback=HParamCallback())
model_name = "alignment-" + model.__class__.__name__ + \
    "_" + OBS._value_ + "_tt" + str(TOTAL_TIMESTAMPS)
model.save(model_name)
mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=10, deterministic=False)
print("------------------------------------------------")
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
print("------------------------------------------------")
