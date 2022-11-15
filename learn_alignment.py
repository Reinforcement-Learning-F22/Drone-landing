import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

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
        return True

TOTAL_TIMESTAMPS = 200_000
OBS = ObservationType.KIN

env = make_vec_env("alignment-aviary-v0", n_envs=7, env_kwargs={'obs': OBS})
# env = gym.make("landing-aviary-v0", obs = obs, act=act)
try:
    raise
    model_name = "alignment-SAC_kin_tt200000"
    model = PPO.load(model_name, env=env)
except:
    model = PPO(
            "MlpPolicy",
            # "CnnPolicy",
            env,
            # learning_rate = 0.001,
            batch_size=64,
            n_epochs = 5, 
            n_steps = 64,
            ent_coef = 0.01,
            tensorboard_log="./tensorboard/",
            seed=0,
            verbose=1
            )

model.learn(total_timesteps=TOTAL_TIMESTAMPS, callback=HParamCallback())
model_name = "alignment-" + model.__class__.__name__ + "_" + OBS._value_ + "_tt" + str(TOTAL_TIMESTAMPS)
model.save(model_name)
mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=10, deterministic=False)
print("------------------------------------------------")
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
print("------------------------------------------------")

