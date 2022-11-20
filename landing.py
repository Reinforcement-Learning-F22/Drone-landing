import time
import numpy as np
from stable_baselines3 import PPO, SAC

from drone_landing.env.LandingAviary import LandingAviary
from drone_landing.env.BaseSingleAgentAviary import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync


GUI = True
OBS = ObservationType.KIN
ACT = ActionType.ONE_D_RPM

env = LandingAviary(gui=GUI, obs=OBS, act=ACT, record=True)

model_name = "landing-SAC_kin_tt400000lan_SAC_2_kin"
model = SAC.load(model_name, env=env)

obs = env.reset()
start = time.time()
total_reward = 0
for i in range((env.EPISODE_LEN_SEC + 10) * int(env.SIM_FREQ/env.AGGR_PHY_STEPS)):
    action, _states = model.predict(obs)
    # Action example
    # action = np.array([[-0.3]])

    obs, reward, done, info = env.step(action)
    total_reward += reward
    if i % env.SIM_FREQ == 0:
        env.render()
    sync(i, start, env.AGGR_PHY_STEPS * env.TIMESTEP)
    if done:
        print("Episode reward", total_reward)
        total_reward = 0
        obs = env.reset()
env.close()
