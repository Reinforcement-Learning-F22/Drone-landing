import time
from stable_baselines3 import PPO, SAC

from drone_landing.env.AlignmentAviary import AlignmentAviary
from drone_landing.env.BaseSingleAgentAviary import ObservationType
from gym_pybullet_drones.utils.utils import sync


GUI = True
OBS = ObservationType.KIN

env = AlignmentAviary(gui=GUI, obs = OBS, record=True)

# model_name = "trained_models/landing-SAC_kin_tt200000"
# model = SAC.load(model_name, env=env)

obs = env.reset()
start = time.time()
total_reward = 0
actions = [0, 1, 1, 2, 3, 4]
for i in range((env.EPISODE_LEN_SEC + 10) * int(env.SIM_FREQ/env.AGGR_PHY_STEPS)):
    # Action example
    action = actions[i%len(actions)]

    # action, _states = model.predict(obs)
    obs, reward, done, infos = env.step(action, start)
    total_reward += reward
    if i % env.SIM_FREQ == 0:
        env.render()
    if done:
        print("Episode reward", total_reward)
        total_reward = 0
        obs = env.reset()
        start = time.time()

env.close()