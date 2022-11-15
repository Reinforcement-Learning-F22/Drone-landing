import time
from stable_baselines3 import PPO

from drone_landing.env.AlignmentAviary import AlignmentAviary
from drone_landing.env.BaseSingleAgentAviary import ObservationType
from gym_pybullet_drones.utils.utils import sync


GUI = True
OBS = ObservationType.KIN

env = AlignmentAviary(gui=GUI, obs = OBS, record=True)

model_name = "trained_models/alignment-PPO_kin_tt100000"
model = PPO.load(model_name, env=env)

obs = env.reset()
start = time.time()
total_reward = 0
actions = [1, 3]
for i in range((env.EPISODE_LEN_SEC + 10) * int(env.SIM_FREQ/env.AGGR_PHY_STEPS)):
    # Action example
    action = actions[i%len(actions)]

    action, states_ = model.predict(obs)
    obs, reward, done, info = env.step(action, start)
    total_reward += reward
    if i % env.SIM_FREQ == 0:
        env.render()
    if done:
        print("Time elapsed:", env.step_counter/env.SIM_FREQ)
        print("Episode reward", total_reward)
        total_reward = 0
        obs = env.reset()
        start = time.time()

env.close()