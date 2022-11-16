import time
from stable_baselines3 import PPO, DQN

from drone_landing.env.AlignmentAviary import AlignmentAviary
from drone_landing.env.BaseSingleAgentAviary import ObservationType
from gym_pybullet_drones.utils.utils import sync


GUI = True
OBS = ObservationType.RGB

env = AlignmentAviary(gui=GUI, obs=OBS, record=True)

model_name = "trained_models/alignment-DQN_rgb_tt60000"
model = DQN.load(model_name, env=env, buffer_size=10000)

obs = env.reset()
start = time.time()
total_reward = 0
for i in range((env.EPISODE_LEN_SEC + 10) * int(env.SIM_FREQ/env.AGGR_PHY_STEPS)):
    # Action example
    # action = 1

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
