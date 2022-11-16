import time
from stable_baselines3 import PPO, DQN, SAC
import numpy as np
import pybullet as p

from drone_landing.env.Aviary import Aviary
from drone_landing.env.LandingAviary import LandingAviary
from drone_landing.env.BaseSingleAgentAviary import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync


def alignment_step(env, action, start_time):
    if action == 0:
        shift = np.array([0, 0, 0])
    elif action == 1:
        shift = np.array([0.05, 0, 0])
    elif action == 2:
        shift = np.array([-0.05, 0, 0])
    elif action == 3:
        shift = np.array([0, 0.05, 0])
    elif action == 4:
        shift = np.array([0, -0.05, 0])

    state = env._getDroneStateVector(0)
    target_pos = state[:3] + shift

    for i in range(env.SIM_FREQ):
        state = env._getDroneStateVector(0)
        action_, error, _ = env.ctrl.computeControlFromState(control_timestep=env.TIMESTEP,
                                                             state=state,
                                                             target_pos=target_pos,
                                                             )
        obs, reward, done, info = env.step(action_)
        if start_time is not None:
            sync(env.step_counter, start_time,
                 env.AGGR_PHY_STEPS * env.TIMESTEP)

    return obs, reward, done, info


def landing_step(env, action):
    if LANDING_ACT == ActionType.RPM:
        action = np.array(env.HOVER_RPM * (1+0.05*action))
    elif LANDING_ACT == ActionType.ONE_D_RPM:
        action = np.repeat(env.HOVER_RPM * (1+0.05*action), 4)
    return env.step(action)


GUI = True
OBS = ObservationType.RGB
LANDING_ACT = ActionType.ONE_D_RPM

env = Aviary(gui=GUI, obs=OBS, record=True)

model_name = "trained_models/alignment-DQN_rgb_tt60000"
alignment_model = DQN.load(model_name, buffer_size=10000)

model_name = "landing2-SAC_rgb_tt200000"
landing_model = SAC.load(model_name)

obs = env.reset()
start = time.time()

for i in range(20):
    action, states_ = alignment_model.predict(obs)
    alignment_step(env, action, start)

# Stabilize drone position
print("Stabilization")
target_pos = env._getDroneStateVector(0)[:3]
for i in range(int(2 * env.SIM_FREQ / env.AGGR_PHY_STEPS)):
    state = env._getDroneStateVector(0)
    action_, error, _ = env.ctrl.computeControlFromState(control_timestep=env.TIMESTEP,
                                                         state=state,
                                                         target_pos=target_pos,
                                                         )
    obs, reward, done, info = env.step(action_)
    sync(env.step_counter, start, env.AGGR_PHY_STEPS * env.TIMESTEP)
# Sorry for that
p.resetBaseVelocity(env.DRONE_IDS[0], np.zeros(
    3), np.zeros(3), physicsClientId=env.CLIENT)

while True:
    action, states_ = landing_model.predict(obs)
    obs, reward, done, info = landing_step(env, action)
    sync(env.step_counter, start, env.AGGR_PHY_STEPS * env.TIMESTEP)

    if done:
        print("Time elapsed:", env.step_counter/env.SIM_FREQ)
        break

# Wait some time before closing environment
print("Waiting")
time.sleep(5)

env.close()
