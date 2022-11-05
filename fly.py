"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import time
import argparse
import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from drone_landing.env.LandingAviary import LandingAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = True
DEFAULT_USER_DEBUG_GUI = True
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 60
DEFAULT_CONTROL_FREQ_HZ = 10
DEFAULT_DURATION_SEC = 60


def run(
        drone=DEFAULT_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        aggregate=DEFAULT_AGGREGATE,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
):
    #### Initialize the simulation #############################
    H = 2
    R = .3
    INIT_XYZS = np.array([0,  0, H])
    INIT_RPYS = np.array([0, 0, 0])
    AGGR_PHY_STEPS = int(simulation_freq_hz /
                         control_freq_hz) if aggregate else 1

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP, 3))
    # TARGET_POS[:,2] = np.linspace(0.5, 25, NUM_WP)
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0], R*np.sin(
            (i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[1], 3
    wp_counter = 0

    env = LandingAviary(drone_model=drone,
                        num_drones=1,
                        initial_xyzs=INIT_XYZS[None, :],
                        initial_rpys=INIT_RPYS[None, :],
                        physics=physics,
                        neighbourhood_radius=10,
                        freq=simulation_freq_hz,
                        aggregate_phy_steps=AGGR_PHY_STEPS,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = DSLPIDControl(drone_model=drone)
    elif drone in [DroneModel.HB]:
        ctrl = SimplePIDControl(drone_model=drone)

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/control_freq_hz))
    action = np.array([0, 0, 0, 0])
    START = time.time()
    for i in range(0, int(duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #### Compute control at the desired frequency ##############
        if i % CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            action, _, _ = ctrl.computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                        state=info["state"],
                                                        target_pos=TARGET_POS[wp_counter],
                                                        target_rpy=INIT_RPYS
                                                        )
            print("hover", env.HOVER_RPM)
            print("action", action)
            print("max", env.MAX_RPM)
            action = (action/env.HOVER_RPM-1)/0.05

            #### Go to the next way point and loop #####################
            wp_counter = wp_counter + 1 if wp_counter < (NUM_WP-1) else 0

        #### Printout ##############################################
        if i % env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,
                        help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,
                        help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,
                        help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,
                        type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,
                        help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=DEFAULT_AGGREGATE,       type=str2bool,
                        help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,
                        help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,
                        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,
                        type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,
                        help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
