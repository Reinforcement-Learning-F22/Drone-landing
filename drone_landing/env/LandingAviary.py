import os
import gc
from enum import Enum
import numpy as np
from gym import spaces
import pybullet as p

# from drone_landing.env.BaseAviary import BaseAviary
from drone_landing.env.BaseSingleAgentAviary import BaseSingleAgentAviary
from drone_landing.env.BaseSingleAgentAviary import ObservationType, ActionType
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType

################################################################################

class LandingAviary(BaseSingleAgentAviary):
    """Multi-drone environment class for control applications using vision."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=np.array([[0, 0, 1]]),
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=20,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 output_folder='results',
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of an aviary environment for control applications using vision.

        Attribute `vision_attributes` is automatically set to True when calling
        the superclass `__init__()` method.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        self.EPISODE_LEN_SEC = 10
        self.prev_shaping = None
        initial_xyzs = np.array([[np.random.uniform(-0.15, 0.15),
                                  np.random.uniform(-0.15, 0.15), 
                                  np.random.uniform(0.5, 5)]])

        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         output_folder=output_folder,
                         obs=obs,
                         act=act
                         )

    ################################################################################

    def reset(self):
        """Resets the environment.

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.

        """
        self.prev_shaping = None
        self.INIT_XYZS = np.array([[np.random.uniform(-0.15, 0.15),
                                    np.random.uniform(-0.15, 0.15),
                                    np.random.uniform(0.5, 5)]])
        gc.collect()
        return super().reset()

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        p.setAdditionalSearchPath(os.path.dirname(os.path.abspath(__file__)), physicsClientId=self.CLIENT)

        p.loadURDF("data/aruco.urdf",
                [0, 0, 0.005],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT
                )
        p.loadURDF("data/platform.urdf",
                [0, 0, 0.002],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT
                )
        p.loadURDF("data/wall.urdf",
                [5, 0, 5],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT
                )
        p.loadURDF("data/wall.urdf",
                [-5, 0, 5],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT
                )
        p.loadURDF("data/wall.urdf",
                [0, 5, 5],
                p.getQuaternionFromEuler([0, 0, np.pi/2]),
                physicsClientId=self.CLIENT
                )
        self.k = p.loadURDF("data/wall.urdf",
                [0, -5, 5],
                p.getQuaternionFromEuler([0, 0, np.pi/2]),
                physicsClientId=self.CLIENT
                )
        p.loadURDF("data/wall.urdf",
                [0, 0, 10],
                p.getQuaternionFromEuler([0, np.pi/2, 0]),
                physicsClientId=self.CLIENT
                ) 
    
    ################################################################################

    TARGET_RADIUS = 0.1
    XYZ_PENALTY_FACTOR = 10
    VEL_PENALTY_FACTOR = 20
    INSIDE_RADIUS_BONUS = 60
    VEL = 0

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(state[:3])
        vel = np.linalg.norm(state[10:13])
        ang_vel = np.linalg.norm(state[13:16])
        shaping = -(self.XYZ_PENALTY_FACTOR * (dist + dist**2) +\
                    self.ANG_VEL_PENALTY_FACTOR * ang_vel)

        reward = ((shaping - self.prev_shaping)
                  if self.prev_shaping is not None
                  else 0)

        if state[2] < 1 and (self.prev_shaping is not None):
            reward += self.VEL_PENALTY_FACTOR * (self.prev_vel - vel)
        
        self.prev_shaping = shaping
        self.prev_vel = vel

        if state[2] <= 0.05:
            if np.linalg.norm(state[:3]) < self.TARGET_RADIUS:
                reward += self.INSIDE_RADIUS_BONUS
        return reward

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        if self.step_counter/self.SIM_FREQ >= self.EPISODE_LEN_SEC:
            self.done = True
            return True

        state = self._getDroneStateVector(0)
        self.done = state[2] <= 0.05
        return self.done

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return None


    ################################################################################
    
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 8 
        MAX_LIN_VEL_Z = 8

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
