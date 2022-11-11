import os
from enum import Enum
import numpy as np
from gym import spaces
import pybullet as p

from drone_landing.env.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType

################################################################################

class ObservationType(Enum):
    """Observation type enumeration class."""
    KIN = "kin"     # Kinematic information (pose, linear and angular velocities)
    RGB = "rgb"     # RGB camera capture in each drone's POV

################################################################################

class HoverAviary(BaseAviary):
    """Multi-drone environment class for control applications using vision."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=np.array([[0, 0, 1]]),
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=20,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obstacles=True,
                 user_debug_gui=False,
                 obs: ObservationType=ObservationType.KIN,
                 output_folder='results'
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
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.EPISODE_LEN_SEC = 3
        self.prev_shaping = None
        
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         vision_attributes=vision_attributes,
                         output_folder=output_folder
                         )
        if vision_attributes:
            self.rgb = self.rgb[0]

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

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        dict[str, ndarray]
            A Dict of Box(4,) with NUM_DRONES entries,
            indexed by drone Id in string format.

        """
        return spaces.Box(low=-1*np.ones(4)/10,
                          high=np.ones(4)/10,
                          dtype=np.float32
                          )
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.IMG_RES[1], self.IMG_RES[0], 4),
                              dtype=np.uint8
                              )
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
            # obs_lower_bound = np.array([-1,      -1,      0,      -1,  -1,  -1,  -1,  -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1,      -1,           -1,           -1,           -1])
            # obs_upper_bound = np.array([1,       1,       1,      1,   1,   1,   1,   1,      1,      1,      1,       1,       1,       1,       1,       1,       1,            1,            1,            1])          
            # return spaces.Box( low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32 )
            ############################################################
            #### OBS SPACE OF SIZE 12
            return spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1]),
                              high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1]),
                              dtype=np.float32
                              )
            ############################################################
        else:
            print("[ERROR] in LandingAviary._observationSpace()")
    
    ################################################################################
  
    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of key "state", see the implementation of `_getDroneStateVector()`,
        the value of key "neighbors" is the drone's own row of the adjacency matrix,
        "rgb", "dep", and "seg" are matrices containing POV camera captures.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES), Box(H,W,4), Box(H,W), Box(H,W)}.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                self.rgb, _, _ = self._getDroneImages(0)
                #### Printing observation to PNG frames example ############
                if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
                                            img_input=self.rgb,
                                            path=self.ONBOARD_IMG_PATH,
                                            frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                            )
            return self.rgb
        elif self.OBS_TYPE == ObservationType.KIN: 
            obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            # return obs
            ############################################################
            #### OBS SPACE OF SIZE 12
            ret = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            return ret.astype('float32')
            ############################################################
        else:
            print("[ERROR] in LandingAviary._computeObs()")        

    ################################################################################
    
    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : dict[str, ndarray]
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        return np.array(self.HOVER_RPM * (1+0.05*action))

    ################################################################################

    TARGET_RADIUS = 0.5
    ANG_VEL_PENALTY_FACTOR = 2
    XYZ_PENALTY_FACTOR = 100
    VEL_PENALTY_FACTOR = 40
    INSIDE_RADIUS_BONUS = 100
    

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(np.array([0, 0, 1]) - state[:3])

        dist_penalty = self.XYZ_PENALTY_FACTOR * (dist)
        angle_z_pen = (abs(state[9]))  #+ state[9]**2)
        shaping = -(dist_penalty + angle_z_pen)
        reward = ((shaping - self.prev_shaping) 
                   if self.prev_shaping is not None else 0)

        self.prev_shaping = shaping

        if state[2] <= 0.05:
            reward -= 20
        else:
            reward += 10
        # reward = -dist - dist**2
        # angle_z_pen = abs(state[9]) + state[9]**2
        # reward -= 3 * angle_z_pen
        # print("r", reward)
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
        state = self._getDroneStateVector(0)
        if state[2] <= 0.05:
            return True 
        if self.step_counter/self.SIM_FREQ >= self.EPISODE_LEN_SEC:
            return True
        return False
    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"state": self._getDroneStateVector(0)}


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
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

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
