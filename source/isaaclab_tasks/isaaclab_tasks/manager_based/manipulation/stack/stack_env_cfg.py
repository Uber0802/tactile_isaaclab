# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        stack_1 = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()

@configclass
class RewardsCfg:
    """Reward terms for stacking."""

    ee_to_blue_cube = RewTerm(
        func=mdp.ee_to_cube_distance_reward,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "cube_cfg": SceneEntityCfg("cube_1"),
            "max_distance": 0.25,
            "max_reward": 1.0,
        },
        weight=0.5,
    )

    blue_cube_z_reward_exp = RewTerm(
        func=mdp.cube_z_reward_exp,
        params={
            "cube_1_cfg": SceneEntityCfg("cube_1"),
        },
        weight=1.0,
    )

    blue_cube_xy_precision = RewTerm(
        func=mdp.cube_precision_xy_reward,
        params={
            "cube_1_cfg": SceneEntityCfg("cube_1"),
            "cube_2_cfg": SceneEntityCfg("cube_2"),
            "stack_height_offset": 0.0468,
            "height_tolerance": 0.005,
            "max_xy_distance": 0.16,
            "max_reward": 0.1,
        },
        weight=1.0,
    )

    blue_cube_xy_precision_exp = RewTerm(
        func=mdp.cube_precision_xy_reward_exp,
        params={
            "cube_1_cfg": SceneEntityCfg("cube_1"),
            "cube_2_cfg": SceneEntityCfg("cube_2"),
            "stack_height_offset": 0.0468,
            "height_tolerance": 0.005,
            "distance_offset": 0.1,
            "decay_rate": 30.0,
        },
        weight=1.0,
    )

    red_cube_home_penalty = RewTerm(
        func=mdp.cube_original_xy_penalty,
        params={
            "cube_cfg": SceneEntityCfg("cube_2"),
            "cube_1_cfg": SceneEntityCfg("cube_1"),
            "cube_2_cfg": SceneEntityCfg("cube_2"),
            "stack_height_offset": 0.0468,
            "height_tolerance": 0.005,
            "penalty_scale": 5.0,
        },
        weight=-1.0,
    )

    wrist_posture_penalty = RewTerm(
        func=mdp.joint_deviation_l1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint5", "panda_joint6", "panda_joint7"]),
        },
        weight=-0.1,
    )

    stack_success = RewTerm(
        func=mdp.cubes_stacked,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "cube_1_cfg": SceneEntityCfg("cube_1"),
            "cube_2_cfg": SceneEntityCfg("cube_2"),
        },
        weight=10.0,
    )

    # # Small per-step penalties to discourage waiting or issuing large actions.
    time_penalty = RewTerm(func=mdp.is_alive, weight=-0.002)
    action_penalty = RewTerm(func=mdp.action_l2, weight=-0.001)
    # rewind_tactile_reward = RewTerm(func=mdp.rewind_tactile_reward, weight=0.2)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")}
    )

    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")}
    )

    cube_3_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_3")}
    )

    cube_1_out_of_bounds = DoneTerm(
        func=mdp.root_horizontal_displacement_exceeded,
        params={"max_displacement": 0.4, "asset_cfg": SceneEntityCfg("cube_1")},
    )

    cube_2_out_of_bounds = DoneTerm(
        func=mdp.root_horizontal_displacement_exceeded,
        params={"max_displacement": 0.4, "asset_cfg": SceneEntityCfg("cube_2")},
    )

    cube_3_out_of_bounds = DoneTerm(
        func=mdp.root_horizontal_displacement_exceeded,
        params={"max_displacement": 0.4, "asset_cfg": SceneEntityCfg("cube_3")},
    )

    # success = DoneTerm(func=mdp.cubes_stacked)


@configclass
class StackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    # reward logging
    log_reward_shaping: bool = True
    reward_log_path: str | None = "reward_shaping.txt"
    reward_log_env_idx: int = 135

    # Unused managers
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    events = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 5

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.observations.rgb_camera = None
        self.observations.subtask_terms = None
        self.observations.policy.concatenate_terms = True
