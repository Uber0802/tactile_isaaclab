# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import isaaclab.sim as sim_utils

from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab_tasks.direct.factory.factory_env_cfg import ASSET_DIR
from isaaclab.utils import configclass

from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import ObservationsCfg, RewardsCfg, StackEnvCfg
from isaaclab_tasks.manager_based.manipulation.stack.tactile_stack_env_cfg import TactileFrankaStackEnvCfg



LOCAL_ROBOT_USD_PATH = "./franka_gelsight.usd"
LOCAL_POTTED_MEAT_CAN_USD = "./assets/Props/potted_meat_can_sdf.usd"


@configclass
class GelsightRewardsCfg(RewardsCfg):
    """Reward specifications for the Gelsight environment."""

    rewind_tactile_reward = RewTerm(func=mdp.rewind_tactile_reward, weight=1)

    stack_object_z_reward_exp = RewTerm(
        func=mdp.stack_object_z_reward_exp,
        params={
            "stack_object_cfg": SceneEntityCfg("stack_object"),
            "max_z_distance": 0.0756,
        },
        weight=1.0,
    )

    stack_object_xy_precision = RewTerm(
        func=mdp.stack_object_precision_xy_reward,
        params={
            "stack_object_cfg": SceneEntityCfg("stack_object"),
            "target_cube_cfg": SceneEntityCfg("target_cube"),
            "stack_height_offset": 0.0553,
            "height_tolerance": 0.01,
            "max_xy_distance": 0.16,
            "max_reward": 0.1,
        },
        weight=1.0,
    )

    stack_object_xy_precision_exp = RewTerm(
        func=mdp.stack_object_precision_xy_reward_exp,
        params={
            "stack_object_cfg": SceneEntityCfg("stack_object"),
            "target_cube_cfg": SceneEntityCfg("target_cube"),
            "stack_height_offset": 0.0553,
            "height_tolerance": 0.005,
            "distance_offset": 0.1,
            "decay_rate": 30.0,
        },
        weight=1.0,
    )

    target_cube_home_penalty = RewTerm(
        func=mdp.target_cube_original_xy_penalty,
        params={
            "stack_object_cfg": SceneEntityCfg("stack_object"),
            "target_cube_cfg": SceneEntityCfg("target_cube"),
            "stack_height_offset": 0.0553,
            "height_tolerance": 0.005,
            "penalty_scale": 5.0,
        },
        weight=-1.0,
    )

    stack_success = RewTerm(
        func=mdp.stack_success,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "stack_object_cfg": SceneEntityCfg("stack_object"),
            "target_cube_cfg": SceneEntityCfg("target_cube"),
            "height_diff": 0.0553,
            "xy_threshold": 0.04,
            "height_threshold": 0.005
        },
        weight=10.0,
    )


@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [-0.4536, 0.1362, 0.3922, -2.3182, -0.1029, 2.223, 0.7862, 0.0400, 0.0400],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose_use_rot,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "yaw": (-math.pi / 8, math.pi / 8)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("stack_object"), SceneEntityCfg("target_cube")],
        },
    )

@configclass
class FrankaStackPottedMeatCanEnvCfg(TactileFrankaStackEnvCfg):
    """Configuration for the Franka Gelsight Environment with Potted Meat Can."""

    # Override the observations and rewards
    rewards: GelsightRewardsCfg = GelsightRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set each stacking cube deterministically
        self.scene.stack_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/stack_object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.02, 0.05), rot=(0.7071, 0.7071, 0, 0.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=LOCAL_POTTED_MEAT_CAN_USD,
                rigid_props=self.cube_properties,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                semantic_tags=[("class", "stack_object")],
            ),
        )
