# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils

from isaaclab.assets import DeformableObjectCfg
from isaaclab.assets import RigidObjectCfg
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

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG # isort: skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

LOCAL_ROBOT_USD_PATH = "./franka_gelsight.usd"
LOCAL_BLUE_BLOCK_USD  = "./assets/Props/blue_block_sdf.usd"
LOCAL_RED_BLOCK_USD   = "./assets/Props/red_block_sdf.usd"
LOCAL_GREEN_BLOCK_USD = "./assets/Props/green_block_sdf.usd"
LOCAL_BOWL_USD = "./assets/Props/bottle_sdf.usd"


@configclass
class GelsightObservationsCfg(ObservationsCfg):
    """Observation specifications for the Gelsight environment."""

    def __post_init__(self):
        super().__post_init__()
        '''
        self.policy.left_tactile_normal_force = ObsTerm(
            func=mdp.tactile_normal_force, params={"sensor_cfg": SceneEntityCfg("left_tactile_sensor")}
        )
        self.policy.left_tactile_shear_force = ObsTerm(
            func=mdp.tactile_shear_force, params={"sensor_cfg": SceneEntityCfg("left_tactile_sensor")}
        )
        self.policy.right_tactile_normal_force = ObsTerm(
            func=mdp.tactile_normal_force, params={"sensor_cfg": SceneEntityCfg("right_tactile_sensor")}
        )
        self.policy.right_tactile_shear_force = ObsTerm(
            func=mdp.tactile_shear_force, params={"sensor_cfg": SceneEntityCfg("right_tactile_sensor")}
        )
        '''
@configclass
class GelsightRewardsCfg(RewardsCfg):
    """Reward specifications for the Gelsight environment."""

    rewind_tactile_reward = RewTerm(func=mdp.rewind_tactile_reward, weight=0.2)

    stack_object_z_reward_exp = RewTerm(
        func=mdp.stack_object_z_reward_exp,
        params={
            "stack_object_cfg": SceneEntityCfg("stack_object"),
            "max_z_distance": 0.0734, # 0.0112 0.0734604001045227 0.0568 0.0203
        },
        weight=1.0,
    )

    stack_object_xy_precision = RewTerm(
        func=mdp.stack_object_precision_xy_reward,
        params={
            "stack_object_cfg": SceneEntityCfg("stack_object"),
            "target_cube_cfg": SceneEntityCfg("target_cube"),
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
            "stack_height_offset": 0.0365,
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
            "stack_height_offset": 0.0365,
            "height_tolerance": 0.01,
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
            "height_diff": 0.0365,
            "height_threshold": 0.01
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
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
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
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 0.7), "y": (-0.20, 0.20), "yaw": (-1.0, 1, 0)},
            "min_separation": 0.3,
            "asset_cfgs": [SceneEntityCfg("stack_object"), SceneEntityCfg("target_cube")],
        },
    )


@configclass
class FrankaStackBowlEnvCfg(StackEnvCfg):
    """Configuration for the Franka Gelsight Environment."""

    # Override the observations and rewards
    observations: GelsightObservationsCfg = GelsightObservationsCfg()
    rewards: GelsightRewardsCfg = GelsightRewardsCfg()

    left_tactile_sensor: VisuoTactileSensorCfg = VisuoTactileSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_elastomer_link/tactile_sensor",
        update_period=1 / 15,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=True,
        tactile_array_size=(20, 25),
        tactile_margin=0.003,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/stack_object",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_elastomer_tip_link/cam",
            update_period=1 / 15,
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )
    right_tactile_sensor: VisuoTactileSensorCfg = VisuoTactileSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_elastomer_link/tactile_sensor",
        update_period=1 / 15,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=True,
        tactile_array_size=(20, 25),
        tactile_margin=0.003,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/stack_object",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_elastomer_tip_link/cam",
            update_period=1 / 15,
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.sim.physx.enable_ccd = True
        self.decimation = 10
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005  # 100Hz
        self.sim.render_interval = 10

        # Set events
        self.events = EventCfg()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileWithCompliantContactCfg(
                usd_path=LOCAL_ROBOT_USD_PATH,
                activate_contact_sensors=True,
                rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=1
                ),
                collision_props=FRANKA_PANDA_CFG.spawn.collision_props,
                compliant_contact_stiffness=1000.0,
                compliant_contact_damping=100.0,
                physics_material_prim_path=[
                    "left_elastomer_link",
                    "right_elastomer_link",
                ],
            ),
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Add tactile sensors to the scene
        self.scene.left_tactile_sensor = self.left_tactile_sensor
        self.scene.right_tactile_sensor = self.right_tactile_sensor

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint[1-7]"], scale=1.0, use_default_offset=True
        )
        # self.actions.gripper_action = mdp.AbsBinaryJointPositionActionCfg(
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # utilities for gripper status check
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        # Rigid body properties of each cube
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=100.0,
            disable_gravity=False,
        )

        # Set each stacking cube deterministically
        self.scene.stack_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/stack_object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.0113), rot=(1, 0, 0, 0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=LOCAL_BOWL_USD,
                rigid_props=cube_properties,
            ),
        )

        self.scene.target_cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/target_cube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.00, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1, 1, 1),
                rigid_props=cube_properties,
                semantic_tags=[("class", "target_cube")],
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )