import os
import isaaclab.sim as sim_utils


from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim import MeshCollisionPropertiesCfg, SDFMeshPropertiesCfg
from isaaclab.utils import configclass

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import ObservationsCfg, RewardsCfg, StackEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

LOCAL_ROBOT_USD_PATH = "./franka_gelsight.usd"
LOCAL_BLUE_BLOCK_USD  = "./assets/Props/blue_block_sdf.usd"
LOCAL_RED_BLOCK_USD   = "./assets/Props/red_block_sdf.usd"
LOCAL_GREEN_BLOCK_USD = "./assets/Props/green_block_sdf.usd"

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

    rewind_tactile_reward = RewTerm(func=mdp.rewind_tactile_reward, weight=1)


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
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "yaw": (-1.0, 1)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("stack_object"), SceneEntityCfg("target_cube")],
        },
    )


@configclass
class TactileFrankaStackEnvCfg(StackEnvCfg):
    """Configuration for the Franka Gelsight Environment."""

    # Override the observations and rewards
    observations: GelsightObservationsCfg = GelsightObservationsCfg()
    rewards: GelsightRewardsCfg = GelsightRewardsCfg()

    enable_front_cam: bool = os.environ.get("FORGE_ENABLE_FRONT_CAM", "0") == "1"
    """Whether to enable the front-facing camera in the scene and observations."""
    enable_sensor: bool = os.environ.get("FORGE_ENABLE_SENSOR", "0") == "1"

    left_tactile_sensor: VisuoTactileSensorCfg = VisuoTactileSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_elastomer_link/tactile_sensor",
        update_period=1 / 15,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=False,
        enable_force_field=False,
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
        enable_camera_tactile=False,
        enable_force_field=False,
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

        # Set events
        self.events = EventCfg()

        '''
        # Set Franka as robot
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=LOCAL_PEG_INSERT_ROBOT_USD_PATH,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=1,
                    fix_root_link=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "panda_joint1": -0.4536,
                    "panda_joint2": 0.1362,
                    "panda_joint3": 0.3922,
                    "panda_joint4": -2.3182,
                    "panda_joint5": -0.1029,
                    "panda_joint6": 2.223,
                    "panda_joint7": 0.7862,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                },
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            actuators={
                "panda_shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[1-4]"],
                    stiffness=15.0,
                    damping=8.0,
                    friction=0.0,
                    armature=0.0,
                    effort_limit_sim=87,
                    velocity_limit_sim=124.6,
                ),
                "panda_forearm": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[5-7]"],
                    stiffness=15.0,
                    damping=8.0,
                    friction=0.0,
                    armature=0.0,
                    effort_limit_sim=12,
                    velocity_limit_sim=149.5,
                ),
                "panda_hand": ImplicitActuatorCfg(
                    joint_names_expr=["panda_finger_joint[1-2]"],
                    effort_limit_sim=200.0,
                    velocity_limit_sim=0.05,
                    stiffness=2000.0,
                    damping=173.0,
                    friction=0.1,
                    armature=0.0,
                ),
            },
        )
        '''
        solver_count = 48
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=LOCAL_ROBOT_USD_PATH,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=solver_count,
                    solver_velocity_iteration_count=4,
                    max_contact_impulse=1e32,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=solver_count, solver_velocity_iteration_count=4
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.569,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.810,
                    "panda_joint5": 0.0,
                    "panda_joint6": 3.037,
                    "panda_joint7": 0.741,
                    "panda_finger_joint.*": 0.04,
                },
            ),
            actuators={
                "panda_shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[1-4]"],
                    effort_limit_sim=87.0,
                    stiffness=80.0,
                    damping=4.0,
                ),
                "panda_forearm": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[5-7]"],
                    effort_limit_sim=12.0,
                    stiffness=80.0,
                    damping=4.0,
                ),
                "panda_hand": ImplicitActuatorCfg(
                    joint_names_expr=["panda_finger_joint.*"],
                    effort_limit_sim=200.0,
                    stiffness=2e3,
                    damping=1e2,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Add tactile sensors to the scene
        if self.enable_sensor:
            self.left_tactile_sensor.enable_camera_tactile = True
            self.left_tactile_sensor.enable_force_field = True
            self.right_tactile_sensor.enable_camera_tactile = True
            self.right_tactile_sensor.enable_force_field = True
            self.scene.left_tactile_sensor = self.left_tactile_sensor
            self.scene.right_tactile_sensor = self.right_tactile_sensor

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint[1-7]"], scale=0.5, use_default_offset=True
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
        self.cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=solver_count,
            solver_velocity_iteration_count=4,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        self.scene.target_cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/target_cube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=LOCAL_RED_BLOCK_USD,
                scale=(1.0, 1.0, 1.0),
                rigid_props=self.cube_properties,
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

        if self.enable_front_cam:
            # Add front camera to see the boxes and front of robot
            self.scene.front_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/front_cam",
                update_period=0.0,
                height=224,
                width=224,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(1.0, 0.0, 0.4), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"
                ),
            )
