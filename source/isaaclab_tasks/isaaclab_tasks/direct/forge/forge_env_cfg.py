# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg
from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, CtrlCfg, FactoryEnvCfg, ObsRandCfg
from isaaclab_tasks.utils.runtime_cfg import TactileEncoderCfg, TactileSaveCfg, VisualRewardCfg
from isaaclab_tasks.utils.tactile_reward_import import TactileRewardCfg

from .forge_events import randomize_dead_zone
from .forge_tasks_cfg import ForgeGearMesh, ForgeNutThread, ForgePegInsert, ForgeTask


LOCAL_PEG_INSERT_ROBOT_USD_PATH = "./franka_gelsight.usd"

PEG_INSERT_ROBOT_USD_PATH = (
    LOCAL_PEG_INSERT_ROBOT_USD_PATH if os.path.exists(LOCAL_PEG_INSERT_ROBOT_USD_PATH) else FactoryEnvCfg.robot.spawn.usd_path
)

OBS_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})
# Register absolute fixed/held positions for actor obs (already in STATE_DIM_CFG defaults).
OBS_DIM_CFG.update({"fixed_pos": 3, "held_pos": 3})

STATE_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})
OBS_DIM_CFG.update(
    {
        "left_tactile_normal_force": 500,
        "right_tactile_normal_force": 500,
    }
)
STATE_DIM_CFG.update(
    {
        "left_tactile_normal_force": 500,
        "left_tactile_shear_force": 1000,
        "right_tactile_normal_force": 500,
        "right_tactile_shear_force": 1000,
    }
)


@configclass
class ForgeCtrlCfg(CtrlCfg):
    ema_factor_range = [0.025, 0.1]
    default_task_prop_gains = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]
    task_prop_gains_noise_level = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]
    pos_threshold_noise_level = [0.25, 0.25, 0.25]
    rot_threshold_noise_level = [0.29, 0.29, 0.29]
    default_dead_zone = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]


@configclass
class ForgeObsRandCfg(ObsRandCfg):
    fingertip_pos = 0.00025
    fingertip_rot_deg = 0.1
    ft_force = 1.0


@configclass
class EventCfg:
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "mass_distribution_params": (-0.005, 0.005),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    held_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    fixed_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("fixed_asset"),
            "static_friction_range": (0.25, 1.25),  # TODO: Set these values based on asset type.
            "dynamic_friction_range": (0.25, 0.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    dead_zone_thresholds = EventTerm(
        func=randomize_dead_zone,
        mode="interval",
        interval_range_s=(2.0, 2.0),  # (0.25, 0.25)
    )


@configclass
class ForgeEnvCfg(FactoryEnvCfg):
    action_space: int = 7
    obs_rand: ForgeObsRandCfg = ForgeObsRandCfg()
    ctrl: ForgeCtrlCfg = ForgeCtrlCfg()
    task: ForgeTask = ForgeTask()
    events: EventCfg = EventCfg()

    # Dense tactile progress reward. Empty ckpt = disabled.
    tactile_reward: TactileRewardCfg = TactileRewardCfg()

    # Dense visual progress reward (RGB -> DINOv2 -> ReWiND). Empty ckpt = disabled.
    visual_reward: VisualRewardCfg = VisualRewardCfg()

    # Per-episode trajectory dumping. force_field=False disables it.
    tactile_save: TactileSaveCfg = TactileSaveCfg()

    # Frozen tactile encoder for the embedding obs. Empty ckpt = disabled.
    tactile_encoder: TactileEncoderCfg = TactileEncoderCfg()

    ft_smoothing_factor: float = 0.25

    # Optional 3rd-person RGB camera (for video logging / dataset collection).
    # Activated by env var FORGE_ENABLE_FRONT_CAM=1. Independent from tactile
    # sensors — works alongside or without them. Requires --enable_cameras flag.
    enable_front_cam: bool = os.environ.get("FORGE_ENABLE_FRONT_CAM", "0") == "1"

    def _apply_tactile_state_obs(self) -> None:
        """Append the frozen tactile latent to BOTH actor obs and critic state.

        Shared by every task's ``tactile_state`` baseline. Pairs with the task's
        own ``baseline`` reward shaping, so the ONLY delta against
        ``--baseline baseline`` is the extra input modality — which is what makes
        it a clean "tactile as state" ablation against "tactile as reward"
        (TacReward). The latent comes from an autoencoder-pretrained
        ``TactileCNNEncoder`` (see
        ``external/third-party/Tactile-ReWiND/train_tactile_ae.py``); trained for
        reconstruction only, it carries no task/reward information, unlike
        baseline B2's progress-trained encoder.

        The obs/state vectors are sized here, before the checkpoint is read, so
        ``tactile_encoder.dim`` must be declared and must equal the ckpt's
        ``2 * per_hand_dim`` — ``ForgeEnv._init_tactile_encoder`` asserts that at
        startup rather than letting a wrong width reach the policy.

        GelSight RGB is not consumed, so the tactile cameras are switched off:
        force-field / SDF sensing needs no RTX renderer, which keeps the run on
        compute-only cloud GPUs (no ``--enable_cameras``).
        """
        # Fail fast on incompatible combos — clearer than the KeyError or shape
        # error they would otherwise cause deep inside obs assembly. Test the
        # sensors themselves rather than the flag that drops them: __post_init__
        # has already run, so this catches every reason they are absent.
        if self.left_tactile_sensor is None or self.right_tactile_sensor is None:
            raise RuntimeError(
                "baseline tactile_state reads the GelSight force fields, but the "
                "tactile sensors were dropped from the scene (see "
                "FORGE_SKIP_TACTILE_SENSORS in __post_init__)."
            )
        if not (self.tactile_encoder.ckpt or "").strip():
            raise RuntimeError(
                "baseline tactile_state requires env.tactile_encoder.ckpt "
                "(autoencoder ckpt from train_tactile_ae.py)."
            )
        embed_dim = int(self.tactile_encoder.dim)
        if embed_dim <= 0:
            raise RuntimeError(
                "baseline tactile_state requires env.tactile_encoder.dim "
                "(= 2 * per_hand_dim of the ckpt) so the obs vector can be sized "
                "before the ckpt is loaded."
            )

        OBS_DIM_CFG.update({"tactile_embedding": embed_dim})
        STATE_DIM_CFG.update({"tactile_embedding": embed_dim})
        self.obs_order = list(self.obs_order) + ["tactile_embedding"]
        self.state_order = list(self.state_order) + ["tactile_embedding"]

        # Force-field only: drop the tactile RGB camera pipeline for speedup.
        self.left_tactile_sensor.enable_camera_tactile = False
        self.right_tactile_sensor.enable_camera_tactile = False

    def _attach_front_cam_if_enabled(self) -> None:
        """Attach a 3rd-person RGB camera to self.scene when enabled.

        Called from each task-specific cfg's __post_init__ after the scene is
        set up. The camera is positioned 1 m in front of the robot, 0.4 m above
        the table, looking down at the workspace — matches the StackTactileEnv
        offset since the Franka/table layout is the same.
        """
        if not self.enable_front_cam:
            return
        # Use TiledCameraCfg (not CameraCfg) — it batches all envs into a
        # single tiled render target, so the GPU only allocates one parameter
        # block per type instead of `num_envs` of them. With 128+ envs and the
        # GelSight sensors also rendering, regular CameraCfg exhausts the RTX
        # descriptor pool and throws "Failed to allocate ParameterBlock".
        self.scene.front_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/front_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                # Bumped from (0.1, 2.0) to (0.1, 5.0) since the back-off
                # below puts more of the workspace beyond the old 2 m far clip.
                clipping_range=(0.1, 2.5),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                # Moved camera back from 1.0 m to 1.6 m and up from 0.4 to 0.5
                # so the Franka arm doesn't crowd the foreground.
                pos=(1.2, 0.0, 0.4),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros",
            ),
        )

    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "ft_force",
        "force_threshold",
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "task_prop_gains",
        "ema_factor",
        "ft_force",
        "pos_threshold",
        "rot_threshold",
        "force_threshold",
    ]


@configclass
class ForgeTaskPegInsertCfg(ForgeEnvCfg):
    task_name = "peg_insert"
    task = ForgePegInsert()
    episode_length_s = 10.0
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "ft_force",
        "force_threshold",
        "left_tactile_normal_force",
        "right_tactile_normal_force",
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "task_prop_gains",
        "ema_factor",
        "ft_force",
        "pos_threshold",
        "rot_threshold",
        "force_threshold",
        "left_tactile_normal_force",
        "left_tactile_shear_force",
        "right_tactile_normal_force",
        "right_tactile_shear_force",
    ]

    left_tactile_sensor: VisuoTactileSensorCfg = VisuoTactileSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_elastomer_link/tactile_sensor",
        update_period=1 / 15,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=True,
        tactile_array_size=(20, 25),
        tactile_margin=0.003,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/HeldAsset",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_elastomer_tip_link/cam",
            update_period=1 / 15,  # match VisuoTactileSensor update_period — was 1/60 (4× wasted renders)
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
        contact_object_prim_path_expr="{ENV_REGEX_NS}/HeldAsset",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_elastomer_tip_link/cam",
            update_period=1 / 15,  # match VisuoTactileSensor update_period — was 1/60 (4× wasted renders)
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation
        self.scene.replicate_physics = False
        self.scene.clone_in_fabric = False
        # Speed escape hatch: when FORGE_SKIP_TACTILE_SENSORS=1, drop the
        # GelSight sensors from the cfg BEFORE InteractiveScene auto-detects
        # them (its attribute scan instantiates anything that is-a
        # SensorBaseCfg, so a runtime _setup_scene skip is too late — the
        # camera gets spawned during the parent scene init).
        if os.getenv("FORGE_SKIP_TACTILE_SENSORS", "0") == "1":
            self.left_tactile_sensor = None
            self.right_tactile_sensor = None
        # Optional 3rd-person RGB camera (env var FORGE_ENABLE_FRONT_CAM=1).
        self._attach_front_cam_if_enabled()
        self.robot = self.robot.replace(
            spawn=sim_utils.UsdFileWithCompliantContactCfg(
                usd_path=PEG_INSERT_ROBOT_USD_PATH,
                activate_contact_sensors=True,
                rigid_props=self.robot.spawn.rigid_props,
                articulation_props=self.robot.spawn.articulation_props,
                collision_props=self.robot.spawn.collision_props,
                compliant_contact_stiffness=1000.0,
                compliant_contact_damping=100.0,
                physics_material_prim_path=[
                    "left_elastomer_link",
                    "right_elastomer_link",
                ],
            )
        )


@configclass
class ForgeTaskGearMeshCfg(ForgeEnvCfg):
    task_name = "gear_mesh"
    task = ForgeGearMesh()
    episode_length_s = 20.0

    left_tactile_sensor: VisuoTactileSensorCfg = VisuoTactileSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_elastomer_link/tactile_sensor",
        update_period=1 / 15,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=True,
        tactile_array_size=(20, 25),
        tactile_margin=0.003,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/HeldAsset",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_elastomer_tip_link/cam",
            update_period=1 / 15,  # match VisuoTactileSensor update_period — was 1/60 (4× wasted renders)
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
        contact_object_prim_path_expr="{ENV_REGEX_NS}/HeldAsset",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_elastomer_tip_link/cam",
            update_period=1 / 15,  # match VisuoTactileSensor update_period — was 1/60 (4× wasted renders)
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation
        self.scene.replicate_physics = False
        self.scene.clone_in_fabric = False
        # Speed escape hatch: when FORGE_SKIP_TACTILE_SENSORS=1, drop the
        # GelSight sensors from the cfg BEFORE InteractiveScene auto-detects
        # them (its attribute scan instantiates anything that is-a
        # SensorBaseCfg, so a runtime _setup_scene skip is too late — the
        # camera gets spawned during the parent scene init).
        if os.getenv("FORGE_SKIP_TACTILE_SENSORS", "0") == "1":
            self.left_tactile_sensor = None
            self.right_tactile_sensor = None
        # Optional 3rd-person RGB camera (env var FORGE_ENABLE_FRONT_CAM=1).
        self._attach_front_cam_if_enabled()
        self.robot = self.robot.replace(
            spawn=sim_utils.UsdFileWithCompliantContactCfg(
                usd_path=PEG_INSERT_ROBOT_USD_PATH,
                activate_contact_sensors=True,
                rigid_props=self.robot.spawn.rigid_props,
                articulation_props=self.robot.spawn.articulation_props,
                collision_props=self.robot.spawn.collision_props,
                compliant_contact_stiffness=1000.0,
                compliant_contact_damping=100.0,
                physics_material_prim_path=[
                    "left_elastomer_link",
                    "right_elastomer_link",
                ],
            )
        )


@configclass
class ForgeTaskNutThreadCfg(ForgeEnvCfg):
    task_name = "nut_thread"
    task = ForgeNutThread()
    episode_length_s = 30.0
    # NutThread baseline A: drop F/T sensing from actor + critic, expose absolute
    # bolt (`fixed_pos`) and nut (`held_pos`) positions to the actor for the
    # randomized-init variant.
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        # "ft_force",        # commented for no-tactile / no-F/T baseline
        # "force_threshold", # commented for no-tactile / no-F/T baseline
        "fixed_pos",
        "held_pos",
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "task_prop_gains",
        "ema_factor",
        # "ft_force",        # commented for no-F/T baseline
        "pos_threshold",
        "rot_threshold",
        # "force_threshold", # commented for no-F/T baseline
    ]


    left_tactile_sensor: VisuoTactileSensorCfg = VisuoTactileSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_elastomer_link/tactile_sensor",
        update_period=1 / 15,
        render_cfg=GELSIGHT_R15_CFG,
        enable_camera_tactile=True,
        enable_force_field=True,
        tactile_array_size=(20, 25),
        tactile_margin=0.003,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/HeldAsset",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_elastomer_tip_link/cam",
            update_period=1 / 15,  # match VisuoTactileSensor update_period — was 1/60 (4× wasted renders)
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
        contact_object_prim_path_expr="{ENV_REGEX_NS}/HeldAsset",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_elastomer_tip_link/cam",
            update_period=1 / 15,  # match VisuoTactileSensor update_period — was 1/60 (4× wasted renders)
            height=GELSIGHT_R15_CFG.image_height,
            width=GELSIGHT_R15_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation
        self.scene.replicate_physics = False
        self.scene.clone_in_fabric = False
        # Speed escape hatch: when FORGE_SKIP_TACTILE_SENSORS=1, drop the
        # GelSight sensors from the cfg BEFORE InteractiveScene auto-detects
        # them (its attribute scan instantiates anything that is-a
        # SensorBaseCfg, so a runtime _setup_scene skip is too late — the
        # camera gets spawned during the parent scene init).
        if os.getenv("FORGE_SKIP_TACTILE_SENSORS", "0") == "1":
            self.left_tactile_sensor = None
            self.right_tactile_sensor = None
        # Optional 3rd-person RGB camera (env var FORGE_ENABLE_FRONT_CAM=1).
        self._attach_front_cam_if_enabled()
        # Bump hand_init_pos[2] from the default 1.5 cm to 3.5 cm (matching GearMesh).
        # The franka_gelsight fingertip is thicker than the stock Franka fingertip,
        # so 1.5 cm clearance above the bolt tip puts the gelsight tip inside / below
        # the bolt — IK fails to converge, falls back to default pose, and the nut
        # never lands in the gripper.
        self.task.hand_init_pos = [0.0, 0.0, 0.035]
        # Bolt only randomizes in XY (no z noise) so the bolt always sits on the table.
        # Nut (in gripper) gets a wider XY range so the policy sees more variation in
        # gripper-relative-to-bolt offsets without ending up too far from the bolt.
        self.task.fixed_asset_init_pos_noise = [0.05, 0.05, 0.0]
        self.task.hand_init_pos_noise = [0.04, 0.04, 0.02]
        self.robot = self.robot.replace(
            spawn=sim_utils.UsdFileWithCompliantContactCfg(
                usd_path=PEG_INSERT_ROBOT_USD_PATH,
                activate_contact_sensors=True,
                rigid_props=self.robot.spawn.rigid_props,
                articulation_props=self.robot.spawn.articulation_props,
                collision_props=self.robot.spawn.collision_props,
                compliant_contact_stiffness=1000.0,
                compliant_contact_damping=100.0,
                physics_material_prim_path=[
                    "left_elastomer_link",
                    "right_elastomer_link",
                ],
            )
        )
