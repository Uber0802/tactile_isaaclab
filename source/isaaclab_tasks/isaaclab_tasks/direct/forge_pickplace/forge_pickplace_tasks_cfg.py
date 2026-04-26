# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import Hole8mm
from isaaclab_tasks.direct.forge.forge_tasks_cfg import ForgePegInsert


@configclass
class ForgePegInsertPickPlace(ForgePegInsert):
    """Pick-and-place variant: peg starts inside a source hole; place it in the destination hole."""

    # Offset of source hole relative to destination hole (x, y, z), applied in world frame at reset.
    source_hole_offset: list = [0.0, 0.10, 0.0]

    # Gripper starts clearly above the peg top (peg sticks ~2.5 cm above the source hole),
    # leaving room for the fingers to descend and grasp.
    hand_init_pos: list = [0.0, 0.0, 0.10]
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn_noise: list = [0.0, 0.0, 0.2]

    # Franka finger joint width when the gripper is fully open (meters, each finger).
    gripper_open_width: float = 0.04

    # Insertion depth of the peg into the source hole at reset.
    # 0 places the peg base at the hole base (peg sticks out (peg_height - hole_height)).
    peg_source_insertion: float = 0.0

    # Approach reward: fingertip → peg grasp point (peg top minus 1/4 peg height).
    # r_approach = exp(-dist / approach_scale), in [0, 1]. Smaller scale = sharper gradient.
    approach_reward_scale: float = 1.0
    approach_scale: float = 0.03
    # Grasp target on the peg along peg-local +z from the peg base.
    # Default = 0.75 * peg_height (= 0.0375 for an 8 mm peg), i.e. one-quarter down from the peg top.
    peg_grasp_z_offset: float = 0.0375

    # Heuristic thresholds used only for logging "peg grasped" (not for reward).
    grasp_log_dist_threshold: float = 0.01
    grasp_log_gripper_threshold: float = 0.01

    # Gripper EMA factor used instead of the body EMA. Higher = more responsive but also more
    # likely to pinch and eject the peg. 0.3 was chosen empirically: ~6× faster than the body
    # EMA (~0.05) while still spreading a close/open command over ~4 env steps.
    gripper_ema: float = 0.3

    # Peg-speed penalty keeps physics from launching the peg during aggressive closes.
    # Penalty = speed_penalty_scale * relu(|peg_velocity| - peg_speed_threshold).
    peg_speed_threshold: float = 0.5  # m/s
    peg_speed_penalty_scale: float = 0.5

    # Lift reward: encourages raising the peg above the source-hole tip (clearance for pull-out).
    # r_lift = tanh(peg_z_above_source_base / lift_scale), so tiny lifts already give clear
    # feedback and the reward saturates to 1 once the peg comes out cleanly.
    # 2.0 matches the 4/19 run that learned pick + transport to destination top.
    lift_reward_scale: float = 2.0
    lift_scale: float = 0.015
    lift_clear_margin: float = 0.02  # kept for logging compat; not used by the new reward.

    # Descent reward (hard-gated on XY alignment): once the peg is XY-aligned within
    # `descent_xy_threshold` of the destination centre, reward descending the peg toward
    # the destination base. Outside the gate this term is strictly 0 so it can't distort
    # the reach/lift phase the 4/19 policy already learned.
    #   r_descent = [xy_dist < thresh] * exp(-max(0, z_above_dest) / descent_z_scale)
    descent_reward_scale: float = 3.0
    descent_xy_threshold: float = 0.01  # 1 cm — policy must first line up XY
    descent_z_scale: float = 0.01       # at 3 cm → 0.05, at 1 cm → 0.37, at 0 → 1

    source_fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/SourceFixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=Hole8mm().usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=Hole8mm().mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.1, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
