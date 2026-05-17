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

    # Source-hole pose relative to destination hole, sampled per env at reset.
    # `source_hole_offset_range` is the half-range of a uniform U(-r, +r) sample on each axis
    # (x, y, z), applied in world frame. `source_hole_min_distance` is the minimum allowed
    # XY centre-to-centre distance; samples below it are rejected so the two hole bases
    # never overlap.
    source_hole_offset_range: list = [0.10, 0.10, 0.0]
    source_hole_min_distance: float = 0.05
    # Optional deterministic source-hole offset. When non-None it overrides the
    # rejection-sampled offset above so every env spawns the source hole at the
    # exact same place relative to the destination hole. Used by the `single_pos`
    # baseline for deterministic tactile trajectories. None = original behavior.
    source_hole_fixed_offset: list = None

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
    # the destination base. Widened from 1 cm → 4 cm so descent gradient appears well
    # before the policy stumbles into a tight gate; the continuous `r_xy_align` below
    # supplies the actual horizontal pull.
    #   r_descent = [xy_dist < thresh] * exp(-max(0, z_above_dest) / descent_z_scale)
    descent_reward_scale: float = 3.0
    descent_xy_threshold: float = 0.04
    descent_z_scale: float = 0.01       # at 3 cm → 0.05, at 1 cm → 0.37, at 0 → 1

    # Continuous XY-alignment reward (mirrors GearMesh PickPlace). Provides a dense
    # horizontal gradient pulling the peg toward the destination during the transport
    # phase, since Baseline A's only prior xy signal was the binary descent gate.
    # Gated on `r_lift` so the policy can't farm it by sliding the peg along the table.
    #   r_xy_align = r_lift * exp(-peg_to_dest_xy / xy_coarse_scale)
    xy_align_reward_scale: float = 1.5
    xy_coarse_scale: float = 0.05  # exp(-d/0.05): d=5 cm → 0.37, d=10 cm → 0.14

    # Z-descent bridge (mirrors NutThread PickPlace `r_z_descend`). The sharp
    # `r_descent` above only fires within ~1 cm of the destination z, so during
    # transport (peg lifted, xy-roughly-aligned, but still 2–10 cm above the hole)
    # the policy has no descent gradient — it XY-aligns and then plateaus. This
    # coarse term gives a smooth downward pull from ~10 cm. Gated on `r_lift *
    # xy_coarse` so it can't be farmed by an unlifted or unaligned peg.
    #   r_z_descend = r_lift * xy_coarse * exp(-max(0, z_above_dest) / z_coarse_scale)
    z_align_reward_scale: float = 1.5
    z_coarse_scale: float = 0.05  # exp(-z/0.05): z=5 cm → 0.37, z=10 cm → 0.14

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
