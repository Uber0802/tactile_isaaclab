# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.direct.forge.forge_tasks_cfg import ForgeGearMesh


@configclass
class ForgeGearMeshPickPlace(ForgeGearMesh):
    """Pick-and-place variant: medium gear rests on the table; pick it up and mesh on the gear post."""

    # Gear's table position relative to the gear base (in BASE LOCAL frame, see
    # randomize_initial_state for the rotation). Bumped from -0.07 to -0.10 to
    # leave more clearance from the rectangular plate edge — at 7 cm the noise
    # box (±3 cm) reaches in to ~4 cm from base centre, which can clip the plate.
    # Still well short of the -0.15 that broke lift exploration.
    gear_table_offset: list = [0.0, -0.10, 0.0]
    gear_table_pos_noise: list = [0.03, 0.03, 0.0]

    # Gear yaw range when lying flat on the table (rad). Full 2*pi = arbitrary spin.
    gear_table_yaw_range: float = 3.14159

    # Gear base (mesh) yaw is fully randomized — medium peg sweeps around the base centre.
    fixed_asset_init_orn_range_deg: float = 360.0

    # Drop z noise on the gear base. Parent default is [0.05, 0.05, 0.05] which
    # randomizes peg height ±5 cm — peg can end up floating 10 cm above the table
    # or sunk below it, making the descent target inconsistent across resets.
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.0]

    # Gripper starts above the gear's table position.
    hand_init_pos: list = [0.0, 0.0, 0.10]
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn_noise: list = [0.0, 0.0, 0.2]

    # Franka finger joint width when the gripper is fully open (meters, each finger).
    gripper_open_width: float = 0.04

    # Faster gripper EMA so the policy can actually close.
    gripper_ema: float = 0.3

    # Reach reward: fingertip → gear grasp point.
    # Aim fingertip at the gear mid-section (gear bottom + ~5 mm) so closing the
    # gripper actually pinches the gear sides instead of slamming the top and
    # ejecting it. Held low so approach can't dominate lift.
    approach_reward_scale: float = 0.3
    approach_scale: float = 0.05
    gear_grasp_z_offset: float = 0.005

    # Lift reward: encourages raising the gear above its initial table height.
    lift_reward_scale: float = 2.0
    lift_scale: float = 0.01
    # Softer proximity gate (was /0.05). The previous gate collapsed to ~0 the
    # instant the gear was nudged away, killing the gradient before the policy
    # could discover grasp; /0.1 keeps signal alive out to ~10 cm.
    lift_proximity_scale: float = 0.1


    # Gear-speed penalty keeps physics from launching the gear during aggressive closes.
    # Scale weakened 10x (0.5 → 0.05): grasping naturally bumps the gear to ~1 m/s,
    # and the old penalty (-0.5) outweighed approach reward (+0.3 peak), so policy
    # learned to hover near the gear without ever touching it. Now penalty is a
    # gentle regularizer instead of a dominant negative signal.
    gear_speed_threshold: float = 0.3  # m/s
    gear_speed_penalty_scale: float = 0.05

    # Descent reward — drives gear's geometric base toward the success target
    # (`get_target_held_base_pose`, i.e., medium-peg position with the gear-base
    # offset already applied). Continuous exp gates so the policy keeps gradient
    # all the way down to the 2.5 mm / 1 mm success tolerances.
    descent_reward_scale: float = 1.5
    descent_z_scale: float = 0.005
    # Continuous fine XY gate — exp(-d/scale): d=2.5 mm → 0.61, d=5 mm → 0.37.
    xy_alignment_scale: float = 0.005

    # Transport-phase XY alignment — coarse scale that gives gradient at 5–10 cm
    # offsets (fine `xy_alignment_scale` saturates to ~0 past ~2 cm). Gated on
    # `r_lift` so the policy can't farm it by sliding the gear on the table.
    # Without this, after grasp+lift the policy has no signal pulling the gear
    # horizontally toward the bolt, and `r_descent` only fires once both xy and
    # z are already inside ~5 mm.
    xy_align_reward_scale: float = 1.0
    xy_coarse_scale: float = 0.05  # exp(-d/0.05): d=5 cm → 0.37, d=10 cm → 0.14

    # Once XY is roughly aligned, reward driving Z down. Gated on the coarse
    # `xy_coarse` alignment so the policy doesn't get z-descent reward while
    # still in transit far from the bolt, and on `r_lift` so it can't be farmed
    # by leaving the gear on the table.
    # `r_descent` (fine 5 mm × 5 mm) handles the final 1–2 cm; this term keeps
    # gradient alive in the 2–10 cm height-above-target range.
    z_align_reward_scale: float = 1.0
    z_coarse_scale: float = 0.05  # exp(-z/0.05): z=5 cm → 0.37, z=10 cm → 0.14

    # Yaw alignment reward — gear_yaw must match fixed_yaw within ~6° for success
    # (geometric-base offsets only cancel when yaws agree). Gated on xy alignment
    # so the policy isn't pulled to spin the gripper while still in transit.
    yaw_reward_scale: float = 1.0
    yaw_alignment_scale: float = 0.1  # rad ≈ 6°
