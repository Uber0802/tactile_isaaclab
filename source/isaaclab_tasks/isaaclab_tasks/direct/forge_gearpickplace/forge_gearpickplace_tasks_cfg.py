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
    # Dense scales /10 (2026-05-23) so sparse curr_success (×5 in factory_env)
    # dominates the gradient near the goal.
    approach_reward_scale: float = 0.1
    approach_scale: float = 0.05
    gear_grasp_z_offset: float = 0.005

    # Lift reward: encourages raising the gear above its initial table height.
    lift_reward_scale: float = 0.4
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
    descent_reward_scale: float = 0.3
    descent_z_scale: float = 0.005
    # Continuous fine XY gate — exp(-d/scale): d=2.5 mm → 0.61, d=5 mm → 0.37.
    xy_alignment_scale: float = 0.005

    # Transport-phase XY alignment — coarse scale that gives gradient at 5–10 cm
    # offsets (fine `xy_alignment_scale` saturates to ~0 past ~2 cm). Gated on
    # `r_lift` so the policy can't farm it by sliding the gear on the table.
    # Without this, after grasp+lift the policy has no signal pulling the gear
    # horizontally toward the bolt, and `r_descent` only fires once both xy and
    # z are already inside ~5 mm.
    xy_align_reward_scale: float = 0.3
    xy_coarse_scale: float = 0.05  # exp(-d/0.05): d=5 cm → 0.37, d=10 cm → 0.14

    # Sharp version of r_xy_align — bridges the coarse 5 cm scale and the fine
    # 5 mm scale (used by r_descent). Gives a stronger gradient pulling the gear
    # right over the post in the last ~1–2 cm before fine descent kicks in.
    # Gated on r_lift just like the coarse term.
    xy_align_sharp_reward_scale: float = 0.3
    xy_align_sharp_scale: float = 0.01  # exp(-d/0.01): d=1 cm → 0.37, d=2 cm → 0.14

    # Once XY is roughly aligned, reward driving Z down. Gated on the coarse
    # `xy_coarse` alignment so the policy doesn't get z-descent reward while
    # still in transit far from the bolt, and on `r_lift` so it can't be farmed
    # by leaving the gear on the table.
    # `r_descent` (fine 5 mm × 5 mm) handles the final 1–2 cm; this term keeps
    # gradient alive in the 2–10 cm height-above-target range.
    z_align_reward_scale: float = 0.3
    z_coarse_scale: float = 0.05  # exp(-z/0.05): z=5 cm → 0.37, z=10 cm → 0.14

    # Yaw alignment reward — gear_yaw must match fixed_yaw within ~6° for success
    # (geometric-base offsets only cancel when yaws agree). Gated on xy alignment
    # so the policy isn't pulled to spin the gripper while still in transit.
    yaw_reward_scale: float = 1.0
    yaw_alignment_scale: float = 0.1  # rad ≈ 6°
    # Z-proximity gate on r_yaw — exp(-z_dist / scale). Only reward yaw
    # alignment once the gear is genuinely close to the meshing pose, so the
    # policy doesn't spin the gripper at 5 cm height while ignoring descent.
    # 2026-07-14: widened 0.02 → 0.05 so the yaw gradient is alive during
    # transport (z_dist=5 cm → 0.37 instead of 0.08); the hover-trap this
    # guarded against is now handled by the dense depth reward.
    yaw_z_gate_scale: float = 0.05

    # Dense depth reward — pulls gear from "hover above bolt" past the
    # success boundary. Linear from 0 at z_disp=0 (target z) to 1 at
    # z_disp=ideal_z_disp (15 mm below target for A_hard_success). Gated by
    # r_lift × xy_strict so it ONLY fires when xy is inside the 2.5 mm
    # success criterion — forces "align xy first, then press down". Big scale
    # (1.0) so the descent step dominates the "hover trap".
    depth_reward_scale: float = 1.0
    # Window above target z (m) over which r_depth starts ramping. At
    # gear_z = +depth_approach_scale the reward is 0.
    depth_approach_scale: float = 0.02
    # Depth below target z (m) at which r_depth reaches its max of 1.0 (the gear
    # is "seated"). 2026-07-27: added so r_depth keeps a downward gradient PAST
    # the target plane instead of saturating at z=0. Previously the ramp hit 1.0
    # at the target plane and went flat below it, so the policy learned to rest
    # exactly at z_disp≈0 (engaged but not seated) — and success (which needs
    # z_disp < 0) never fired. Now r_depth is a single continuous ramp:
    #   z = +depth_approach_scale (2 cm above) → 0
    #   z = 0 (target plane)                   → 0.8  (still climbing, not flat)
    #   z = -depth_seat_scale (5 mm below)     → 1.0  (seated)
    # so there is a constant downward pull all the way through the plane. Keep
    # it aligned with success_threshold (-0.1 ≈ 5 mm) so the reward saturates
    # right where success fires.
    depth_seat_scale: float = 0.005

    # XY gate on the z-progress rewards (r_z_descend, r_depth).
    # 2026-07-14: widened 0.0025 -> 0.0075. Matching the gate to the 2.5 mm
    # success criterion made it a winner-take-all cliff: r_depth carries the
    # largest weight (1.0) and r_z_descend 0.3, and both were multiplied by
    # exp(-xy/0.0025), which is ~0.09 at xy=6 mm. Episodes that ended even
    # slightly wide got essentially no "press down" gradient, so they could
    # never recover — success went bimodal (~52% nail it, ~48% stuck outside
    # with no signal) and plateaued. The success *criterion* should not double
    # as the *learning* gate. At 0.0075: xy=2.5 mm → 0.72, 6 mm → 0.45,
    # 10 mm → 0.26 — near-misses keep a usable pull inward and down.
    xy_strict_gate_scale: float = 0.0075
