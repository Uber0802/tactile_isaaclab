# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.direct.forge.forge_tasks_cfg import ForgeNutThread


@configclass
class ForgeNutThreadPickPlace(ForgeNutThread):
    """Pick-and-place variant: M16 nut rests on the table; pick it up and thread onto the bolt."""

    # Nut's table position relative to the bolt, plus XY noise (Z stays on table).
    # Pulled in -y so the nut isn't on top of the bolt or the bolt head footprint.
    nut_table_offset: list = [0.0, -0.10, 0.0]
    nut_table_pos_noise: list = [0.03, 0.03, 0.0]

    # Nut yaw range when lying flat on the table (rad). Hex symmetry is 60°, so a
    # ±π range still spans every distinguishable orientation many times over.
    nut_table_yaw_range: float = 3.14159

    # Bolt yaw — wider than the base NutThread (±15°) so the policy generalizes
    # across all bolt orientations the pick-place arm might encounter.
    fixed_asset_init_orn_range_deg: float = 360.0

    # Drop z noise on the bolt. Parent default is [0.05, 0.05, 0.05] which would
    # randomize bolt height ±5 cm — bolt could float or sink, making the descent
    # target inconsistent across resets.
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.0]

    # Gripper starts above the nut on the table.
    hand_init_pos: list = [0.0, 0.0, 0.10]
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn_noise: list = [0.0, 0.0, 0.2]

    # Franka finger joint width when the gripper is fully open (meters, each finger).
    # Nut hex spans ~24 mm (M16), so 40 mm is comfortably open.
    gripper_open_width: float = 0.04

    # Faster gripper EMA so the policy can actually close.
    gripper_ema: float = 0.3

    # Reach reward: fingertip → nut grasp point.
    # Nut is only 10 mm tall — aim near the mid-section so closing the gripper
    # pinches the hex flats instead of slamming the top face.
    approach_reward_scale: float = 0.3
    approach_scale: float = 0.05
    nut_grasp_z_offset: float = 0.005

    # Lift reward: encourages raising the nut above its initial table height.
    lift_reward_scale: float = 2.0
    lift_scale: float = 0.01
    # Softer proximity gate (out to ~10 cm) so gradient survives during approach.
    lift_proximity_scale: float = 0.1

    # Nut-speed penalty keeps physics from launching the nut during aggressive closes.
    nut_speed_threshold: float = 0.3  # m/s
    nut_speed_penalty_scale: float = 0.5

    # Descent reward — drives nut's geometric base toward the bolt-thread target
    # used by the success check. Continuous exp gates so the policy keeps gradient
    # all the way down to the 2.5 mm / 0.75 mm success tolerances.
    descent_reward_scale: float = 1.5
    descent_z_scale: float = 0.005
    # Continuous XY gate scale. exp(-d/scale): d=2.5 mm → 0.61, d=5 mm → 0.37.
    xy_alignment_scale: float = 0.005

    # Transport-phase XY alignment — coarse scale that gives gradient at 5–10 cm
    # offsets (fine `xy_alignment_scale` saturates to ~0 past ~2 cm). Gated on
    # `r_lift` so the policy can't farm it by sliding the nut on the table.
    # Without this, after grasp+lift the policy has no signal pulling the nut
    # horizontally toward the bolt, and `r_descent` only fires once both xy and
    # z are already inside ~5 mm.
    xy_align_reward_scale: float = 1.0
    xy_coarse_scale: float = 0.05  # exp(-d/0.05): d=5 cm → 0.37, d=10 cm → 0.14

    # Once XY is roughly aligned, reward driving Z down. Gated on the coarse
    # `xy_coarse` alignment so the policy doesn't get z-descent reward while
    # still in transit far from the bolt, and on `r_lift` so it can't be farmed
    # by leaving the nut on the table at z=0 above the bolt origin.
    # `r_descent` (fine 5 mm × 5 mm) handles the final 1–2 cm; this term keeps
    # gradient alive in the 2–10 cm height-above-target range.
    z_align_reward_scale: float = 1.0
    z_coarse_scale: float = 0.05  # exp(-z/0.05): z=5 cm → 0.37, z=10 cm → 0.14

    # Wrist-yaw reward — success.check_rot=True for nut_thread requires
    # `wrap_yaw(fingertip_yaw) < 0`. `unidirectional_rot` only biases the
    # action; without dense yaw shaping the policy gets no gradient pulling
    # wrist into the success window until xy/z/yaw all align simultaneously.
    # Gated on `xy_coarse` so the policy doesn't waste yaw budget mid-transit.
    yaw_reward_scale: float = 1.0
    yaw_progress_scale: float = 0.5  # rad ≈ 28.6°; exp(-yaw/0.5): yaw=π/2 → 0.04
