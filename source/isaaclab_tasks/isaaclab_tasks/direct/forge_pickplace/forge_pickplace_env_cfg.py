# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeCtrlCfg, ForgeTaskPegInsertCfg

from .forge_pickplace_tasks_cfg import ForgePegInsertPickPlace


# Peg pick-place needs richer geometric cues than the base peg-insert task.
# `held_pos` / `fixed_pos` are already registered for the actor in the parent
# ForgeEnvCfg; here we add the finger-relative-to-peg vector, the peg quaternion,
# and the peg-relative-to-destination vector so the policy gets direct shortcuts
# for grasp / transport / place instead of having to reconstruct them.
OBS_DIM_CFG.update(
    {
        "fingertip_pos_rel_held": 3,
        "held_pos_rel_fixed": 3,
        "held_quat": 4,
    }
)


@configclass
class PickPlaceCtrlCfg(ForgeCtrlCfg):
    # Action frame stays centered on the destination hole, but the policy must also
    # reach the source hole (~10 cm away), so widen the absolute target bounds.
    pos_action_bounds = [0.15, 0.2, 0.2]


@configclass
class ForgeTaskPegInsertPickPlaceCfg(ForgeTaskPegInsertCfg):
    task_name = "peg_insert"
    task = ForgePegInsertPickPlace()
    ctrl: PickPlaceCtrlCfg = PickPlaceCtrlCfg()
    # Pick-place is a longer-horizon task (grasp → lift → move → place).
    episode_length_s = 20.0
    # 6 pose dims + success prediction (index 6) + gripper (index 7).
    action_space: int = 8

    # Actor obs — BASELINE A (no tactile/force sensing). Expose finger pose plus
    # absolute peg / hole positions so the policy can locate the peg on the
    # source side and the destination hole without any force cues.
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_pos_rel_held",          # finger → peg vector: grasp-phase shortcut
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        # "ft_force",                    # commented for no-tactile baseline
        # "force_threshold",             # commented for no-tactile baseline
        # "left_tactile_normal_force",   # commented for no-tactile baseline
        # "right_tactile_normal_force",  # commented for no-tactile baseline
        "held_pos",
        "held_pos_rel_fixed",              # peg → destination vector: transport/place shortcut
        "held_quat",                       # peg orientation: pre-insertion alignment
        "fixed_pos",
    ]

    # Critic state — also strip force-related entries so the asymmetric critic
    # can't leak F/T information back into value estimates / gradients during
    # the no-tactile baseline. Mirror the parent ForgeEnvCfg state_order minus
    # `ft_force` and `force_threshold`.
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
        # "ft_force",        # commented for no-tactile baseline
        "pos_threshold",
        "rot_threshold",
        # "force_threshold", # commented for no-tactile baseline
    ]

    # ------------------------------------------------------------------
    # Baseline switch (called by train.py after env_cfg is loaded).
    # Baseline A = original frozen reference (no-op). Future baselines
    # (B, C, D, ...) are dispatched here without touching A's code path.
    # ------------------------------------------------------------------
    def apply_baseline(self, baseline: str) -> None:
        if baseline == "baseline":
            self._apply_baseline_baseline()
            return
        if baseline == "naive":
            self._apply_baseline_naive()
            return
        if baseline == "single_pos":
            self._apply_baseline_single_pos()
            return
        raise ValueError(
            f"Unknown baseline {baseline!r} for ForgeTaskPegInsertPickPlaceCfg. "
            f"Implemented: baseline (half transport bridges + tight 1cm descent gate), "
            f"naive (all peg-specific dense shaping off — sparse curr_success only), "
            f"single_pos (baseline obs + all reset randomization zeroed, "
            f"source hole at +10cm X from destination)."
        )

    def _apply_baseline_baseline(self) -> None:
        """baseline (was A_legacy): midway between full shaping (easy) and
        the strict May-15 setting (no transport bridges, baseline stuck at 0%
        after 57h). Designed so baselineA can solve the task eventually but
        slower than today's default, leaving clean headroom for tactile reward
        to demonstrate a measurable speedup / final-success boost.

        Difference vs current A (today's default):
          - `xy_align_reward_scale = 0.5`: today's coarse XY bridge is 1.5;
            halved here so transport-phase XY pull is weaker but non-zero
            (earlier 0.0 left baseline floating 67mm above target indefinitely).
          - `z_align_reward_scale = 0.5`: today's coarse Z bridge is 1.5;
            halved here so the policy still gets a descent gradient during
            transport, just weaker.
          - `descent_xy_threshold = 0.01`: today's value is 4cm; reverting to
            1cm keeps the final-insertion gate strict so the policy must
            actually align precisely before getting r_descent.

        Result vs original strict-0 A_legacy: baselineA should reach success
        eventually (slow), tactile still wins by providing the explicit
        peg-in-hole contact signal that supplements the half-strength bridges.
        Vs default A: baseline is slower (half bridges + tight descent gate),
        so tactile's contribution is measurable rather than buried in noise.
        """
        self.task.xy_align_reward_scale = 0.5
        self.task.z_align_reward_scale = 0.5
        self.task.descent_xy_threshold = 0.01

    def _apply_baseline_naive(self) -> None:
        """naive (was A_naive): kill ALL peg-specific dense shaping — leave only the
        factory base (kp_baseline/coarse/fine, action penalties, curr_engaged,
        curr_success) so the only "learn-the-task" signal comes from the
        sparse `curr_success` (and curr_engaged) bonus. Designed as the
        backdrop for adding ONE external reward (e.g. visual reward shaping
        via FORGE_VISUAL_REWARD_CKPT) and isolating its contribution — none
        of the peg-specific hand-crafted shaping interferes.

        What stays on (factory_env._get_rewards):
          - kp_baseline / kp_coarse / kp_fine   (× 0.1)
          - action_penalty_ee / action_grad_penalty
          - curr_engaged                        (× 1.0)
          - curr_success                        (× 50.0)
        """
        self.task.approach_reward_scale = 0.0
        self.task.lift_reward_scale = 0.0
        self.task.xy_align_reward_scale = 0.0
        self.task.z_align_reward_scale = 0.0
        self.task.descent_reward_scale = 0.0

    def _apply_baseline_single_pos(self) -> None:
        """Single-position baseline: identical to A in obs/state, but every
        reset-time pose randomizer is zeroed and the source hole is pinned
        at +10 cm X from the destination hole. Used for collecting
        deterministic tactile trajectories. Only mutates this cfg instance —
        does not affect A/B/B2.
        """
        self.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
        self.task.fixed_asset_init_orn_range_deg = 0.0
        self.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
        self.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
        self.task.source_hole_fixed_offset = [0.10, 0.0, 0.0]
