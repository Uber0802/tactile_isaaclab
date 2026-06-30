# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeCtrlCfg, ForgeTaskNutThreadCfg

from .forge_nutpickplace_tasks_cfg import ForgeNutThreadPickPlace


# The nut lives on the table at reset, so the actor (not just the critic) needs
# to see it. Register the absolute / cross-frame entries that show up in
# `obs_order` below — finger, nut, bolt absolute positions plus finger-relative-
# to-nut, plus the held-asset orientation.
OBS_DIM_CFG.update(
    {
        "held_pos": 3,
        "held_pos_rel_fixed": 3,
        "held_quat": 4,
        "fixed_pos": 3,
        "fingertip_pos_rel_held": 3,
    }
)


@configclass
class NutPickPlaceCtrlCfg(ForgeCtrlCfg):
    # Action frame stays centered on the bolt tip, but the policy must also reach
    # the nut sitting on the table (~10 cm away), so widen the absolute target bounds.
    pos_action_bounds = [0.15, 0.2, 0.2]


@configclass
class ForgeTaskNutThreadPickPlaceCfg(ForgeTaskNutThreadCfg):
    task_name = "nut_thread"
    task = ForgeNutThreadPickPlace()
    ctrl: NutPickPlaceCtrlCfg = NutPickPlaceCtrlCfg()
    # Pick-place is a longer-horizon task (grasp → lift → move → thread).
    episode_length_s = 30.0
    # 6 pose dims + success prediction (index 6) + gripper (index 7).
    action_space: int = 8

    # Actor obs — BASELINE A (no tactile/force sensing). Expose absolute finger /
    # nut / bolt positions plus the finger-relative-to-nut vector so the policy
    # has every geometric cue it needs for grasp + transport + thread without
    # having to reconstruct them from a single bolt-relative frame.
    obs_order: list = [
        "fingertip_pos",
        "fingertip_pos_rel_held",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        # "ft_force",        # commented for no-tactile baseline
        # "force_threshold", # commented for no-tactile baseline
        "held_pos",
        "held_quat",
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

    def __post_init__(self):
        super().__post_init__()
        # Parent ForgeTaskNutThreadCfg.__post_init__ hard-codes hand_init_pos /
        # hand_init_pos_noise for the original task where the gripper starts just
        # above the bolt. Pick-place needs the gripper to start above the nut on
        # the table (~10 cm away), so re-apply our own values after super().
        self.task.hand_init_pos = [0.0, 0.0, 0.10]
        self.task.hand_init_pos_noise = [0.02, 0.02, 0.01]
        # Re-enable gravity on the held nut: in the base NutThread task it floats
        # in the gripper, but here it has to physically rest on the table at reset
        # so closing the gripper friction-grips it instead of pushing the
        # weightless nut out of the fingers.
        self.task.held_asset.spawn.rigid_props.disable_gravity = False

    # ------------------------------------------------------------------
    # Baseline switch (called by train.py after env_cfg is loaded).
    # Baseline A = original frozen reference (no-op). Future baselines
    # (B, C, D, E, ...) are dispatched here without touching A's code path.
    # ------------------------------------------------------------------
    def apply_baseline(self, baseline: str) -> None:
        if baseline == "baseline":
            self._apply_baseline_baseline()
            return
        if baseline == "single_pos":
            self._apply_baseline_single_pos()
            return
        raise ValueError(
            f"Unknown baseline {baseline!r} for ForgeTaskNutThreadPickPlaceCfg. "
            f"Implemented: baseline (yaw_reward=0 + wider pose noise + nut threaded "
            f"~3.5 pitches deep for success, bidirectional rotation), "
            f"single_pos (baseline obs + all reset randomization zeroed)."
        )

    def _apply_baseline_baseline(self) -> None:
        """baseline (was A_hard_success): cut the dense yaw shaping, widen the
        initial pose randomization, and tighten the success criterion so the nut
        must be threaded several pitches deep — the regime where tactile reward
        has measurable room to help over the no-tactile policy.

          - `yaw_reward_scale = 0.0`: remove `r_yaw = xy_coarse * yaw_progress`;
            the policy must learn the wrist-yaw<0 success condition from the
            sparse `curr_success` bonus alone.
          - Wider hand init (2cm→8cm xy, 1cm→3cm z) and bolt init (5cm→10cm xy,
            +3cm z) so transport can't lean on hard-coded heuristics.
          - `success_threshold = -3.5`: factory's z check is
            `z_disp < thread_pitch * success_threshold`; with thread_pitch=2mm
            the nut must reach target_z - 7mm (~3.5 wrist revolutions past
            engagement) — the phase where tactile patterns (engaged vs slipping
            vs jamming) differ most. The dense r_descent/r_z_descend peaks at
            z_disp = -6mm (forge_nutpickplace_env._get_rewards).
          - `unidirectional_rot = False`: drop the hardware constraint that
            silently clamps positive delta_yaw, so the policy must learn which
            direction to rotate rather than drifting negative for free.
        """
        # Yaw shaping ablation + wider initial randomization.
        self.task.yaw_reward_scale = 0.0
        self.task.hand_init_pos_noise = [0.08, 0.08, 0.03]
        self.task.fixed_asset_init_pos_noise = [0.10, 0.10, 0.03]
        # Tighter success: nut threaded ~3.5 pitches (~7mm) deep.
        self.task.success_threshold = -3.5
        # Must learn rotation direction (no unidirectional clamp).
        self.task.unidirectional_rot = False

    def _apply_baseline_single_pos(self) -> None:
        """Single-position baseline: identical to A in obs/state, but zero out
        every reset-time pose randomizer so bolt / nut / hand spawn at exactly
        the same pose every episode. Used for collecting deterministic tactile
        trajectories. Only mutates this cfg instance — does not affect A/B/B2.
        """
        self.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
        self.task.fixed_asset_init_orn_range_deg = 0.0
        self.task.nut_table_pos_noise = [0.0, 0.0, 0.0]
        self.task.nut_table_yaw_range = 0.0
        self.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
        self.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
