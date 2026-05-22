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
        if baseline == "A":
            return
        if baseline == "A_legacy":
            self._apply_baseline_A_legacy()
            return
        if baseline == "A_hard_success":
            self._apply_baseline_A_hard_success()
            return
        if baseline == "B":
            self._apply_baseline_B()
            return
        if baseline == "B2":
            self._apply_baseline_B2()
            return
        if baseline == "single_pos":
            self._apply_baseline_single_pos()
            return
        raise ValueError(
            f"Unknown baseline {baseline!r} for ForgeTaskPegInsertPickPlaceCfg. "
            f"Implemented: A (frozen), "
            f"A_legacy (May 15 commit ed32dd8 shaping — no transport bridges), "
            f"A_hard_success (A obs + transport shapings off + tight success), "
            f"B (tactile force fields), "
            f"B2 (frozen ReWiND CNN -> 768-dim embedding), "
            f"single_pos (A obs + all reset position randomization zeroed, "
            f"source hole at +10cm X from destination)."
        )

    def _apply_baseline_A_legacy(self) -> None:
        """A_legacy: midway between today's default A (full shaping, easy) and
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

    def _apply_baseline_A_hard_success(self) -> None:
        """A_hard_success: same obs/state as A, but the transport-phase coarse
        gradients (xy_align, z_align) are cut and the success criterion is
        tightened so the peg must be fully bottomed in the destination hole.

        Peg has no yaw_reward (cylindrical symmetry, action space rolls/pitch
        also locked), so the nut/gear "cut yaw" trick doesn't apply. Instead,
        the harder regime here strips:

          - `xy_align_reward_scale = 0.0`: removes the dense coarse XY gradient
            (r_xy_align = r_lift * xy_coarse, ~5cm scale) that pulled the peg
            horizontally toward the destination during transport. Policy now
            relies on factory keypoint reward (sparse-ish) to find the hole.
          - `z_align_reward_scale = 0.0`: removes the dense coarse Z bridge
            (r_z_descend = r_lift * xy_coarse * z_coarse, ~5cm scale) that
            pulled the peg down once xy-aligned. Without this, the only z
            gradient is the sharp `r_descent` which fires only within ~1cm of
            target z — leaving a large dead zone in the transport phase.
          - `success_threshold = 0.0`: factory's check is
            `z_disp < hole_height * success_threshold`. Original 0.04 allowed
            the peg to be ~1mm above the hole base (peg "barely inserted")
            to count as success. With 0.0 the peg must actually touch the
            hole bottom — no slack. Combined with the cut transport gradients,
            the policy can only succeed with a clean approach + drop into the
            hole using residual signals (approach, lift, sharp r_descent,
            factory keypoint).
        """
        # Reward shaping ablation: cut both coarse transport gradients.
        self.task.xy_align_reward_scale = 0.0
        self.task.z_align_reward_scale = 0.0
        # Tighter success: peg must reach the hole bottom (no above-target slack).
        self.task.success_threshold = 0.0

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

    def _apply_baseline_B(self) -> None:
        """Baseline B: feed the (left, right) GelSight force fields (normal + shear)
        to both actor and critic. Matches the (T, 40, 25, 3) layout that gets saved
        to disk by FORGE_SAVE_TACTILE_FORCE_FIELD: 1500 dims per side
        (500 normal + 1000 shear), 3000 dims total.
        """
        rows, cols = self.left_tactile_sensor.tactile_array_size  # (20, 25)
        num_pts = rows * cols
        normal_dim = num_pts          # flat (B, num_pts)
        shear_dim = num_pts * 2       # flat (B, num_pts*2)
        tactile_dims = {
            "left_tactile_normal_force": normal_dim,
            "right_tactile_normal_force": normal_dim,
            "left_tactile_shear_force": shear_dim,
            "right_tactile_shear_force": shear_dim,
        }
        OBS_DIM_CFG.update(tactile_dims)
        STATE_DIM_CFG.update(tactile_dims)

        tactile_keys = [
            "left_tactile_normal_force",
            "right_tactile_normal_force",
            "left_tactile_shear_force",
            "right_tactile_shear_force",
        ]
        # Keep the relative order of A's existing entries; just append tactile.
        self.obs_order = list(self.obs_order) + tactile_keys
        self.state_order = list(self.state_order) + tactile_keys

    def _apply_baseline_B2(self) -> None:
        """Baseline B2: frozen ReWiND CNN encoder produces a 768-dim tactile
        embedding (per env, per step) that replaces baseline B's 3000-dim raw
        force-field obs. Symmetric: same embedding is appended to both actor
        and critic.

        The encoder is loaded inside `ForgeEnv._init_tactile_encoder` when
        `FORGE_TACTILE_ENCODER_CKPT` is set. It is frozen (eval mode, no grad).
        Only `forge_pickplace_env._get_observations` actually populates the
        `tactile_embedding` key in the obs dict, so this baseline is currently
        wired only for the peg-insert pick-place task.
        """
        embed_dim = 768  # TactileCNNEncoder output_dim = 2 * per_hand_dim (384*2)
        OBS_DIM_CFG.update({"tactile_embedding": embed_dim})
        STATE_DIM_CFG.update({"tactile_embedding": embed_dim})

        self.obs_order = list(self.obs_order) + ["tactile_embedding"]
        self.state_order = list(self.state_order) + ["tactile_embedding"]
