# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeCtrlCfg, ForgeTaskGearMeshCfg

from .forge_gearpickplace_tasks_cfg import ForgeGearMeshPickPlace


# The gear lives on the table at reset, so the actor (not just the critic) needs
# to see it. Register held-asset obs dims so they can appear in `obs_order`.
OBS_DIM_CFG.update({"held_pos_rel_fixed": 3, "held_quat": 4})

# Success-target signals — explicit "where the gear must go" info so the actor
# doesn't have to derive it from peg-tip-relative obs. `target_pos` is the
# absolute target (medium-peg base position with base yaw applied);
# `fingertip_to_target` and `gear_to_target` are the direct error vectors the
# transport / descent rewards minimise.
OBS_DIM_CFG.update({"target_pos": 3, "fingertip_to_target": 3, "gear_to_target": 3})

# Yaw mismatch (gear_yaw − fixed_yaw) as (sin, cos) pair — continuous
# representation avoids ±π wrap discontinuity. Lets the actor explicitly
# observe "how far off in yaw" instead of having to extract it from raw
# `held_quat` / `fixed_quat`.
OBS_DIM_CFG.update({"yaw_diff_to_fixed": 2})
STATE_DIM_CFG.update({"yaw_diff_to_fixed": 2})


@configclass
class GearPickPlaceCtrlCfg(ForgeCtrlCfg):
    # Action frame stays centered on the gear post, but the policy must also reach the
    # gear sitting on the table (~7 cm away), so widen the absolute target bounds.
    pos_action_bounds = [0.15, 0.2, 0.2]


@configclass
class ForgeTaskGearMeshPickPlaceCfg(ForgeTaskGearMeshCfg):
    task_name = "gear_mesh"
    task = ForgeGearMeshPickPlace()
    ctrl: GearPickPlaceCtrlCfg = GearPickPlaceCtrlCfg()
    # Pick-place is a longer-horizon task (grasp → lift → move → mesh).
    episode_length_s = 20.0
    # 6 pose dims + success prediction (index 6) + gripper (index 7).
    action_space: int = 8

    # Actor obs — BASELINE A (no tactile/force sensing). Wrist F/T (`ft_force`)
    # and the contact-penalty threshold (`force_threshold`) are commented out so
    # we have a clean no-touch baseline to compare against the tactile baseline
    # (which adds the 40×35 tactile force field saved via FORGE_SAVE_TACTILE_FORCE_FIELD).
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        # "ft_force",        # commented for no-tactile baseline
        # "force_threshold", # commented for no-tactile baseline
        "held_pos_rel_fixed",
        "held_quat",
        # Direct success-target signals.
        "target_pos",
        "fingertip_to_target",
        "gear_to_target",
        # Explicit yaw mismatch (sin, cos of gear_yaw − fixed_yaw).
        "yaw_diff_to_fixed",
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
        "yaw_diff_to_fixed",
        "task_prop_gains",
        "ema_factor",
        # "ft_force",        # commented for no-tactile baseline
        "pos_threshold",
        "rot_threshold",
        # "force_threshold", # commented for no-tactile baseline
    ]

    def __post_init__(self):
        super().__post_init__()
        # Re-enable gravity on the held gear: in the base GearMesh task it floats in
        # the gripper, but here it has to physically rest on the table at reset so
        # that closing the gripper friction-grips it instead of pushing the
        # weightless gear out of the fingers.
        self.task.held_asset.spawn.rigid_props.disable_gravity = False

        # Curriculum-rollout compat: old gear ckpts (May 10 baselineA series)
        # were trained with obs_dim=37, before `yaw_diff_to_fixed` (2-dim
        # sin/cos) was added. Setting FORGE_DISABLE_YAW_DIFF_OBS=1 strips the
        # new feature from obs/state so the old ckpts can be re-loaded for
        # rollout. New training runs (with yaw_diff in obs) leave the env var
        # unset and keep the 39-dim layout.
        if os.environ.get("FORGE_DISABLE_YAW_DIFF_OBS", "0") == "1":
            self.obs_order = [o for o in self.obs_order if o != "yaw_diff_to_fixed"]
            self.state_order = [s for s in self.state_order if s != "yaw_diff_to_fixed"]

        # Gear-only PhysX bump: support 4096-env speed runs. Default factory
        # cfg uses 2**29 (~0.5 GB) collision stack which PhysX overflows at
        # ~1.5 GB needed for 4096 gear envs (flanking gears triple the contact
        # pairs vs peg / nut). 2**31 (~2.1 GB) gives headroom. Scoped to gear
        # so peg / nut keep their lighter defaults.
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_max_rigid_patch_count = 2**24
        self.sim.physx.gpu_collision_stack_size = 2**31

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
            f"Unknown baseline {baseline!r} for ForgeTaskGearMeshPickPlaceCfg. "
            f"Implemented: baseline (yaw_reward=0 + gear meshed ~5mm deep for "
            f"success + fixed yaw randomization), "
            f"single_pos (baseline obs + all reset randomization zeroed)."
        )

    def _apply_baseline_baseline(self) -> None:
        """baseline (was A_hard_success): cut the dense yaw shaping and tighten
        the success criterion so the gear must descend far enough for real teeth
        engagement — the regime where tactile reward has room to help.

          - `yaw_reward_scale = 0.0`: remove the dense yaw-alignment gradient;
            the policy must find the <6° meshing angle from sparse
            curr_engaged/curr_success bonuses alone.
          - `success_threshold = -0.1`: gear must descend ~5mm below target z
            (first-tooth mesh), not just the original "barely touching" 2.5mm
            slack — so yaw genuinely has to align for the teeth to mesh.
          - Fix yaw randomization (gear_table_yaw_range=0,
            fixed_asset_init_orn_range_deg=0, hand_init_orn_noise=0) to match
            the tactile-reward-model training distribution (collected with
            single_pos at a single fixed yaw); otherwise the model sees OOD
            patterns and gives noise instead of usable yaw signal.
        """
        # Yaw shaping ablation.
        self.task.yaw_reward_scale = 0.0
        # 5 mm below target z (~10% of shaft) — first-tooth mesh.
        self.task.success_threshold = -0.1
        # Fixed yaw to stay in the tactile-reward-model training distribution.
        self.task.gear_table_yaw_range = 0.0
        self.task.fixed_asset_init_orn_range_deg = 0.0
        self.task.hand_init_orn_noise = [0.0, 0.0, 0.0]

    def _apply_baseline_single_pos(self) -> None:
        """Single-position baseline: identical to A in obs/state, but zero out
        every reset-time pose randomizer so base / gear / hand spawn at exactly
        the same pose every episode. Used for collecting deterministic tactile
        trajectories. Only mutates this cfg instance — does not affect A/B/B2.
        """
        self.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
        self.task.fixed_asset_init_orn_range_deg = 0.0
        self.task.gear_table_pos_noise = [0.0, 0.0, 0.0]
        self.task.gear_table_yaw_range = 0.0
        self.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
        self.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
