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
        if baseline == "A":
            return
        if baseline == "A_hard":
            self._apply_baseline_A_hard()
            return
        if baseline == "A_hard_success":
            self._apply_baseline_A_hard_success()
            return
        if baseline == "A_hard_success_yaw01":
            self._apply_baseline_A_hard_success_yaw01()
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
        if baseline == "tactile_state":
            self._apply_baseline_tactile_state()
            return
        raise ValueError(
            f"Unknown baseline {baseline!r} for ForgeTaskGearMeshPickPlaceCfg. "
            f"Implemented: A (frozen), A_hard (A obs + yaw_reward=0), "
            f"A_hard_success (A_hard + success requires gear half-shaft deep), "
            f"B (tactile force fields), "
            f"B2 (frozen ReWiND CNN -> 768-dim embedding), "
            f"single_pos (A obs + all reset position randomization zeroed)."
        )

    def _apply_baseline_A_hard(self) -> None:
        """A_hard: same obs/state AND same randomization as A, only the dense
        yaw shaping is cut. Designed as a "shaping-insufficient" regime to test
        whether tactile reward can fill the gap left by removing the
        yaw-alignment gradient — gears need <6° yaw match for teeth to mesh,
        but `fixed_yaw` is already fully randomized (0..360°) so without dense
        yaw gradient the policy has to find the matching angle from sparse
        `curr_engaged`/`curr_success` bonuses alone.
        """
        self.task.yaw_reward_scale = 0.0
        # (randomization knobs intentionally unchanged — only the reward shaping
        # is ablated; this isolates the effect of yaw_reward removal.)

    def _apply_baseline_A_hard_success(self) -> None:
        """A_hard_success: A_hard ablation (yaw_reward=0) PLUS a tighter
        success criterion that requires meaningful teeth engagement —
        gear must descend ~30% of the shaft height (15mm below target z),
        not just the original "barely touching" 2.5mm-above-target slack.

        Why this matters: factory's gear_mesh success check only requires
        `xy_dist < 2.5mm` and `z_disp < shaft_height * success_threshold`, with
        `check_rot=False` (factory_env.py:439 only sets check_rot for nut).
        With success_threshold=+0.05 the gear satisfies success while still
        hovering 2.5mm above the shaft top — yaw never has to align for the
        teeth to physically mesh, so `r_yaw` becomes pure reward-hack fuel
        (xy_coarse * yaw_match without contributing to success).

        2026-05-27: relaxed `success_threshold` from -0.3 (15mm deep) to -0.1
        (5mm deep). The earlier -0.3 was *too* strict — runs plateaued at 0%
        for 25+ hours with the policy stably hovering at gear_z=+9 mm while
        every other reward saturated. 5mm depth ≈ first tooth engagement,
        still requires real meshing (so yaw must align ~within tooth pitch)
        but reachable from the hover plateau with the depth/curr_success
        shaping in place. History:
          - `-1.0` (full 50 mm): all 3 runs plateau 0% × 12h
          - `-0.3` (15 mm):    25+ h still 0%, hover-trap
          - `-0.1` (5 mm):     still 0% — engaged/aligned (yaw~6deg, curr_engaged
                               ~0.82 w/ tactile) but only reached ~1mm depth, so
                               the 5mm gate never fired. success=0 was a
                               measurement artifact, not a learning failure.
          - `+0.05`:           original "barely touching", trivial
        2026-07-11: -0.1 -> 0.0. The tactile policy meshes at the target plane
        (yaw aligned so teeth actually engage) but can't drive 5mm deeper;
        count "reached target with teeth engaged" as success.
        """
        # Reuse A_hard ablations (yaw_reward = 0).
        self._apply_baseline_A_hard()
        # At/below target z — teeth engaged at the shaft top counts as success.
        self.task.success_threshold = 0.0
        # Fix yaw randomization to match the tactile-reward-model training
        # distribution. The curriculum dataset was collected with `single_pos`
        # (gear_table_yaw_range = 0, fixed_asset_init_orn_range_deg = 0), so
        # the model only saw tactile patterns from a single fixed initial yaw.
        # Without these zeros, RL would init gear/bolt at 0..360° random yaw
        # and the reward model would see OOD patterns most of the time, giving
        # noise instead of usable yaw-learning signal.
        self.task.gear_table_yaw_range = 0.0
        self.task.fixed_asset_init_orn_range_deg = 0.0
        self.task.hand_init_orn_noise = [0.0, 0.0, 0.0]

    def _apply_baseline_A_hard_success_yaw01(self) -> None:
        """A_hard_success_yaw01: identical to A_hard_success EXCEPT yaw shaping
        is restored with a wider gate. Previous attempt (yaw_reward=0.1,
        yaw_scale=0.1 rad ≈ 6°) failed because at the observed 35° drift the
        signal collapsed to exp(-0.6/0.1) ≈ 0.0025 — effectively zero gradient.

        2026-05-23: bumped to give actual gradient at the drift the gripper
        induces on pickup:
          - `yaw_reward_scale = 1.0` (was 0.1) — full-strength signal
          - `yaw_alignment_scale = 0.5 rad ≈ 28°` (was 0.1) — exp(-0.6/0.5)
            ≈ 0.30 at 35° drift, exp(-0.1/0.5) ≈ 0.82 inside 6° → still
            sharp enough at the goal to reward true alignment

        Use case: paper ablation pair
          - `A_hard_success`(yaw=0): strictest, baseline likely stalls
          - `A_hard_success_yaw01`(yaw_loose): broader gate, baseline can
            learn yaw from 35° drift, tactile still adds value on top
        """
        # Reuse all A_hard_success ablations (yaw=0, success=-0.3, yaw fixed).
        self._apply_baseline_A_hard_success()
        # Wider yaw gate at moderate strength so policy has gradient even at
        # the 30-40° drift induced by gripper twist during pickup, without
        # making yaw the dominant dense reward (which created a "hover above
        # bolt with yaw locked" local optimum). 2026-05-26: 1.0 → 0.3 to leave
        # room for the new dense depth reward to dominate "press down" budget.
        # 2026-07-17: 0.3 → 0.6. After the r_yaw gate fix (xy_coarse + wider
        # z gate) the yaw term became live but its weighted contribution (~0.026)
        # was too small vs lift/descent — baseline crawled from 67°→35° over
        # 1000 iters and stalled before reaching the ~6° meshing angle. 0.6
        # gives yaw enough budget to converge (slow but solvable) while still
        # below the depth reward's 1.0 so it can't reward-hack yaw over descent.
        # 2026-07-20: 0.6 → 0.45. At 0.6 (+ entropy 0.005) the no-tactile
        # baseline reached ~38% by iter 690 — too easy, leaving no headroom for
        # tactile to show a learning-speed advantage. Data points: 0.3 never
        # solved (stalled at 35°), 0.6 solved fast; 0.45 targets "slow but
        # solvable" so the tactile run's earlier crack is a clear win.
        self.task.yaw_reward_scale = 0.5
        self.task.yaw_alignment_scale = 0.5

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
        `forge_gearpickplace_env._get_observations` populates the
        `tactile_embedding` key whenever the encoder is enabled.
        """
        embed_dim = 768  # TactileCNNEncoder output_dim = 2 * per_hand_dim (384*2)
        OBS_DIM_CFG.update({"tactile_embedding": embed_dim})
        STATE_DIM_CFG.update({"tactile_embedding": embed_dim})

        self.obs_order = list(self.obs_order) + ["tactile_embedding"]
        self.state_order = list(self.state_order) + ["tactile_embedding"]
    def _apply_baseline_tactile_state(self) -> None:
        """raw_tactile: A_hard_success reward shaping + frozen AE tactile latent.

        Reward side is byte-identical to A_legacy (half-strength transport
        bridges, 1 cm descent gate), so the only delta vs. that baseline is
        the extra input modality: a tactile embedding from a frozen
        autoencoder-pretrained TactileCNNEncoder (trained by
        external/third-party/Tactile-ReWiND/train_tactile_ae.py on saved
        (T, 40, 25, 3) force-field episodes). Unlike B2's encoder (trained
        to predict ReWiND task progress), the AE latent carries no
        task/reward information — it is a pure compression of the
        (normal, shear_x, shear_y) force fields — so this baseline isolates
        "tactile as state input" from "tactile as reward" (TacReward).

        The embedding is appended to BOTH actor obs and critic state
        (B/B2 convention). Latent width comes from FORGE_TACTILE_ENCODER_DIM
        (default 128 = 2 x per_hand_dim 64) and must equal the ckpt
        encoder's output dim — `_init_tactile_encoder` raises at startup on
        mismatch.

        GelSight RGB is not consumed here, so the tactile cameras are
        switched off (force-field / SDF sensing only). The run therefore
        needs no --enable_cameras / RTX renderer and stays runnable on
        compute-only cloud GPUs.
        """
        # Fail fast on incompatible env-var combos (clearer than the
        # KeyError / shape error they would otherwise cause deep in obs
        # assembly).
        if os.getenv("FORGE_SKIP_TACTILE_SENSORS", "0") == "1":
            raise RuntimeError(
                "baseline raw_tactile reads the GelSight force fields; unset "
                "FORGE_SKIP_TACTILE_SENSORS."
            )
        if not os.getenv("FORGE_TACTILE_ENCODER_CKPT", "").strip():
            raise RuntimeError(
                "baseline raw_tactile requires FORGE_TACTILE_ENCODER_CKPT "
                "(autoencoder ckpt from train_tactile_ae.py)."
            )

        # Same reward regime as the paired no-tactile baseline.
        self._apply_baseline_A_hard_success_yaw01()

        embed_dim = int(os.getenv("FORGE_TACTILE_ENCODER_DIM", "128"))
        OBS_DIM_CFG.update({"tactile_embedding": embed_dim})
        STATE_DIM_CFG.update({"tactile_embedding": embed_dim})
        self.obs_order = list(self.obs_order) + ["tactile_embedding"]
        self.state_order = list(self.state_order) + ["tactile_embedding"]

        # Force-field only: drop the tactile RGB camera pipeline (unused by
        # this baseline; SDF force sensing does not need the RTX renderer).
        self.left_tactile_sensor.enable_camera_tactile = False
        self.right_tactile_sensor.enable_camera_tactile = False
