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
            f"Implemented: A (frozen), B (tactile force fields), "
            f"B2 (frozen ReWiND CNN -> 768-dim embedding), "
            f"single_pos (A obs + all reset position randomization zeroed, "
            f"source hole at +10cm X from destination)."
        )

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
