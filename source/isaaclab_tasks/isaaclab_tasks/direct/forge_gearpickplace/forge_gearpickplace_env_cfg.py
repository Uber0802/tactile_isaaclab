# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
        # Re-enable gravity on the held gear: in the base GearMesh task it floats in
        # the gripper, but here it has to physically rest on the table at reset so
        # that closing the gripper friction-grips it instead of pushing the
        # weightless gear out of the fingers.
        self.task.held_asset.spawn.rigid_props.disable_gravity = False

    # ------------------------------------------------------------------
    # Baseline switch (called by train.py after env_cfg is loaded).
    # Baseline A = original frozen reference (no-op). Future baselines
    # (B, C, D, E, ...) are dispatched here without touching A's code path.
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
        raise ValueError(
            f"Unknown baseline {baseline!r} for ForgeTaskGearMeshPickPlaceCfg. "
            f"Implemented: A (frozen), B (tactile force fields), "
            f"B2 (frozen ReWiND CNN -> 768-dim embedding)."
        )

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
