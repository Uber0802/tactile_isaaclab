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
        if baseline == "A":
            return
        if baseline == "B":
            self._apply_baseline_B()
            return
        raise ValueError(
            f"Unknown baseline {baseline!r} for ForgeTaskNutThreadPickPlaceCfg. "
            f"Implemented: A (frozen), B (tactile force fields)."
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
