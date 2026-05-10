# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG
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
