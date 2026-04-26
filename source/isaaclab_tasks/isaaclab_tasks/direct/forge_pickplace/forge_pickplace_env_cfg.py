# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeCtrlCfg, ForgeTaskPegInsertCfg

from .forge_pickplace_tasks_cfg import ForgePegInsertPickPlace


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
