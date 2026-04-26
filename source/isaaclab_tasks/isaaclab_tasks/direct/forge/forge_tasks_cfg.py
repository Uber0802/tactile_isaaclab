# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import FactoryTask, GearMesh, NutThread, PegInsert


@configclass
class ForgeTask(FactoryTask):
    action_penalty_ee_scale: float = 0.0
    action_penalty_asset_scale: float = 0.001
    action_grad_penalty_scale: float = 0.1
    contact_penalty_scale: float = 0.05
    delay_until_ratio: float = 0.25
    contact_penalty_threshold_range = [5.0, 10.0]


@configclass
class ForgePegInsert(PegInsert, ForgeTask):
    contact_penalty_scale: float = 0.2
    # The GelSight fingertip TCP sits differently from the stock Franka fingerpad TCP.
    # Narrow the randomized start pose to the reachable set of the new tip geometry.
    hand_init_pos: list = [0.0, 0.0, 0.08]
    hand_init_pos_noise: list = [0.1, 0.1, 0.02]
    hand_init_orn_noise: list = [0.0, 0.0, 0.4]
    held_asset_in_hand_pos_offset: list = [0.0, 0.0015, 0.0]
    # Hole sits on the table — only x/y position should be randomized, not z.
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.0]


@configclass
class ForgeGearMesh(GearMesh, ForgeTask):
    contact_penalty_scale: float = 0.05


@configclass
class ForgeNutThread(NutThread, ForgeTask):
    contact_penalty_scale: float = 0.05
