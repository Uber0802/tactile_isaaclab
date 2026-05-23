# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_pos(entity) -> torch.Tensor:
    """Helper to get the position of a RigidObject or a FrameTransformer."""
    if isinstance(entity, FrameTransformer):
        return entity.data.target_pos_w[:, 0, :]
    return entity.data.root_pos_w


def stack_success(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stack_object_cfg: SceneEntityCfg = SceneEntityCfg("stack_object"),
    target_cube_cfg: SceneEntityCfg = SceneEntityCfg("target_cube"),
    xy_threshold: float = 0.04,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    min_height: float = 0,
    atol: float = 0.0001,
    rtol: float = 0.0001,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    stack_object = env.scene[stack_object_cfg.name]
    target_cube = env.scene[target_cube_cfg.name]

    pos_diff_c12 = _get_pos(stack_object) - _get_pos(target_cube)

    # Compute position difference in x-y plane
    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)

    # Compute height difference
    h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)

    # Check cube positions
    stacked = xy_dist_c12 < xy_threshold
    stacked = torch.logical_and(h_dist_c12 - height_diff < height_threshold, stacked)
    stacked = torch.logical_and(pos_diff_c12[:, 2] > min_height, stacked)

    # Check gripper positions
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        stacked = torch.logical_and(suction_cup_is_open, stacked)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"

            stacked = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
                stacked,
            )
            stacked = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
                stacked,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return stacked

def root_horizontal_displacement_exceeded(
    env: ManagerBasedRLEnv,
    max_displacement: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("stack_object"),
) -> torch.Tensor:
    """Terminate when the asset's XY displacement from its default pose exceeds ``max_displacement``."""

    asset = env.scene[asset_cfg.name]
    env_origins = env.scene.env_origins[:, :2]
    current_xy = _get_pos(asset)[:, :2] - env_origins
    default_xy = asset.data.default_root_state[:, :2]
    displacement = torch.linalg.vector_norm(current_xy - default_xy, dim=1)
    return displacement > max_displacement
