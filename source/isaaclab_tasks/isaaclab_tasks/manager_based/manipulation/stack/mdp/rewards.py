# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import FrameTransformer
import isaaclab.sim as sim_utils

from .observations import object_grasped


_STACK_TARGET_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/stack_target_column",
    markers={
        "column": sim_utils.CylinderCfg(
            radius=0.005,
            height=1.0,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), roughness=1.0),
        )
    },
)
"""Reusable visualization marker configuration for stack target debug columns."""

_EE_POS_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/ee_position_marker",
    markers={
        "ee": sim_utils.SphereCfg(
            radius=0.2,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.6, 0.9), roughness=0.2),
        )
    },
)
"""Marker used to visualize end-effector target positions."""

JOINT_POS_LOG_ENVS = (135, 136)
"""Environment IDs whose joint positions should be logged each step."""

JOINT_NAMES_TO_LOG = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)
"""Joint names recorded when logging joint positions."""




def _maybe_visualize_stack_target(
    env: ManagerBasedRLEnv,
    cube_pos: torch.Tensor,
    column_height: float,
    marker_name: str = "stack_target",
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
):
    """Draw a thin column anchored at the cube XY to visualize the ideal stacking line."""

    if column_height <= 0.0 or not hasattr(env, "sim") or not env.sim.has_gui():
        return

    marker_attr = f"_stack_target_marker_{marker_name}"
    marker: VisualizationMarkers | None = getattr(env, marker_attr, None)
    if marker is None:
        column_cfg = _STACK_TARGET_MARKER_CFG.markers["column"].replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=1.0)
        )
        marker_cfg = _STACK_TARGET_MARKER_CFG.replace(
            prim_path=f"{_STACK_TARGET_MARKER_CFG.prim_path}_{marker_name}",
            markers={"column": column_cfg},
        )
        marker = VisualizationMarkers(marker_cfg)
        setattr(env, marker_attr, marker)

    translations = cube_pos.clone()
    translations[:, 2] += 0.5 * column_height

    scales = torch.ones((env.num_envs, 3), device=env.device)
    scales[:, 2] = column_height

    marker.visualize(translations=translations, scales=scales)


def _maybe_visualize_ee_pos(env: ManagerBasedRLEnv, ee_positions: torch.Tensor):
    """Draw a small sphere at each environment's end-effector position for debugging."""

    if not hasattr(env, "sim") or not env.sim.has_gui():
        return

    marker: VisualizationMarkers | None = getattr(env, "_ee_position_marker", None)
    if marker is None:
        marker = VisualizationMarkers(_EE_POS_MARKER_CFG)
        setattr(env, "_ee_position_marker", marker)

    scales = torch.ones((env.num_envs, 3), device=env.device) * 0.04
    marker.visualize(translations=ee_positions, scales=scales)


def consecutive_object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    streak_length: int = 5,
) -> torch.Tensor:
    """Reward once when the specified object has been grasped for N consecutive steps."""

    grasped = object_grasped(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg, object_cfg=object_cfg).bool()
    attr_name = f"_grasp_streak_{object_cfg.name}"
    if not hasattr(env, attr_name):
        setattr(env, attr_name, torch.zeros(env.num_envs, device=env.device))

    streak_buf = getattr(env, attr_name)
    streak_buf = torch.where(env.reset_buf.bool(), torch.zeros_like(streak_buf), streak_buf)
    updated_streak = torch.where(grasped, torch.clamp(streak_buf + 1, max=streak_length), torch.zeros_like(streak_buf))

    reward = torch.logical_and(grasped, torch.logical_and(updated_streak == streak_length, streak_buf < streak_length))
    setattr(env, attr_name, updated_streak)
    return reward.float()


def cube_xy_moved(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    displacement_threshold: float = 0.01,
) -> torch.Tensor:
    """Returns 1 when the specified cube has moved beyond the XY displacement threshold."""

    cube: RigidObject = env.scene[cube_cfg.name]
    current_xy = cube.data.root_pos_w[:, :2]
    default_xy = cube.data.default_root_state[:, :2]
    displacement = torch.linalg.vector_norm(current_xy - default_xy, dim=1)
    return (displacement > displacement_threshold).float()


def rewind_tactile_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Query the environment's online ReWiND tactile reward if available."""

    if hasattr(env, "compute_rewind_tactile_reward"):
        return env.compute_rewind_tactile_reward()
    return torch.zeros(env.num_envs, device=env.device)


def grasped_cube_to_stack_target_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    stack_height_offset: float = 0.0406,
    max_distance: float = 0.15,
    max_reward: float = 0.5,
) -> torch.Tensor:
    """Reward cube-1 when grasped and positioned near the stack target above cube-2."""

    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]

    target_pos = cube_2.data.root_pos_w.clone()
    target_pos[:, 2] += stack_height_offset
    distance = torch.linalg.vector_norm(cube_1.data.root_pos_w - target_pos, dim=1)
    shaped_reward = torch.clamp(1.0 - (distance / max_distance), min=0.0, max=1.0) * max_reward

    grasped = object_grasped(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg, object_cfg=cube_1_cfg).bool()
    return torch.where(grasped, shaped_reward, torch.zeros_like(shaped_reward))


def ee_to_cube_distance_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    max_distance: float = 0.6,
    max_reward: float = 1.0,
) -> torch.Tensor:
    """Encourage the end-effector to move close to the specified cube."""

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    cube_pos = cube.data.root_pos_w
    distance = torch.linalg.vector_norm(ee_pos - cube_pos, dim=1)
    shaped_reward = torch.clamp(1.0 - (distance / max_distance), min=0.0, max=1.0) * max_reward

    _maybe_visualize_ee_pos(env, ee_pos)
                
    return shaped_reward

def ee_to_cube_distance_reward_exp(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """Encourage the end-effector to move close to the specified cube."""

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    cube_pos = cube.data.root_pos_w
    distance = torch.linalg.vector_norm(ee_pos - cube_pos, dim=1)
    shaped_reward = torch.exp(-10 * (distance - 0.1)) / 3.0

    return shaped_reward

def cube_z_reward_exp(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    max_z_distance: float = 0.0203 * 3,
) -> torch.Tensor:
    """Apply a base-1.1 logarithmic reward directly on cube-1's Z height."""

    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_height = torch.clamp(cube_1.data.root_pos_w[:, 2], min=0.0, max=max_z_distance)
    exceed = cube_1.data.root_pos_w[:, 2] - (max_z_distance + 0.01)
    exceed = torch.clamp(exceed, min=0.0)

    reward = (torch.log1p(cube_height) / math.log(1.1)) * 2 
    reward -= exceed 
    reward = torch.clamp(reward, min=0.0)

    return reward

def cube_precision_xy_reward(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    stack_height_offset: float = 0.065,
    height_tolerance: float = 0.025,
    max_xy_distance: float = 25.0,
    max_reward: float = 0.3,
) -> torch.Tensor:
    """Provide an extra XY precision reward when cube-1 reaches the target stacking height with linear XY decay."""

    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]


    target_pos = cube_2.data.root_pos_w.clone()
    target_pos[:, 2] += stack_height_offset
    delta = cube_1.data.root_pos_w - target_pos

    _maybe_visualize_stack_target(env, cube_1.data.root_pos_w, 0.5, marker_name="cube_1", color=(0.0, 0.35, 1.0))

    z_aligned = torch.abs(delta[:, 2]) < height_tolerance
    xy_distance = torch.linalg.vector_norm(delta[:, :2], dim=1)
    xy_reward = torch.clamp(1.0 - (xy_distance / max_xy_distance), min=0.0, max=1.0) * max_reward


    # return torch.where(z_aligned, xy_reward, torch.zeros_like(xy_reward))
    return xy_reward


def cube_precision_xy_reward_exp(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    stack_height_offset: float = 0.065,
    height_tolerance: float = 0.01,
    distance_offset: float = 0.1,
    decay_rate: float = 10.0,
) -> torch.Tensor:
    """Provide an extra XY precision reward with exponential decay once cube-1 reaches the stacking height."""

    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]

    target_pos = cube_2.data.root_pos_w.clone()
    target_pos[:, 2] += stack_height_offset
    delta = cube_1.data.root_pos_w - target_pos

    z_aligned = torch.abs(delta[:, 2]) < height_tolerance
    xy_distance = torch.linalg.vector_norm(delta[:, :2], dim=1)
    shaped_reward = torch.exp(-decay_rate * (xy_distance - distance_offset)) / 2.0 - 0.1
    xy_reward = torch.clamp(shaped_reward, min=0.0)

    return torch.where(z_aligned, xy_reward, torch.zeros_like(xy_reward))


def cube_original_xy_penalty(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    penalty_scale: float = 5.0,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    stack_height_offset: float = 0.065,
    height_tolerance: float = 0.025,
) -> torch.Tensor:
    """Penalize XY displacement of the given cube from its default position."""

    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]

    target_pos = cube_2.data.root_pos_w.clone()
    target_pos[:, 2] += stack_height_offset
    z_aligned = torch.abs((cube_1.data.root_pos_w - target_pos)[:, 2]) < height_tolerance

    current_xy = cube.data.root_pos_w[:, :2]
    default_xy = cube.data.default_root_state[:, :2] + env.scene.env_origins[:, :2]
    displacement = torch.linalg.vector_norm(current_xy - default_xy, dim=1)

    penalty = displacement * penalty_scale
    return torch.where(z_aligned, penalty, torch.zeros_like(penalty))
