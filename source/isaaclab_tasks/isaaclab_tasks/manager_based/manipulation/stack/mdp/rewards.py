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


def rewind_tactile_reward(
    env: ManagerBasedRLEnv,
    stack_object_cfg: SceneEntityCfg = SceneEntityCfg("stack_object"),
    target_cube_cfg: SceneEntityCfg = SceneEntityCfg("target_cube"),
    lift_rate_target: float = 0.8,
) -> torch.Tensor:
    """Query the environment's online ReWiND tactile reward, fading it out as the object learns to lift.

    Annealing is driven by ``env.max_lift_rate`` -- the monotonic best-so-far (running max) of the
    fraction of envs holding the object above the grasp height (``StackTactileEnv._z_grasp_threshold``).
    The signal is bounded in [0, 1] and robust to a single lucky env (each contributes 1/num_envs), so
    the tactile bonus fades from full strength (no env lifting) to 0 once the *population* reliably
    grasps the object upward (``max_lift_rate >= lift_rate_target``).
    """

    if not hasattr(env, "compute_tactile_reward"):
        return torch.zeros(env.num_envs, device=env.device)

    raw_reward = env.compute_tactile_reward()

    # Best-so-far lift rate achieved by the population (defaults to 0.0 if not tracked).
    max_lift_rate = getattr(env, "max_lift_rate", 0.0)

    # Fade: 1.0 while no env lifts, decaying to 0.0 as the lift rate reaches lift_rate_target.
    fade_multiplier = 1.0 - min(max(max_lift_rate / lift_rate_target, 0.0), 1.0)

    return raw_reward * fade_multiplier


def ee_to_stack_object_circumference_distance_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    stack_object_cfg: SceneEntityCfg = SceneEntityCfg("stack_object"),
    max_distance: float = 0.6,
    max_reward: float = 1.0,
    object_radius: float = 0.036,
) -> torch.Tensor:
    """Encourage the end-effector to move close to the stack object's surface."""

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    stack_object = env.scene[stack_object_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    stack_object_pos = _get_pos(stack_object)
    xy_center_distance = torch.linalg.vector_norm(ee_pos[:, :2] - stack_object_pos[:, :2], dim=1)
    dz = ee_pos[:, 2] - stack_object_pos[:, 2]
    distance = torch.sqrt((xy_center_distance - object_radius) ** 2 + dz ** 2)
    shaped_reward = torch.clamp(1.0 - (distance / max_distance), min=0.0, max=1.0) * max_reward

    # _maybe_visualize_ee_pos(env, ee_pos + torch.tensor([object_radius * 2, 0, 0], device=ee_pos.device))
    # _maybe_visualize_stack_target(env, stack_object_pos + torch.tensor([object_radius, 0, 0], device=ee_pos.device), 0.5, marker_name="cube_1", color=(0.0, 0.35, 1.0))
    # _maybe_visualize_stack_target(env, stack_object_pos + torch.tensor([-object_radius, 0, 0], device=ee_pos.device), 0.5, marker_name="cube_2", color=(0.0, 0.35, 1.0))



    return shaped_reward

def ee_to_stack_object_distance_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    stack_object_cfg: SceneEntityCfg = SceneEntityCfg("stack_object"),
    max_distance: float = 0.6,
    max_reward: float = 1.0,
) -> torch.Tensor:
    """Encourage the end-effector to move close to the stack object."""

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    stack_object = env.scene[stack_object_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    stack_object_pos = _get_pos(stack_object)
    distance = torch.linalg.vector_norm(ee_pos - stack_object_pos, dim=1)
    shaped_reward = torch.clamp(1.0 - (distance / max_distance), min=0.0, max=1.0) * max_reward

    # _maybe_visualize_ee_pos(env, ee_pos)
                
    return shaped_reward

def stack_object_z_reward_exp(
    env: ManagerBasedRLEnv,
    stack_object_cfg: SceneEntityCfg = SceneEntityCfg("stack_object"),
    max_z_distance: float = 0.0234 * 3,
    min_z: float = 0.0,
) -> torch.Tensor:
    """Apply a base-1.1 logarithmic reward directly on cube-1's Z height."""

    stack_object = env.scene[stack_object_cfg.name]
    stack_object_pos = _get_pos(stack_object)
    cube_height = torch.clamp(stack_object_pos[:, 2], min=min_z, max=max_z_distance)
    exceed = stack_object_pos[:, 2] - (max_z_distance + 0.01)
    exceed = torch.clamp(exceed, min=0.0)

    reward = (torch.log1p(cube_height) / math.log(1.1)) * 2 
    reward -= exceed 
    reward = torch.clamp(reward, min=0.0)

    # _maybe_visualize_stack_target(env, stack_object_pos, 0.5, marker_name="cube_1", color=(0.0, 0.35, 1.0))

    return reward

def stack_object_precision_xy_reward(
    env: ManagerBasedRLEnv,
    stack_object_cfg: SceneEntityCfg = SceneEntityCfg("stack_object"),
    target_cube_cfg: SceneEntityCfg = SceneEntityCfg("target_cube"),
    stack_height_offset: float = 0.065,
    height_tolerance: float = 0.025,
    max_xy_distance: float = 25.0,
    max_reward: float = 0.3,
) -> torch.Tensor:
    """Provide an extra XY precision reward when stack object reaches the target stacking height with linear XY decay."""

    stack_object = env.scene[stack_object_cfg.name]
    target_cube = env.scene[target_cube_cfg.name]


    target_pos = _get_pos(target_cube).clone()
    target_pos[:, 2] += stack_height_offset
    delta = _get_pos(stack_object) - target_pos

    # _maybe_visualize_stack_target(env, _get_pos(stack_object), 0.5, marker_name="cube_1", color=(0.0, 0.35, 1.0))

    z_aligned = torch.abs(delta[:, 2]) < height_tolerance
    xy_distance = torch.linalg.vector_norm(delta[:, :2], dim=1)
    xy_reward = torch.clamp(1.0 - (xy_distance / max_xy_distance), min=0.0, max=1.0) * max_reward


    # return torch.where(z_aligned, xy_reward, torch.zeros_like(xy_reward))
    return xy_reward


def stack_object_precision_xy_reward_exp(
    env: ManagerBasedRLEnv,
    stack_object_cfg: SceneEntityCfg = SceneEntityCfg("stack_object"),
    target_cube_cfg: SceneEntityCfg = SceneEntityCfg("target_cube"),
    stack_height_offset: float = 0.065,
    height_tolerance: float = 0.01,
    distance_offset: float = 0.1,
    decay_rate: float = 10.0,
) -> torch.Tensor:
    """Provide an extra XY precision reward with exponential decay once stack object reaches the stacking height."""

    stack_object = env.scene[stack_object_cfg.name]
    target_cube = env.scene[target_cube_cfg.name]

    target_pos = _get_pos(target_cube).clone()
    target_pos[:, 2] += stack_height_offset
    delta = _get_pos(stack_object) - target_pos

    z_aligned = torch.abs(delta[:, 2]) < height_tolerance
    xy_distance = torch.linalg.vector_norm(delta[:, :2], dim=1)
    shaped_reward = torch.exp(-decay_rate * (xy_distance - distance_offset)) / 2.0 - 0.1
    xy_reward = torch.clamp(shaped_reward, min=0.0)

    return torch.where(z_aligned, xy_reward, torch.zeros_like(xy_reward))


def target_cube_original_xy_penalty(
    env: ManagerBasedRLEnv,
    stack_object_cfg: SceneEntityCfg = SceneEntityCfg("stack_object"),
    target_cube_cfg: SceneEntityCfg = SceneEntityCfg("target_cube"),
    penalty_scale: float = 5.0,
    stack_height_offset: float = 0.065,
    height_tolerance: float = 0.025,
) -> torch.Tensor:
    """Penalize XY displacement of the target cube from its default position."""

    stack_object = env.scene[stack_object_cfg.name]
    target_cube: RigidObject = env.scene[target_cube_cfg.name]

    target_pos = target_cube.data.root_pos_w.clone()
    target_pos[:, 2] += stack_height_offset
    z_aligned = torch.abs((_get_pos(stack_object) - target_pos)[:, 2]) < height_tolerance

    current_xy = target_cube.data.root_pos_w[:, :2]
    default_xy = target_cube.data.default_root_state[:, :2] + env.scene.env_origins[:, :2]
    displacement = torch.linalg.vector_norm(current_xy - default_xy, dim=1)

    penalty = displacement * penalty_scale
    return torch.where(z_aligned, penalty, torch.zeros_like(penalty))


def _get_pos(entity) -> torch.Tensor:
    """Helper to get the position of a RigidObject or a FrameTransformer."""
    if isinstance(entity, FrameTransformer):
        return entity.data.target_pos_w[:, 0, :]
    return entity.data.root_pos_w
