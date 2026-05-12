# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import sys
import time

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

# Periodic text log so we can monitor training without wandb.
_GP_LOG_PATH = os.getenv(
    "GEARPICKPLACE_LOG",
    "/mnt/home/tactile/tactile_isaaclab/logs/gearpickplace_metrics.log",
)
_GP_LOG_INTERVAL = int(os.getenv("GEARPICKPLACE_LOG_INTERVAL", "200"))
os.makedirs(os.path.dirname(_GP_LOG_PATH), exist_ok=True)


def _gp_log(line: str):
    print(line, file=sys.stderr, flush=True)
    try:
        with open(_GP_LOG_PATH, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor
from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

from .forge_gearpickplace_env_cfg import ForgeTaskGearMeshPickPlaceCfg


class ForgeGearMeshPickPlaceEnv(ForgeEnv):
    """Pick-and-place gear-mesh variant.

    The medium gear rests freely on the table at a randomized offset from the gear base.
    The gripper starts open above the gear. The policy must close the gripper, pick up
    the gear, transport it over the medium gear post, and lower it onto the post.
    Action index 7 controls the gripper; index 6 remains the success predictor.
    """

    cfg: ForgeTaskGearMeshPickPlaceCfg
    _gp_metric_step = 0
    _gp_metric_start_t = time.perf_counter()

    def _setup_scene(self):
        """Replicate FactoryEnv + ForgeEnv scene setup (gear base + flanking gears + tactile)."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )
        table_cfg.func(
            "/World/envs/env_.*/Table", table_cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.add_flanking_gears:
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.add_flanking_gears:
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if hasattr(self.cfg, "left_tactile_sensor"):
            left_tactile_cfg = copy.deepcopy(self.cfg.left_tactile_sensor)
            left_tactile_cfg.prim_path = left_tactile_cfg.prim_path.format(ENV_REGEX_NS=self.scene.env_regex_ns)
            left_tactile_cfg.camera_cfg.prim_path = left_tactile_cfg.camera_cfg.prim_path.format(
                ENV_REGEX_NS=self.scene.env_regex_ns
            )
            left_tactile_cfg.contact_object_prim_path_expr = left_tactile_cfg.contact_object_prim_path_expr.format(
                ENV_REGEX_NS=self.scene.env_regex_ns
            )
            self._left_tactile_sensor = VisuoTactileSensor(left_tactile_cfg)
            self.scene.sensors["left_tactile_sensor"] = self._left_tactile_sensor

        if hasattr(self.cfg, "right_tactile_sensor"):
            right_tactile_cfg = copy.deepcopy(self.cfg.right_tactile_sensor)
            right_tactile_cfg.prim_path = right_tactile_cfg.prim_path.format(ENV_REGEX_NS=self.scene.env_regex_ns)
            right_tactile_cfg.camera_cfg.prim_path = right_tactile_cfg.camera_cfg.prim_path.format(
                ENV_REGEX_NS=self.scene.env_regex_ns
            )
            right_tactile_cfg.contact_object_prim_path_expr = right_tactile_cfg.contact_object_prim_path_expr.format(
                ENV_REGEX_NS=self.scene.env_regex_ns
            )
            self._right_tactile_sensor = VisuoTactileSensor(right_tactile_cfg)
            self.scene.sensors["right_tactile_sensor"] = self._right_tactile_sensor

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Force-open the gripper at every reset (default Factory behavior clamps around the asset)."""
        gripper_width = self.cfg_task.gripper_open_width
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def _apply_action(self):
        """Same control law as ForgeEnv but with a policy-controlled gripper width."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))

        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions

        rot_actions[:, 0:2] = 0.0
        rot_actions[:, 2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0
        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )
        rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )
        ctrl_target_fingertip_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self.fingertip_midpoint_pos
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped

        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_fingertip_preclipped_quat)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        desired_yaw = factory_utils.wrap_yaw(desired_yaw)
        self.delta_yaw = desired_yaw - curr_yaw
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch)
        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll

        curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch)
        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
        desired_xyz[:, 1] = curr_pitch + clipped_pitch

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
        )

        # Map action[7] from [-1, 1] to Franka finger joint width [0, gripper_open_width].
        gripper_action = (self.actions[:, 7] + 1.0) * 0.5 * self.cfg_task.gripper_open_width

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=gripper_action.unsqueeze(-1),
        )

    def _reset_idx(self, env_ids):
        """Initialize the gripper-open action so EMA smoothing doesn't snap it closed."""
        super()._reset_idx(env_ids)
        self.actions[:, 7] = self.prev_actions[:, 7] = 1.0

    def _get_observations(self):
        """Reimplement ForgeEnv._get_observations with held-asset pose added to the
        actor obs dict.

        Base ForgeEnv only exposes fingertip / force info to the actor; held_pos /
        held_quat live only in the critic state. For pick-place the gear's table
        position is essential — without it the actor can't locate the gear at all.
        We can't call super(): it calls collapse_obs_dict against our extended
        obs_order before we get a chance to inject the missing keys.
        """
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        if "left_tactile_sensor" in self.scene.sensors:
            left_normal_force, left_shear_force = self._get_tactile_force_tensors("left_tactile_sensor")
            right_normal_force, right_shear_force = self._get_tactile_force_tensors("right_tactile_sensor")
            self._save_env0_tactile_force_field()
            # Populate the same 4 tactile entries into both dicts. Whether they
            # actually feed into the actor / critic is decided by `obs_order` /
            # `state_order` (see `apply_baseline` in env_cfg). Baseline A does
            # not list any of these, so its actor vector stays unchanged.
            tactile_dict = {
                "left_tactile_normal_force": left_normal_force,
                "left_tactile_shear_force": left_shear_force,
                "right_tactile_normal_force": right_normal_force,
                "right_tactile_shear_force": right_shear_force,
            }
            # Baseline B2: frozen ReWiND CNN encoder -> 768-dim embedding.
            # Only added when the encoder is loaded (env var-gated in ForgeEnv).
            # Baseline A/B obs_order do not reference this key, so this is a
            # no-op for them; only `_apply_baseline_B2()` lists it in obs_order
            # / state_order.
            if getattr(self, "_tactile_encoder_enabled", False):
                tactile_dict["tactile_embedding"] = self._compute_tactile_embedding()
            obs_dict.update(tactile_dict)
            state_dict.update(tactile_dict)

        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        # Compute the success target (geometric-base target — gear-base centre +
        # rotated medium-peg X offset, at base z). This is the EXACT position the
        # success criterion checks gear's geometric base against. Feeding it to
        # the actor gives a direct "this is where to put the gear" signal so the
        # policy doesn't have to triangulate from peg-tip-relative obs + learn the
        # constant base_height + height z-offset.
        target_held_base_pos, _ = factory_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        held_base_pos, _ = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )

        obs_dict.update(
            {
                "fingertip_pos": self.noisy_fingertip_pos,
                "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
                "fingertip_quat": self.noisy_fingertip_quat,
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "ft_force": self.noisy_force,
                "prev_actions": prev_actions,
                "held_pos_rel_fixed": self.held_pos - noisy_fixed_pos,
                "held_quat": self.held_quat,
                # Success target signals — see comment above.
                "target_pos": target_held_base_pos,
                "fingertip_to_target": target_held_base_pos - self.noisy_fingertip_pos,
                "gear_to_target": target_held_base_pos - held_base_pos,
            }
        )

        state_dict.update(
            {
                "ema_factor": self.ema_factor,
                "ft_force": self.force_sensor_smooth[:, 0:3],
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "prev_actions": prev_actions,
            }
        )

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _pre_physics_step(self, action):
        """Faster EMA for the gripper so it can actually close, but not instant."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        raw_action = action.clone().to(self.device)
        self.actions = self.ema_factor * raw_action + (1 - self.ema_factor) * self.actions
        gripper_ema = self.cfg_task.gripper_ema
        self.actions[:, 7] = gripper_ema * raw_action[:, 7] + (1 - gripper_ema) * self.actions[:, 7]

    def _get_rewards(self):
        """Forge gear-mesh reward plus pick-place shaping (reach + lift + descent).

        No binary "is gear grasped" detection — only continuous distance / lift /
        descent shaping plus a gear-speed penalty for physics regularization.
        """
        rew_buf = super()._get_rewards()

        # (1) Reach: gripper → gear grasp point (gear top minus 1/4 gear height).
        # Use a sum of sharp + broad exponentials so the gradient survives at the
        # ~60 cm initial distance instead of being numerically zero.
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        grasp_offset_local = torch.zeros((self.num_envs, 3), device=self.device)
        grasp_offset_local[:, 2] = self.cfg_task.gear_grasp_z_offset
        _, gear_grasp_pos = torch_utils.tf_combine(
            self.held_quat, self.held_pos, identity_quat, grasp_offset_local
        )
        fingertip_to_gear = torch.norm(self.fingertip_midpoint_pos - gear_grasp_pos, p=2, dim=-1)
        r_approach = 0.5 * (
            torch.exp(-fingertip_to_gear / self.cfg_task.approach_scale)
            + torch.exp(-fingertip_to_gear / 0.3)
        )

        # (2) Object lift: gear z above its initial table z, multiplied by a
        # continuous proximity gate so the policy can't farm lift by raising the
        # gripper without bringing the gear along. (Gate is exp-distance, not a
        # binary "is grasped" check.)
        gear_z_rel_table = self.held_pos[:, 2] - self.gear_initial_z
        proximity_gate = torch.exp(-fingertip_to_gear / self.cfg_task.lift_proximity_scale)
        r_lift = proximity_gate * torch.tanh(
            torch.clamp(gear_z_rel_table, min=0.0) / self.cfg_task.lift_scale
        )

        # (3) Gear speed penalty — discourages closing the gripper so hard the gear is
        # ejected at multi-m/s speeds (physics regularizer, not a grasp signal).
        gear_speed = torch.norm(self._held_asset.data.root_lin_vel_w, dim=-1)
        gear_speed_excess = torch.clamp(gear_speed - self.cfg_task.gear_speed_threshold, min=0.0)
        r_gear_speed_penalty = -self.cfg_task.gear_speed_penalty_scale * gear_speed_excess

        # Compute geometric-base poses exactly like the success criterion does
        # (factory_env._get_curr_successes). target_held_base_pos for gear_mesh =
        # fixed_pos + R(fixed_yaw) @ (medium_gear_base_offset, 0, 0), so all the
        # base-offset / yaw geometry that breaks naive ||held_pos - fixed_pos||
        # is folded in here. r_descent therefore reduces xy_dist exactly toward
        # the success threshold (2.5 mm), no detour.
        held_base_pos, _ = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, _ = factory_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        gear_to_target_xy = torch.norm(held_base_pos[:, 0:2] - target_held_base_pos[:, 0:2], dim=-1)
        gear_above_target_z = held_base_pos[:, 2] - target_held_base_pos[:, 2]
        # Symmetric z distance — using clamp(z, min=0) lets the policy farm the
        # z-progress reward by parking the gear slightly below target without
        # actually meshing. |z_above_target| punishes deviations either way.
        z_dist = torch.abs(gear_above_target_z)

        # (4) Fine XY + Z descent reward — keeps gradient alive all the way down
        # to the (2.5 mm xy, 1 mm z) success tolerances.
        xy_aligned = torch.exp(-gear_to_target_xy / self.cfg_task.xy_alignment_scale)
        z_progress = torch.exp(-z_dist / self.cfg_task.descent_z_scale)
        r_descent = xy_aligned * z_progress

        # (5) Transport-phase coarse XY alignment. Fine `xy_aligned` saturates to
        # ~0 past 2 cm, and r_descent is killed by z_progress while gear is still
        # high. Coarse term provides horizontal pull during transport. Gated on
        # `r_lift` so it can't be farmed by sliding the gear on the table.
        xy_coarse = torch.exp(-gear_to_target_xy / self.cfg_task.xy_coarse_scale)
        r_xy_align = r_lift * xy_coarse

        # (6) Coarse Z descent — bridges the gap between r_xy_align (no z signal)
        # and r_descent (5 mm fine, both saturated during transport). Fires once
        # gear is roughly XY-aligned over the bolt at 2–10 cm height.
        z_progress_coarse = torch.exp(-z_dist / self.cfg_task.z_coarse_scale)
        r_z_descend = r_lift * xy_coarse * z_progress_coarse

        # (7) Yaw alignment reward — geometric-base offsets in xy_dist cancel only
        # when gear_yaw ≈ fixed_yaw. Gated on `xy_coarse` so the policy doesn't
        # burn yaw budget during pure transport.
        _, _, fixed_yaw = torch_utils.get_euler_xyz(self.fixed_quat)
        _, _, gear_yaw = torch_utils.get_euler_xyz(self.held_quat)
        yaw_diff = gear_yaw - fixed_yaw
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))  # wrap to [-pi, pi]
        r_yaw = xy_coarse * torch.exp(-torch.abs(yaw_diff) / self.cfg_task.yaw_alignment_scale)

        rew_buf = (
            rew_buf
            + self.cfg_task.approach_reward_scale * r_approach
            + self.cfg_task.lift_reward_scale * r_lift
            + r_gear_speed_penalty
            + self.cfg_task.xy_align_reward_scale * r_xy_align
            + self.cfg_task.z_align_reward_scale * r_z_descend
            + self.cfg_task.descent_reward_scale * r_descent
            + self.cfg_task.yaw_reward_scale * r_yaw
        )

        # ---- Diagnostics (logged only; do NOT feed back into the policy) ----
        gripper_action_cmd = self.actions[:, 7]
        gripper_joint_width = self.joint_pos[:, 7]
        gripper_open_frac = torch.clamp(gripper_joint_width / self.cfg_task.gripper_open_width, 0.0, 1.0)

        self.extras["logs_rew_approach"] = r_approach.mean()
        self.extras["logs_rew_lift"] = r_lift.mean()
        self.extras["logs_rew_gear_speed_penalty"] = r_gear_speed_penalty.mean()
        self.extras["logs_rew_descent"] = r_descent.mean()
        self.extras["logs_fingertip_to_gear"] = fingertip_to_gear.mean()
        self.extras["logs_fingertip_to_gear_min"] = fingertip_to_gear.min()
        self.extras["logs_gear_z_rel_table"] = gear_z_rel_table.mean()
        self.extras["logs_gear_z_rel_table_max"] = gear_z_rel_table.max()
        self.extras["logs_proximity_gate_mean"] = proximity_gate.mean()
        self.extras["logs_gear_speed_mean"] = gear_speed.mean()
        self.extras["logs_gear_speed_max"] = gear_speed.max()
        self.extras["logs_gear_to_target_xy"] = gear_to_target_xy.mean()
        self.extras["logs_gear_to_target_xy_min"] = gear_to_target_xy.min()
        self.extras["logs_gear_above_target_z"] = gear_above_target_z.mean()
        self.extras["logs_z_dist_mean"] = z_dist.mean()
        self.extras["logs_yaw_diff_abs"] = torch.abs(yaw_diff).mean()
        self.extras["logs_rew_yaw"] = r_yaw.mean()
        self.extras["logs_rew_xy_align"] = r_xy_align.mean()
        self.extras["logs_rew_z_descend"] = r_z_descend.mean()
        self.extras["logs_gripper_action_cmd"] = gripper_action_cmd.mean()
        self.extras["logs_gripper_width"] = gripper_joint_width.mean()
        self.extras["logs_gripper_open_frac"] = gripper_open_frac.mean()
        self.extras["logs_xy_aligned_mean"] = xy_aligned.mean()
        self.extras["logs_xy_coarse_mean"] = xy_coarse.mean()

        type(self)._gp_metric_step += 1
        if self._gp_metric_step % _GP_LOG_INTERVAL == 0:
            success_rate = self.extras.get("successes")
            if isinstance(success_rate, torch.Tensor):
                success_rate = success_rate.item()
            elapsed = time.perf_counter() - self._gp_metric_start_t
            _gp_log(
                f"[gp step={self._gp_metric_step} t={elapsed:.0f}s] "
                f"rew={rew_buf.mean().item():+.3f} "
                f"approach={r_approach.mean().item():.3f} "
                f"lift={r_lift.mean().item():.3f} "
                f"fin2gear(min/mean)={fingertip_to_gear.min().item():.3f}/{fingertip_to_gear.mean().item():.3f} "
                f"gripper(cmd/width)={gripper_action_cmd.mean().item():+.2f}/{gripper_joint_width.mean().item():.4f} "
                f"gearLift(mean/max)={gear_z_rel_table.mean().item():+.3f}/{gear_z_rel_table.max().item():+.3f} "
                f"gearV(mean/max)={gear_speed.mean().item():.2f}/{gear_speed.max().item():.2f} "
                f"gear2tgt_xy(min/mean)={gear_to_target_xy.min().item():.4f}/{gear_to_target_xy.mean().item():.4f} "
                f"gear_z_rel_tgt(min/mean)={gear_above_target_z.min().item():+.4f}/{gear_above_target_z.mean().item():+.4f} "
                f"yaw_diff_abs(mean)={torch.abs(yaw_diff).mean().item():.3f} "
                f"r_xy_align={r_xy_align.mean().item():.3f} "
                f"r_z_descend={r_z_descend.mean().item():.3f} "
                f"r_descent={r_descent.mean().item():.3f} "
                f"yaw_rew={r_yaw.mean().item():.3f} "
                f"xyCoarse={xy_coarse.mean().item():.3f} "
                f"xyFine={xy_aligned.mean().item():.3f} "
                f"success={success_rate if success_rate is not None else 'nan'}"
            )
        return rew_buf

    def randomize_initial_state(self, env_ids):
        """Place gear freely on the table at a random offset from the gear base; gripper opens above the gear."""
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1) Randomize gear-base (fixed_asset) pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        # Force gear-base local z to 0 (overrides USD default + any z noise).
        fixed_state[:, 2] = self.scene.env_origins[env_ids, 2]

        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # (1b) Park flanking gears co-located with the gear base (their pegs sit inside the base USD).
        if self.cfg_task.add_flanking_gears:
            small_state = self._small_gear_asset.data.default_root_state.clone()[env_ids]
            small_state[:, 0:7] = fixed_state[:, 0:7]
            small_state[:, 7:] = 0.0
            self._small_gear_asset.write_root_pose_to_sim(small_state[:, 0:7], env_ids=env_ids)
            self._small_gear_asset.write_root_velocity_to_sim(small_state[:, 7:], env_ids=env_ids)
            self._small_gear_asset.reset()

            large_state = self._large_gear_asset.data.default_root_state.clone()[env_ids]
            large_state[:, 0:7] = fixed_state[:, 0:7]
            large_state[:, 7:] = 0.0
            self._large_gear_asset.write_root_pose_to_sim(large_state[:, 0:7], env_ids=env_ids)
            self._large_gear_asset.write_root_velocity_to_sim(large_state[:, 7:], env_ids=env_ids)
            self._large_gear_asset.reset()

        # (1c) Noisy observation of the gear base (action target frame).
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

        default_hand_quat = self.fingertip_midpoint_quat.clone()

        # Medium-peg tip — observation/action frame (matches FactoryEnv gear_mesh logic).
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        fixed_tip_pos_local[:, 0] = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]
        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # (2) Place the gear on the table at a randomized offset from the gear base.
        # The offset is applied IN BASE LOCAL FRAME (rotated by base yaw) so the gear
        # always lands in the base's "narrow side" — pegs sit along the base's local
        # +X axis (small @ +5 cm, medium @ +2 cm, large @ -3 cm) so the rectangular
        # plate is long in X, narrow in Y. With the offset = [0, -7 cm, 0] in base
        # local, the gear sits 7 cm off the narrow edge regardless of how the base
        # is rotated. (Previously the offset was applied in env-local, so when the
        # base rotated 90° its long axis pointed at the gear and they overlapped.)
        gear_table_offset = torch.tensor(
            self.cfg_task.gear_table_offset, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)
        rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        gear_pos_noise = (2 * (rand_sample - 0.5)) @ torch.diag(
            torch.tensor(self.cfg_task.gear_table_pos_noise, device=self.device)
        )

        # Combined offset in base's local frame, then rotated by base yaw into env-local.
        offset_local_xy = gear_table_offset[:, 0:2] + gear_pos_noise[:, 0:2]
        _, _, base_yaw = torch_utils.get_euler_xyz(self._fixed_asset.data.root_quat_w)
        cos_y, sin_y = torch.cos(base_yaw), torch.sin(base_yaw)
        offset_world_x = offset_local_xy[:, 0] * cos_y - offset_local_xy[:, 1] * sin_y
        offset_world_y = offset_local_xy[:, 0] * sin_y + offset_local_xy[:, 1] * cos_y

        fixed_pos_local_w = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        gear_pos_local = fixed_pos_local_w.clone()
        gear_pos_local[:, 0] += offset_world_x
        gear_pos_local[:, 1] += offset_world_y
        # Force gear local z to 0 — same convention as gear base, both sit on the table.
        gear_pos_local[:, 2] = 0.0
        # Gear z = gear-base z (gear USD origin sits at the gear bottom face — same
        # convention the small/large flanking gears use, so this puts the gear flat
        # on the table next to the base). disable_gravity holds it there at reset.

        # Random yaw around the world z-axis so the gear spins flat on the table.
        yaw_rand = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0 * self.cfg_task.gear_table_yaw_range
        zeros = torch.zeros_like(yaw_rand)
        gear_quat = torch_utils.quat_from_euler_xyz(roll=zeros, pitch=zeros, yaw=yaw_rand)

        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = gear_pos_local + self.scene.env_origins
        held_state[:, 3:7] = gear_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

        # (3) IK the gripper to a randomized pose above the gear; gripper stays open.
        gear_top_pos = gear_pos_local.clone()
        gear_top_pos[:, 2] += self.cfg_task.held_asset_cfg.height

        bad_envs = env_ids.clone()
        hand_task_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        hand_task_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        while True:
            n_bad = bad_envs.shape[0]

            above_gear_pos = gear_top_pos.clone()
            above_gear_pos[:, 2] += self.cfg_task.hand_init_pos[2]

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_gear_pos_rand = 2 * (rand_sample - 0.5)
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
            above_gear_pos_rand = above_gear_pos_rand @ torch.diag(hand_init_pos_rand)
            above_gear_pos[bad_envs] += above_gear_pos_rand

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_gear_orn_noise = 2 * (rand_sample - 0.5)
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_gear_orn_noise = above_gear_orn_noise @ torch.diag(hand_init_orn_rand)
            orn_noise_quat = torch_utils.quat_from_euler_xyz(
                roll=above_gear_orn_noise[:, 0],
                pitch=above_gear_orn_noise[:, 1],
                yaw=above_gear_orn_noise[:, 2],
            )
            hand_task_quat[bad_envs, :] = torch_utils.quat_mul(default_hand_quat[bad_envs], orn_noise_quat)
            hand_task_pos[bad_envs, :] = above_gear_pos[bad_envs]

            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=hand_task_pos,
                ctrl_target_fingertip_midpoint_quat=hand_task_quat,
                env_ids=bad_envs,
            )
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
            angle_error = torch.norm(aa_error, dim=1) > 1e-3
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            if bad_envs.shape[0] == 0:
                break

            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], env_ids=bad_envs
            )

        self.step_sim_no_action()

        # (4) Restore gains and zero out previous-step buffers.
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

        # Let the gear settle on the table under gravity before caching its resting z.
        # The initial set z sits ~3 cm above the table (gear-base origin convention),
        # so capturing before settling makes lift reward structurally negative and
        # clamps it to 0 forever — which is exactly what we observed in the log.
        for _ in range(5):
            self.step_sim_no_action()
        gear_pos_settled = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.gear_initial_z = gear_pos_settled[:, 2].clone()
