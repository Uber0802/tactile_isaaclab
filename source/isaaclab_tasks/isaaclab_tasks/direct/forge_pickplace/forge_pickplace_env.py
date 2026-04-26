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
# Default path sits under the IsaacLab tree so machines that mount /mnt/home share it;
# override with PICKPLACE_LOG if you want somewhere else.
_PP_LOG_PATH = os.getenv(
    "PICKPLACE_LOG",
    "/mnt/home/uber/IsaacLab/logs/pickplace_metrics.log",
)
_PP_LOG_INTERVAL = int(os.getenv("PICKPLACE_LOG_INTERVAL", "200"))
os.makedirs(os.path.dirname(_PP_LOG_PATH), exist_ok=True)

def _pp_log(line: str):
    print(line, file=sys.stderr, flush=True)
    try:
        with open(_PP_LOG_PATH, "a") as f:
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

from .forge_pickplace_env_cfg import ForgeTaskPegInsertPickPlaceCfg


class ForgePegInsertPickPlaceEnv(ForgeEnv):
    """Pick-and-place peg variant.

    Scene contains two holes (source and destination). At reset, the peg rests freely
    inside the source hole and the gripper starts open above it. The policy must
    learn to close the gripper, pick up the peg, and insert it into the destination
    hole. Action index 7 controls the gripper; index 6 remains the success predictor.
    """

    cfg: ForgeTaskPegInsertPickPlaceCfg
    _pp_metric_step = 0
    _pp_metric_start_t = time.perf_counter()

    def _setup_scene(self):
        """Replicate FactoryEnv + ForgeEnv scene setup and add the source hole."""
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
        self._source_fixed_asset = Articulation(self.cfg_task.source_fixed_asset)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        self.scene.articulations["source_fixed_asset"] = self._source_fixed_asset

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

    def _set_assets_to_default_pose(self, env_ids):
        """Also reset the source hole to its default pose."""
        super()._set_assets_to_default_pose(env_ids)

        source_state = self._source_fixed_asset.data.default_root_state.clone()[env_ids]
        source_state[:, 0:3] += self.scene.env_origins[env_ids]
        source_state[:, 7:] = 0.0
        self._source_fixed_asset.write_root_pose_to_sim(source_state[:, 0:7], env_ids=env_ids)
        self._source_fixed_asset.write_root_velocity_to_sim(source_state[:, 7:], env_ids=env_ids)
        self._source_fixed_asset.reset()

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Force-open the gripper at every reset (default Factory behavior clamps around the peg)."""
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

    def _pre_physics_step(self, action):
        """Faster EMA for the gripper so it can actually close, but not instant (which
        made the gripper slam on the peg and fling it up). ~0.3 gives a few env-step
        ramp instead of the body EMA's ~30-step ramp or the earlier zero-step bypass.
        """
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        raw_action = action.clone().to(self.device)
        self.actions = self.ema_factor * raw_action + (1 - self.ema_factor) * self.actions
        gripper_ema = self.cfg_task.gripper_ema
        self.actions[:, 7] = gripper_ema * raw_action[:, 7] + (1 - gripper_ema) * self.actions[:, 7]

    def _get_rewards(self):
        """Forge peg-insert reward plus pick-place shaping (approach + lift)."""
        rew_buf = super()._get_rewards()

        # (1) Gripper → peg grasp point (peg top minus 1/4 peg height) in world frame.
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        grasp_offset_local = torch.zeros((self.num_envs, 3), device=self.device)
        grasp_offset_local[:, 2] = self.cfg_task.peg_grasp_z_offset
        _, peg_grasp_pos = torch_utils.tf_combine(
            self.held_quat, self.held_pos, identity_quat, grasp_offset_local
        )
        fingertip_to_peg = torch.norm(self.fingertip_midpoint_pos - peg_grasp_pos, p=2, dim=-1)
        r_approach = torch.exp(-fingertip_to_peg / self.cfg_task.approach_scale)

        # (2) Peg lift height above the source-hole base, capped by fingertip height.
        # Why cap: without this, policy learned to punt the peg with the gripper — peg
        # flies up and lift reward saturates even though nothing was actually grasped.
        # min(peg_z, fingertip_z) only grows when peg and fingertip rise together,
        # i.e. when the peg is physically moving with the hand.
        source_pos_local = self._source_fixed_asset.data.root_pos_w - self.scene.env_origins
        source_base_z = source_pos_local[:, 2]
        peg_z_rel_source = self.held_pos[:, 2] - source_base_z
        fingertip_z_rel_source = self.fingertip_midpoint_pos[:, 2] - source_base_z
        effective_lift = torch.clamp(
            torch.minimum(peg_z_rel_source, fingertip_z_rel_source), min=0.0
        )
        r_lift = torch.tanh(effective_lift / self.cfg_task.lift_scale)

        # (3) Peg speed penalty — discourages the physics-hack where closing the gripper
        # too hard ejects the peg at multi-m/s speeds.
        peg_speed = torch.norm(self._held_asset.data.root_lin_vel_w, dim=-1)
        peg_speed_excess = torch.clamp(peg_speed - self.cfg_task.peg_speed_threshold, min=0.0)
        r_peg_speed_penalty = -self.cfg_task.peg_speed_penalty_scale * peg_speed_excess

        # Peg → destination geometry (used for diagnostics and descent gate).
        dest_pos_local = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        peg_to_dest_xy = torch.norm(self.held_pos[:, 0:2] - dest_pos_local[:, 0:2], dim=-1)
        peg_above_dest_z = self.held_pos[:, 2] - dest_pos_local[:, 2]
        peg_to_dest_3d = torch.sqrt(peg_to_dest_xy**2 + torch.clamp(peg_above_dest_z, min=0.0) ** 2)

        # (4) XY-gated descent reward — strictly zero outside the gate so it doesn't
        # distort the reach/lift phase. Once the peg is XY-aligned within threshold,
        # pay for lowering it toward the destination base.
        xy_aligned = (peg_to_dest_xy < self.cfg_task.descent_xy_threshold).float()
        z_progress = torch.exp(
            -torch.clamp(peg_above_dest_z, min=0.0) / self.cfg_task.descent_z_scale
        )
        r_descent = xy_aligned * z_progress

        rew_buf = (
            rew_buf
            + self.cfg_task.approach_reward_scale * r_approach
            + self.cfg_task.lift_reward_scale * r_lift
            + r_peg_speed_penalty
            + self.cfg_task.descent_reward_scale * r_descent
        )

        # ---- Rich diagnostics (reward-only; these do NOT feed back into the policy) ----
        # Gripper state.
        gripper_action_cmd = self.actions[:, 7]
        gripper_joint_width = self.joint_pos[:, 7]
        gripper_open_frac = torch.clamp(gripper_joint_width / self.cfg_task.gripper_open_width, 0.0, 1.0)

        # Lift above source tip (not base) — positive means peg is clear of source hole.
        source_tip_z = source_base_z + self.cfg_task.fixed_asset_cfg.base_height + self.cfg_task.fixed_asset_cfg.height
        peg_z_above_source_tip = self.held_pos[:, 2] - source_tip_z

        # Heuristic grasp detection (logged only): fingertip close AND gripper narrow.
        grasped = (
            (fingertip_to_peg < self.cfg_task.grasp_log_dist_threshold)
            & (gripper_joint_width < self.cfg_task.grasp_log_gripper_threshold)
        ).float()

        # Heuristic "lifted-and-grasped" — peg clearly out of source hole AND grasp signal.
        lifted = (peg_z_above_source_tip > 0.01).float()
        pick_success = (grasped * lifted)

        self.extras["logs_rew_approach"] = r_approach.mean()
        self.extras["logs_rew_lift"] = r_lift.mean()
        self.extras["logs_fingertip_to_peg"] = fingertip_to_peg.mean()
        self.extras["logs_fingertip_to_peg_min"] = fingertip_to_peg.min()
        self.extras["logs_peg_z_rel_source"] = peg_z_rel_source.mean()
        self.extras["logs_peg_z_above_source_tip"] = peg_z_above_source_tip.mean()
        self.extras["logs_peg_z_above_source_tip_max"] = peg_z_above_source_tip.max()
        self.extras["logs_peg_to_dest_xy"] = peg_to_dest_xy.mean()
        self.extras["logs_peg_to_dest_xy_min"] = peg_to_dest_xy.min()
        self.extras["logs_peg_above_dest_z"] = peg_above_dest_z.mean()
        self.extras["logs_peg_above_dest_z_min"] = peg_above_dest_z.min()
        self.extras["logs_peg_to_dest_3d_min"] = peg_to_dest_3d.min()
        self.extras["logs_gripper_action_cmd"] = gripper_action_cmd.mean()
        self.extras["logs_gripper_width"] = gripper_joint_width.mean()
        self.extras["logs_gripper_open_frac"] = gripper_open_frac.mean()
        self.extras["logs_grasp_heuristic_frac"] = grasped.mean()
        self.extras["logs_pick_lifted_frac"] = pick_success.mean()
        # Co-lift: how closely peg is tracking the fingertip in z (small = grasped, big = launched).
        peg_fingertip_dz = self.held_pos[:, 2] - self.fingertip_midpoint_pos[:, 2]
        self.extras["logs_peg_minus_fingertip_z"] = peg_fingertip_dz.mean()
        self.extras["logs_fingertip_z_rel_source"] = fingertip_z_rel_source.mean()
        self.extras["logs_effective_lift_mean"] = effective_lift.mean()
        self.extras["logs_effective_lift_max"] = effective_lift.max()
        self.extras["logs_peg_speed_mean"] = peg_speed.mean()
        self.extras["logs_peg_speed_max"] = peg_speed.max()
        self.extras["logs_rew_peg_speed_penalty"] = r_peg_speed_penalty.mean()
        self.extras["logs_rew_descent"] = r_descent.mean()
        self.extras["logs_xy_aligned_frac"] = xy_aligned.mean()

        # Periodic text log so we can tail metrics from stderr / the log file.
        type(self)._pp_metric_step += 1
        if self._pp_metric_step % _PP_LOG_INTERVAL == 0:
            success_rate = self.extras.get("successes")
            if isinstance(success_rate, torch.Tensor):
                success_rate = success_rate.item()
            elapsed = time.perf_counter() - self._pp_metric_start_t
            _pp_log(
                f"[pp step={self._pp_metric_step} t={elapsed:.0f}s] "
                f"rew={rew_buf.mean().item():+.3f} "
                f"approach={r_approach.mean().item():.3f} "
                f"lift={r_lift.mean().item():.3f} "
                f"fin2peg(min/mean)={fingertip_to_peg.min().item():.3f}/{fingertip_to_peg.mean().item():.3f} "
                f"gripper(cmd/width)={gripper_action_cmd.mean().item():+.2f}/{gripper_joint_width.mean().item():.4f} "
                f"peg_lift(mean/max)={peg_z_above_source_tip.mean().item():+.3f}/{peg_z_above_source_tip.max().item():+.3f} "
                f"eff_lift(mean/max)={effective_lift.mean().item():.3f}/{effective_lift.max().item():.3f} "
                f"peg-fing_dz={peg_fingertip_dz.mean().item():+.3f} "
                f"pegV(mean/max)={peg_speed.mean().item():.2f}/{peg_speed.max().item():.2f} "
                f"pegXY2dest(min/mean)={peg_to_dest_xy.min().item():.3f}/{peg_to_dest_xy.mean().item():.3f} "
                f"pegZabvDest(min/mean)={peg_above_dest_z.min().item():+.4f}/{peg_above_dest_z.mean().item():+.4f} "
                f"descent={r_descent.mean().item():.3f} "
                f"xyAlign%={xy_aligned.mean().item():.2f} "
                f"grasp%={grasped.mean().item():.2f} "
                f"lifted%={pick_success.mean().item():.2f} "
                f"success={success_rate if success_rate is not None else 'nan'}"
            )
        return rew_buf

    def randomize_initial_state(self, env_ids):
        """Place peg freely inside the source hole and leave the gripper open above it."""
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1) Randomize destination hole pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]

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

        # (1b) Place source hole at destination hole + configured offset.
        source_offset = torch.tensor(self.cfg_task.source_hole_offset, dtype=torch.float32, device=self.device)
        source_state = self._source_fixed_asset.data.default_root_state.clone()[env_ids]
        source_state[:, 0:3] = fixed_state[:, 0:3] + source_offset.unsqueeze(0)
        source_state[:, 3:7] = fixed_state[:, 3:7]
        source_state[:, 7:] = 0.0
        self._source_fixed_asset.write_root_pose_to_sim(source_state[:, 0:7], env_ids=env_ids)
        self._source_fixed_asset.write_root_velocity_to_sim(source_state[:, 7:], env_ids=env_ids)
        self._source_fixed_asset.reset()

        # (1c) Noisy observation of the destination hole (insertion target).
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

        default_hand_quat = self.fingertip_midpoint_quat.clone()

        # Destination tip — observation/action frame.
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # Source tip — gripper spawn reference.
        source_pos_w = self._source_fixed_asset.data.root_pos_w - self.scene.env_origins
        source_quat_w = self._source_fixed_asset.data.root_quat_w
        _, source_tip_pos = torch_utils.tf_combine(
            source_quat_w,
            source_pos_w,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            fixed_tip_pos_local,
        )

        # (2) Seat the peg inside the source hole (upright, gravity-free in reset window).
        held_state = self._held_asset.data.default_root_state.clone()
        peg_pos_local = source_pos_w.clone()
        peg_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height + self.cfg_task.peg_source_insertion
        held_state[:, 0:3] = peg_pos_local + self.scene.env_origins
        held_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

        # (3) IK the gripper to a randomized pose above the source hole tip; gripper stays open.
        bad_envs = env_ids.clone()
        hand_task_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        hand_task_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        while True:
            n_bad = bad_envs.shape[0]

            above_source_pos = source_tip_pos.clone()
            above_source_pos[:, 2] += self.cfg_task.hand_init_pos[2]

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_source_pos_rand = 2 * (rand_sample - 0.5)
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
            above_source_pos_rand = above_source_pos_rand @ torch.diag(hand_init_pos_rand)
            above_source_pos[bad_envs] += above_source_pos_rand

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_source_orn_noise = 2 * (rand_sample - 0.5)
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_source_orn_noise = above_source_orn_noise @ torch.diag(hand_init_orn_rand)
            orn_noise_quat = torch_utils.quat_from_euler_xyz(
                roll=above_source_orn_noise[:, 0],
                pitch=above_source_orn_noise[:, 1],
                yaw=above_source_orn_noise[:, 2],
            )
            hand_task_quat[bad_envs, :] = torch_utils.quat_mul(default_hand_quat[bad_envs], orn_noise_quat)
            hand_task_pos[bad_envs, :] = above_source_pos[bad_envs]

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

        # (4) Restore gains and warm up a few steps; peg has gravity disabled so it stays put.
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
