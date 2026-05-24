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
_NP_LOG_PATH = os.getenv(
    "NUTPICKPLACE_LOG",
    "/mnt/home/tactile/tactile_isaaclab/logs/nutpickplace_metrics.log",
)
_NP_LOG_INTERVAL = int(os.getenv("NUTPICKPLACE_LOG_INTERVAL", "200"))
os.makedirs(os.path.dirname(_NP_LOG_PATH), exist_ok=True)


def _np_log(line: str):
    print(line, file=sys.stderr, flush=True)
    try:
        with open(_NP_LOG_PATH, "a") as f:
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

from .forge_nutpickplace_env_cfg import ForgeTaskNutThreadPickPlaceCfg


class ForgeNutThreadPickPlaceEnv(ForgeEnv):
    """Pick-and-place nut-threading variant.

    The M16 nut rests freely on the table at a randomized offset from the bolt.
    The gripper starts open above the nut. The policy must close the gripper, pick
    up the nut, transport it over the bolt, lower it onto the threads, and rotate
    the wrist (negative direction) to thread it down.
    Action index 7 controls the gripper; index 6 remains the success predictor.
    """

    cfg: ForgeTaskNutThreadPickPlaceCfg
    _np_metric_step = 0
    _np_metric_start_t = time.perf_counter()

    def _setup_scene(self):
        """Replicate FactoryEnv + ForgeEnv scene setup (bolt + nut + tactile)."""
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

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset

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
        """Same control law as ForgeEnv but with a policy-controlled gripper width.

        Yaw is interpreted as an absolute target in bolt frame (range [-180°, +90°]),
        then clipped per-step. We re-apply unidirectional_rot afterward by clamping
        commanded delta yaw to the negative half — required because the parent
        FactoryEnv applies that constraint inside its own _apply_action, which we
        bypass here.
        """
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
        # Bug fix: factory_utils.wrap_yaw only puts angles >235° into (-125°, 0°],
        # not a true [-π, π] wrap. Subtracting two wrapped yaws across the
        # boundary gives a "long-way-around" delta whose sign is flipped from
        # the actual short arc. Re-wrap the delta to (-π, π] so the
        # unidirectional_rot clamp below acts on the true threading-direction
        # delta, otherwise it zeros out legitimate negative rotations and the
        # wrist gets stuck.
        self.delta_yaw = torch.atan2(
            torch.sin(desired_yaw - curr_yaw), torch.cos(desired_yaw - curr_yaw)
        )
        # Threading must rotate one direction (negative) only — clamp positive deltas to 0.
        if getattr(self.cfg_task, "unidirectional_rot", False):
            self.delta_yaw = torch.clamp(self.delta_yaw, max=0.0)
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
        held_quat live only in the critic state. For pick-place the nut's table
        position is essential — without it the actor can't locate the nut at all.
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

        obs_dict.update(
            {
                "fingertip_pos": self.noisy_fingertip_pos,
                "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
                "fingertip_pos_rel_held": self.noisy_fingertip_pos - self.held_pos,
                "fingertip_quat": self.noisy_fingertip_quat,
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "ft_force": self.noisy_force,
                "prev_actions": prev_actions,
                "held_pos": self.held_pos,
                "held_pos_rel_fixed": self.held_pos - noisy_fixed_pos,
                "held_quat": self.held_quat,
                "fixed_pos": self.fixed_pos,
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
        """Forge nut-thread reward plus pick-place shaping (reach + lift + descent).

        No binary "is nut grasped" detection — only continuous distance / lift /
        descent shaping plus a nut-speed penalty for physics regularization.
        Yaw alignment is *not* rewarded here: nut_thread's success XY check is
        yaw-independent (the geometric base offset is purely vertical), and
        wrist-yaw progress is handled by the base ForgeNutThread `curr_success`
        bonus combined with the unidirectional_rot constraint in _apply_action.
        """
        rew_buf = super()._get_rewards()

        # (1) Reach: gripper → nut grasp point (nut top minus a small offset).
        # Sum of sharp + broad exponentials so the gradient survives at the
        # initial ~10 cm distance instead of being numerically zero.
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        grasp_offset_local = torch.zeros((self.num_envs, 3), device=self.device)
        grasp_offset_local[:, 2] = self.cfg_task.nut_grasp_z_offset
        _, nut_grasp_pos = torch_utils.tf_combine(
            self.held_quat, self.held_pos, identity_quat, grasp_offset_local
        )
        fingertip_to_nut = torch.norm(self.fingertip_midpoint_pos - nut_grasp_pos, p=2, dim=-1)
        r_approach = 0.5 * (
            torch.exp(-fingertip_to_nut / self.cfg_task.approach_scale)
            + torch.exp(-fingertip_to_nut / 0.3)
        )

        # (2) Object lift: nut z above its initial table z, multiplied by a
        # continuous proximity gate so the policy can't farm lift by raising the
        # gripper without bringing the nut along.
        nut_z_rel_table = self.held_pos[:, 2] - self.nut_initial_z
        proximity_gate = torch.exp(-fingertip_to_nut / self.cfg_task.lift_proximity_scale)
        r_lift = proximity_gate * torch.tanh(
            torch.clamp(nut_z_rel_table, min=0.0) / self.cfg_task.lift_scale
        )

        # (3) Nut speed penalty — discourages closing the gripper so hard the nut is
        # ejected at multi-m/s speeds (physics regularizer, not a grasp signal).
        nut_speed = torch.norm(self._held_asset.data.root_lin_vel_w, dim=-1)
        nut_speed_excess = torch.clamp(nut_speed - self.cfg_task.nut_speed_threshold, min=0.0)
        r_nut_speed_penalty = -self.cfg_task.nut_speed_penalty_scale * nut_speed_excess

        # Descent target = the *success target* used by `_get_curr_successes`.
        # For nut_thread:
        #   held_base_pos   = nut_pos  + (0, 0, base_height)
        #   target_held_pos = bolt_pos + (0, 0, head_height + shank - 1.5*pitch)
        # Both use identity quat, so the XY component reduces to ||nut_xy - bolt_xy||
        # — yaw of either asset doesn't enter the success XY check.
        held_base_pos, _ = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, _ = factory_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        nut_to_target_xy = torch.norm(held_base_pos[:, 0:2] - target_held_base_pos[:, 0:2], dim=-1)
        nut_above_target_z = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        # (4) Continuous XY + Z gates — exp(-d/scale) instead of hard thresholds so the
        # policy keeps getting gradient as it tightens past the success tolerances
        # (xy < 2.5 mm, z_disp < 0.75 mm).
        #
        # z_dist is centered on the "ideal" z_disp = the threshold at which success
        # turns True (thread_pitch * success_threshold). Two regimes:
        #   - success_threshold ≥ 0 (original A): ideal ≈ target z. Use abs() so
        #     deviations in either direction lose reward — prevents the farming
        #     bug where policy parks a slightly-lifted nut over the bolt without
        #     actually threading.
        #   - success_threshold < 0 (A_hard_success, e.g. -2.0 → 2 turns deep):
        #     ideal is BELOW target z. Use one-sided clamp so going past ideal
        #     (deeper, toward bolt head) keeps full reward — fixes the
        #     "park at z_disp = 0 = local-max trap" that otherwise blocks deep
        #     threading because both r_descent and r_z_descend would decrease as
        #     the nut moves past target.
        xy_aligned = torch.exp(-nut_to_target_xy / self.cfg_task.xy_alignment_scale)
        ideal_z_disp = (self.cfg_task.fixed_asset_cfg.thread_pitch
                        * self.cfg_task.success_threshold)
        if ideal_z_disp >= 0:
            z_dist = torch.abs(nut_above_target_z)
        else:
            z_dist = torch.clamp(nut_above_target_z - ideal_z_disp, min=0.0)
        # `dist_to_success_z` = how much shallower than the success boundary the
        # nut is right now. 0 means at-or-past success in z; positive means must
        # still descend that much further.
        dist_to_success_z = torch.clamp(nut_above_target_z - ideal_z_disp, min=0.0)
        z_progress = torch.exp(-z_dist / self.cfg_task.descent_z_scale)
        r_descent = xy_aligned * z_progress

        # (5) Transport-phase XY alignment. The fine `xy_aligned` saturates to ~0
        # past 2 cm, and `r_descent` is also killed by `z_progress` while the nut
        # is still high above the bolt. Without a coarser xy term the policy has
        # no horizontal pull during transport. Gated on `r_lift` so this can't be
        # farmed by sliding the nut on the table.
        xy_coarse = torch.exp(-nut_to_target_xy / self.cfg_task.xy_coarse_scale)
        r_xy_align = r_lift * xy_coarse

        # (6) Z-descent reward, fires only once XY is roughly aligned (coarse).
        # Bridges the gap between r_xy_align (pure xy, no z signal) and r_descent
        # (5 mm × 5 mm, both saturated during transport): gives gradient pulling
        # the nut down while it's already over the bolt at 2–10 cm height.
        # Bug fix: same one-sided-clamp farming issue as r_descent — use the
        # symmetric |z_above_target| via the shared `z_dist` so a nut parked
        # below the target z doesn't max this term out.
        z_progress_coarse = torch.exp(-z_dist / self.cfg_task.z_coarse_scale)
        r_z_descend = r_lift * xy_coarse * z_progress_coarse

        # (7) Wrist-yaw reward — success requires `wrap_yaw(fingertip_yaw) < 0`
        # (factory_env._get_curr_successes with check_rot=True for nut_thread).
        # `unidirectional_rot` biases the wrist toward negative but only fires
        # via the sparse `curr_success` AND-gate; provide a dense signal so the
        # policy actively drives wrist yaw negative once positioned. Gated on
        # `xy_coarse` so the policy doesn't burn yaw budget far from the bolt.
        _, _, fingertip_yaw_world = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        fingertip_yaw_wrapped = factory_utils.wrap_yaw(fingertip_yaw_world)
        # Saturate at 1.0 once wrist is already in the success window (≤0); decay
        # exponentially as yaw is positive. yaw_progress_scale = 0.5 rad ≈ 28.6°.
        positive_yaw = torch.clamp(fingertip_yaw_wrapped, min=0.0)
        yaw_progress = torch.exp(-positive_yaw / self.cfg_task.yaw_progress_scale)
        r_yaw = xy_coarse * yaw_progress

        rew_buf = (
            rew_buf
            + self.cfg_task.approach_reward_scale * r_approach
            + self.cfg_task.lift_reward_scale * r_lift
            + r_nut_speed_penalty
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
        self.extras["logs_rew_nut_speed_penalty"] = r_nut_speed_penalty.mean()
        self.extras["logs_rew_descent"] = r_descent.mean()
        self.extras["logs_rew_xy_align"] = r_xy_align.mean()
        self.extras["logs_xy_coarse_mean"] = xy_coarse.mean()
        self.extras["logs_rew_z_descend"] = r_z_descend.mean()
        self.extras["logs_z_progress_coarse_mean"] = z_progress_coarse.mean()
        self.extras["logs_rew_yaw"] = r_yaw.mean()
        self.extras["logs_fingertip_yaw_wrapped"] = fingertip_yaw_wrapped.mean()
        self.extras["logs_fingertip_yaw_negative_frac"] = (fingertip_yaw_wrapped < 0).float().mean()
        self.extras["logs_fingertip_to_nut"] = fingertip_to_nut.mean()
        self.extras["logs_fingertip_to_nut_min"] = fingertip_to_nut.min()
        self.extras["logs_nut_z_rel_table"] = nut_z_rel_table.mean()
        self.extras["logs_nut_z_rel_table_max"] = nut_z_rel_table.max()
        self.extras["logs_proximity_gate_mean"] = proximity_gate.mean()
        self.extras["logs_nut_speed_mean"] = nut_speed.mean()
        self.extras["logs_nut_speed_max"] = nut_speed.max()
        self.extras["logs_nut_to_target_xy"] = nut_to_target_xy.mean()
        self.extras["logs_nut_to_target_xy_min"] = nut_to_target_xy.min()
        self.extras["logs_nut_above_target_z"] = nut_above_target_z.mean()
        self.extras["logs_nut_above_target_z_min"] = nut_above_target_z.min()
        self.extras["logs_dist_to_success_z"] = dist_to_success_z.mean()
        self.extras["logs_dist_to_success_z_min"] = dist_to_success_z.min()
        self.extras["logs_gripper_action_cmd"] = gripper_action_cmd.mean()
        self.extras["logs_gripper_width"] = gripper_joint_width.mean()
        self.extras["logs_gripper_open_frac"] = gripper_open_frac.mean()
        self.extras["logs_xy_aligned_mean"] = xy_aligned.mean()

        type(self)._np_metric_step += 1
        if self._np_metric_step % _NP_LOG_INTERVAL == 0:
            success_rate = self.extras.get("successes")
            if isinstance(success_rate, torch.Tensor):
                success_rate = success_rate.item()
            elapsed = time.perf_counter() - self._np_metric_start_t
            _np_log(
                f"[np step={self._np_metric_step} t={elapsed:.0f}s] "
                f"rew={rew_buf.mean().item():+.3f} "
                f"approach={r_approach.mean().item():.3f} "
                f"lift={r_lift.mean().item():.3f} "
                f"fin2nut(min/mean)={fingertip_to_nut.min().item():.3f}/{fingertip_to_nut.mean().item():.3f} "
                f"gripper(cmd/width)={gripper_action_cmd.mean().item():+.2f}/{gripper_joint_width.mean().item():.4f} "
                f"nutLift(mean/max)={nut_z_rel_table.mean().item():+.3f}/{nut_z_rel_table.max().item():+.3f} "
                f"nutV(mean/max)={nut_speed.mean().item():.2f}/{nut_speed.max().item():.2f} "
                f"nut2tgt_xy(min/mean)={nut_to_target_xy.min().item():.4f}/{nut_to_target_xy.mean().item():.4f} "
                f"nut_z_above_tgt(min/mean)={nut_above_target_z.min().item():+.4f}/{nut_above_target_z.mean().item():+.4f} "
                f"distToSuccZ(mean/min)={dist_to_success_z.mean().item():.4f}/{dist_to_success_z.min().item():.4f} "
                f"descent={r_descent.mean().item():.3f} "
                f"xyAlignReward={r_xy_align.mean().item():.3f} "
                f"xyCoarse(mean)={xy_coarse.mean().item():.3f} "
                f"zDescendReward={r_z_descend.mean().item():.3f} "
                f"zCoarse(mean)={z_progress_coarse.mean().item():.3f} "
                f"yawReward={r_yaw.mean().item():.3f} "
                f"yaw(mean/negFrac)={fingertip_yaw_wrapped.mean().item():+.2f}/{(fingertip_yaw_wrapped < 0).float().mean().item():.2f} "
                f"xyAlign(mean)={xy_aligned.mean().item():.3f} "
                f"success={success_rate if success_rate is not None else 'nan'}"
            )
        return rew_buf

    def randomize_initial_state(self, env_ids):
        """Place nut freely on the table at a random offset from the bolt; gripper opens above the nut."""
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1) Randomize bolt (fixed_asset) pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        # Force bolt local z to 0 (overrides USD default + any z noise).
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

        # (1b) Noisy observation of the bolt (action target frame).
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

        default_hand_quat = self.fingertip_midpoint_quat.clone()

        # Bolt tip — observation/action frame. For nut_thread there's no
        # horizontal offset (unlike gear_mesh's medium_gear_base_offset), so the
        # frame is purely the bolt origin + (height + base_height) along z.
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

        # (2) Place the nut on the table at a randomized offset from the bolt.
        nut_table_offset = torch.tensor(
            self.cfg_task.nut_table_offset, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1)
        rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        nut_pos_noise = (2 * (rand_sample - 0.5)) @ torch.diag(
            torch.tensor(self.cfg_task.nut_table_pos_noise, device=self.device)
        )

        fixed_pos_local_w = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        nut_pos_local = fixed_pos_local_w.clone()
        nut_pos_local[:, 0:2] += (nut_table_offset[:, 0:2] + nut_pos_noise[:, 0:2])
        # Force nut local z to 0 — same convention as bolt, both sit on the table.
        nut_pos_local[:, 2] = 0.0

        # Random yaw around the world z-axis so the nut spins flat on the table.
        yaw_rand = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0 * self.cfg_task.nut_table_yaw_range
        zeros = torch.zeros_like(yaw_rand)
        nut_quat = torch_utils.quat_from_euler_xyz(roll=zeros, pitch=zeros, yaw=yaw_rand)

        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = nut_pos_local + self.scene.env_origins
        held_state[:, 3:7] = nut_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

        # (3) IK the gripper to a randomized pose above the nut; gripper stays open.
        nut_top_pos = nut_pos_local.clone()
        nut_top_pos[:, 2] += self.cfg_task.held_asset_cfg.height

        bad_envs = env_ids.clone()
        hand_task_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        hand_task_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        while True:
            n_bad = bad_envs.shape[0]

            above_nut_pos = nut_top_pos.clone()
            above_nut_pos[:, 2] += self.cfg_task.hand_init_pos[2]

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_nut_pos_rand = 2 * (rand_sample - 0.5)
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
            above_nut_pos_rand = above_nut_pos_rand @ torch.diag(hand_init_pos_rand)
            above_nut_pos[bad_envs] += above_nut_pos_rand

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_nut_orn_noise = 2 * (rand_sample - 0.5)
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_nut_orn_noise = above_nut_orn_noise @ torch.diag(hand_init_orn_rand)
            orn_noise_quat = torch_utils.quat_from_euler_xyz(
                roll=above_nut_orn_noise[:, 0],
                pitch=above_nut_orn_noise[:, 1],
                yaw=above_nut_orn_noise[:, 2],
            )
            hand_task_quat[bad_envs, :] = torch_utils.quat_mul(default_hand_quat[bad_envs], orn_noise_quat)
            hand_task_pos[bad_envs, :] = above_nut_pos[bad_envs]

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

        # Let the nut settle on the table under gravity before caching its resting z.
        # Capturing before settling makes lift reward structurally negative and
        # clamps it to 0 forever.
        for _ in range(5):
            self.step_sim_no_action()
        nut_pos_settled = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.nut_initial_z = nut_pos_settled[:, 2].clone()
