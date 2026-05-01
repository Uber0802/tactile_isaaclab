# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import copy
import os
import sys

import numpy as np
import torch

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

import isaacsim.core.utils.torch as torch_utils

from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor
from isaaclab.utils.math import axis_angle_from_quat

from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

from . import forge_utils
from .forge_env_cfg import ForgeEnvCfg


class ForgeEnv(FactoryEnv):
    cfg: ForgeEnvCfg

    def _setup_scene(self):
        """Initialize simulation scene and optional tactile sensors."""
        super()._setup_scene()

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

    def __init__(self, cfg: ForgeEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize additional randomization and logging tensors."""
        super().__init__(cfg, render_mode, **kwargs)

        if "left_tactile_sensor" in self.scene.sensors and "right_tactile_sensor" in self.scene.sensors:
            left_sensor = self.scene.sensors["left_tactile_sensor"]
            right_sensor = self.scene.sensors["right_tactile_sensor"]
            left_sensor.get_initial_render()
            right_sensor.get_initial_render()

        # Success prediction.
        self.success_pred_scale = 0.0
        self.first_pred_success_tx = {}
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh] = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Per-episode success rate tracking (episode X: num_success_envs / num_envs).
        # env_episode_index[i]: how many episodes env i has completed.
        # pending_episode_successes[i]: success result for the current episode (-1 = not yet reported).
        self.env_episode_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.pending_episode_successes = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )

        # Flip quaternions.
        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        # Force sensor information.
        self.force_sensor_body_idx = self._robot.body_names.index("force_sensor")
        self.force_sensor_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_sensor_world_smooth = torch.zeros((self.num_envs, 6), device=self.device)

        # Set nominal dynamics parameters for randomization.
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_dead_zone = torch.tensor(self.cfg.ctrl.default_dead_zone, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()

        self._save_tactile_force_field = os.getenv("FORGE_SAVE_TACTILE_FORCE_FIELD", "0") == "1"
        self._tactile_save_interval = max(1, int(os.getenv("FORGE_TACTILE_SAVE_INTERVAL", "1")))
        self._tactile_save_dir = os.getenv("FORGE_TACTILE_SAVE_DIR", "./tactile_dataset")
        self._tactile_saved_episode_count = 0
        self._tactile_step_in_episode = 0
        self._tactile_episode_frames = []
        if self._save_tactile_force_field:
            os.makedirs(self._tactile_save_dir, exist_ok=True)

        # Optional Tactile-ReWiND progress reward.
        self._init_tactile_reward()

    def _init_tactile_reward(self):
        """Optional dense reward bonus from a Tactile-ReWiND ckpt.

        Activated when env var FORGE_TACTILE_REWARD_CKPT points at a .pth.
        Knobs:
            FORGE_TACTILE_REWARD_CKPT         (str path; empty = disabled)
            FORGE_TACTILE_REWARD_SCALE        (float, default 1.0)
            FORGE_TACTILE_REWARD_INSTRUCTION  (default "grasp peg and insert to another hole")
            FORGE_TACTILE_REWARD_ROOT         (path to Tactile-ReWiND repo for sys.path)
        """
        self._tactile_reward_enabled = False
        ckpt = os.getenv("FORGE_TACTILE_REWARD_CKPT", "").strip()
        if not ckpt:
            return

        # Make Tactile-ReWiND/tools/ importable.
        rewind_root = os.path.expanduser(os.getenv(
            "FORGE_TACTILE_REWARD_ROOT",
            "~/tactile_isaaclab/external/third-party/Tactile-ReWiND",
        ))
        if rewind_root not in sys.path:
            sys.path.insert(0, rewind_root)
        try:
            from tools.tactile_model import TactileReWiNDTransformer
        except Exception as e:
            print(f"[TactileReward] FAILED import (rewind_root={rewind_root}): {e}")
            return

        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        cfg = state.get("args", {})
        num_strided = cfg.get("num_strided_layers", None) or 3
        bimanual_axis = cfg.get("bimanual_axis", None) or "height"
        self._tactile_model = TactileReWiNDTransformer(
            max_length=cfg.get("max_length", 16),
            text_dim=384,
            hidden_dim=cfg.get("hidden_dim", 512),
            num_heads=cfg.get("num_heads", 8),
            num_layers=cfg.get("num_layers", 4),
            per_hand_dim=cfg.get("per_hand_dim", 384),
            num_strided_layers=num_strided,
            bimanual_axis=bimanual_axis,
        ).to(self.device)
        self._tactile_model.load_state_dict(state["model_state_dict"])
        self._tactile_model.eval()
        self._tactile_model_max_length = cfg.get("max_length", 16)

        # Encode instruction once via MiniLM, then drop the encoder.
        instruction = os.getenv(
            "FORGE_TACTILE_REWARD_INSTRUCTION",
            "grasp peg and insert to another hole",
        )
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        minilm = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        ).to(self.device)
        minilm.eval()
        with torch.no_grad():
            enc = tok([instruction], padding=True, return_tensors="pt").to(self.device)
            out = minilm(**enc)
            tok_emb = out[0]
            mask = enc["attention_mask"].unsqueeze(-1).expand(tok_emb.size()).float()
            text_emb = (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        del minilm, tok
        self._tactile_text_emb = text_emb.float()                 # (1, 384)

        # Per-env rolling buffer: (B, T, 40, 25, 2) bimanual H-stacked, Fx/Fy.
        self._tactile_buffer = torch.zeros(
            self.num_envs, self._tactile_model_max_length, 40, 25, 2,
            device=self.device, dtype=torch.float32,
        )
        self._tactile_reward_scale = float(
            os.getenv("FORGE_TACTILE_REWARD_SCALE", "1.0"))
        self._tactile_reward_enabled = True
        print(f"[TactileReward] enabled  ckpt={ckpt}  scale={self._tactile_reward_scale}  "
              f"instruction={instruction!r}")

    def _compute_tactile_reward(self) -> torch.Tensor:
        """(num_envs,) predicted progress as a dense reward bonus."""
        if not getattr(self, "_tactile_reward_enabled", False):
            return torch.zeros(self.num_envs, device=self.device)

        left = self.scene.sensors["left_tactile_sensor"]
        right = self.scene.sensors["right_tactile_sensor"]
        nrows, ncols = left.cfg.tactile_array_size           # (20, 25)
        l_shear = left.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        r_shear = right.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        current = torch.cat([l_shear, r_shear], dim=1).float()   # (B, 40, 25, 2)

        # Roll window left, append current frame.
        self._tactile_buffer = torch.roll(self._tactile_buffer, shifts=-1, dims=1)
        self._tactile_buffer[:, -1] = current

        # (B, T, H, W, C) -> (B, T, C, H, W) for the encoder.
        x = self._tactile_buffer.permute(0, 1, 4, 2, 3).contiguous()
        text = self._tactile_text_emb.expand(self.num_envs, -1)
        with torch.no_grad():
            progress = self._tactile_model(x, text).squeeze(-1)   # (B, T)
        return progress[:, -1] * self._tactile_reward_scale

    def _get_tactile_force_tensors(self, sensor_name: str):
        """Return flattened normal/shear tactile force tensors for a registered sensor."""
        sensor = self.scene.sensors[sensor_name]
        if sensor.cfg.enable_camera_tactile and getattr(sensor, "_nominal_tactile", None) is None:
            sensor.get_initial_render()
        sensor_data = sensor.data
        num_rows, num_cols = sensor.cfg.tactile_array_size
        num_pts = num_rows * num_cols

        normal_force = sensor_data.tactile_normal_force
        if normal_force is None:
            normal_force = torch.zeros((self.num_envs, num_pts), device=self.device)

        shear_force = sensor_data.tactile_shear_force
        if shear_force is None:
            shear_force = torch.zeros((self.num_envs, num_pts, 2), device=self.device)

        return normal_force, shear_force.reshape(self.num_envs, num_pts * 2)

    def get_left_tactile_vector_field(self):
        """Return the left GelSight force field as (N, H, W, 3)."""
        sensor = self.scene.sensors["left_tactile_sensor"]
        if sensor.cfg.enable_camera_tactile and getattr(sensor, "_nominal_tactile", None) is None:
            sensor.get_initial_render()
        nrows, ncols = sensor.cfg.tactile_array_size
        normal_force = sensor.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
        shear_force = sensor.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        return torch.cat((normal_force, shear_force), dim=-1)

    def get_right_tactile_vector_field(self):
        """Return the right GelSight force field as (N, H, W, 3)."""
        sensor = self.scene.sensors["right_tactile_sensor"]
        if sensor.cfg.enable_camera_tactile and getattr(sensor, "_nominal_tactile", None) is None:
            sensor.get_initial_render()
        nrows, ncols = sensor.cfg.tactile_array_size
        normal_force = sensor.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
        shear_force = sensor.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        return torch.cat((normal_force, shear_force), dim=-1)

    def _flush_tactile_episode(self, success: int = 0):
        """Write the buffered target-env tactile tensors for the current episode.

        Saved file is a dict (np.save with allow_pickle=True):
            {"Task":     <fixed task description>,
             "Tactile":  np.ndarray (T, H, W, C) float16,
             "Success":  int 0 / 1}
        Load with `np.load(path, allow_pickle=True).item()`.
        """
        if not self._save_tactile_force_field or not self._tactile_episode_frames:
            return

        episode_path = os.path.join(self._tactile_save_dir, f"ep{self._tactile_saved_episode_count}.npy")
        episode_tensor = np.stack(self._tactile_episode_frames, axis=0).astype(np.float16, copy=False)
        payload = {
            "Task": "grasp peg and insert to another hole",
            "Tactile": episode_tensor,
            "Success": int(success),
        }
        np.save(episode_path, payload, allow_pickle=True)
        self._tactile_episode_frames.clear()
        self._tactile_saved_episode_count += 1

    def _save_env0_tactile_force_field(self):
        """Buffer target-env tactile tensors and flush one .npy file per episode."""
        if not self._save_tactile_force_field:
            return

        target_env_id = min(71, self.num_envs - 1)

        # Detect episode boundary: target env just reset this step.
        if self.reset_buf[target_env_id]:
            # `ep_succeeded[target_env_id]` reflects the just-finished episode's
            # outcome (set during the step's success check, before reset).
            success = (
                int(self.ep_succeeded[target_env_id].item())
                if hasattr(self, "ep_succeeded") else 0
            )
            self._flush_tactile_episode(success=success)
            self._tactile_step_in_episode = 0

        # Respect save interval.
        if self._tactile_step_in_episode % self._tactile_save_interval != 0:
            self._tactile_step_in_episode += 1
            return

        left_sensor = self.scene.sensors["left_tactile_sensor"]
        right_sensor = self.scene.sensors["right_tactile_sensor"]
        left_rows, left_cols = left_sensor.cfg.tactile_array_size
        right_rows, right_cols = right_sensor.cfg.tactile_array_size

        left_normal_all = left_sensor.data.tactile_normal_force.view(self.num_envs, left_rows, left_cols)
        left_shear_all = left_sensor.data.tactile_shear_force.view(self.num_envs, left_rows, left_cols, 2)
        right_normal_all = right_sensor.data.tactile_normal_force.view(self.num_envs, right_rows, right_cols)
        right_shear_all = right_sensor.data.tactile_shear_force.view(self.num_envs, right_rows, right_cols, 2)

        left_normal = left_normal_all[target_env_id].detach().cpu().numpy()
        left_shear = left_shear_all[target_env_id].detach().cpu().numpy()
        right_normal = right_normal_all[target_env_id].detach().cpu().numpy()
        right_shear = right_shear_all[target_env_id].detach().cpu().numpy()

        left_force_field = np.concatenate((left_normal[..., None], left_shear), axis=-1)
        right_force_field = np.concatenate((right_normal[..., None], right_shear), axis=-1)
        tactile_frame = np.concatenate((left_force_field, right_force_field), axis=0)
        self._tactile_episode_frames.append(tactile_frame.astype(np.float16, copy=False))
        self._tactile_step_in_episode += 1

    def _compute_intermediate_values(self, dt):
        """Add noise to observations for force sensing."""
        super()._compute_intermediate_values(dt)

        # Add noise to fingertip position.
        pos_noise_level, rot_noise_level_deg = self.cfg.obs_rand.fingertip_pos, self.cfg.obs_rand.fingertip_rot_deg
        fingertip_pos_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        fingertip_pos_noise = fingertip_pos_noise @ torch.diag(
            torch.tensor([pos_noise_level, pos_noise_level, pos_noise_level], dtype=torch.float32, device=self.device)
        )
        self.noisy_fingertip_pos = self.fingertip_midpoint_pos + fingertip_pos_noise

        rot_noise_axis = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        rot_noise_axis /= torch.linalg.norm(rot_noise_axis, dim=1, keepdim=True)
        rot_noise_angle = torch.randn((self.num_envs,), dtype=torch.float32, device=self.device) * np.deg2rad(
            rot_noise_level_deg
        )
        self.noisy_fingertip_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_from_angle_axis(rot_noise_angle, rot_noise_axis)
        )
        self.noisy_fingertip_quat[:, [0, 3]] = 0.0
        self.noisy_fingertip_quat = self.noisy_fingertip_quat * self.flip_quats.unsqueeze(-1)

        # Repeat finite differencing with noisy fingertip positions.
        self.ee_linvel_fd = (self.noisy_fingertip_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.noisy_fingertip_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.noisy_fingertip_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.ee_angvel_fd[:, 0:2] = 0.0
        self.prev_fingertip_quat = self.noisy_fingertip_quat.clone()

        # Update and smooth force values.
        self.force_sensor_world = self._robot.root_physx_view.get_link_incoming_joint_force()[
            :, self.force_sensor_body_idx
        ]

        alpha = self.cfg.ft_smoothing_factor
        self.force_sensor_world_smooth = alpha * self.force_sensor_world + (1 - alpha) * self.force_sensor_world_smooth

        self.force_sensor_smooth = torch.zeros_like(self.force_sensor_world)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.force_sensor_smooth[:, :3], self.force_sensor_smooth[:, 3:6] = forge_utils.change_FT_frame(
            self.force_sensor_world_smooth[:, 0:3],
            self.force_sensor_world_smooth[:, 3:6],
            (identity_quat, torch.zeros((self.num_envs, 3), device=self.device)),
            (identity_quat, self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise),
        )

        # Compute noisy force values.
        force_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        force_noise *= self.cfg.obs_rand.ft_force
        self.noisy_force = self.force_sensor_smooth[:, 0:3] + force_noise

    def _get_observations(self):
        """Add additional FORGE observations."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()
        if "left_tactile_sensor" in self.scene.sensors:
            left_normal_force, left_shear_force = self._get_tactile_force_tensors("left_tactile_sensor")
            right_normal_force, right_shear_force = self._get_tactile_force_tensors("right_tactile_sensor")
            self._save_env0_tactile_force_field()
            obs_dict.update(
                {
                    "left_tactile_normal_force": left_normal_force,
                    "right_tactile_normal_force": right_normal_force,
                }
            )
            state_dict.update(
                {
                    "left_tactile_normal_force": left_normal_force,
                    "left_tactile_shear_force": left_shear_force,
                    "right_tactile_normal_force": right_normal_force,
                    "right_tactile_shear_force": right_shear_force,
                }
            )

        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        obs_dict.update(
            {
                "fingertip_pos": self.noisy_fingertip_pos,
                "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
                "fingertip_quat": self.noisy_fingertip_quat,
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "ft_force": self.noisy_force,
                "prev_actions": prev_actions,
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

    def _apply_action(self):
        """FORGE actions are defined as targets relative to the fixed asset."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Step (0): Scale actions to allowed range.
        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))

        # Step (1): Compute desired pose targets in EE frame.
        # (1.a) Position. Action frame is assumed to be the top of the bolt (noisy estimate).
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions
        # (1.b) Enforce rotation action constraints.
        rot_actions[:, 0:2] = 0.0

        # Assumes joint limit is in (+x, -y)-quadrant of world frame.
        rot_actions[:, 2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0  # Joint limit.
        # (1.c) Get desired orientation target.
        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )

        rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )

        ctrl_target_fingertip_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        # Step (2): Clip targets if they are too far from current EE pose.
        # (2.a): Clip position targets.
        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self.fingertip_midpoint_pos  # Used for action_penalty.
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped

        # (2.b) Clip orientation targets. Use Euler angles. We assume we are near upright, so
        # clipping yaw will effectively cause slow motions. When we clip, we also need to make
        # sure we avoid the joint limit.

        # (2.b.i) Get current and desired Euler angles.
        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_fingertip_preclipped_quat)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        # (2.b.ii) Correct the direction of motion to avoid joint limit.
        # Map yaws between [-125, 235] degrees
        # (so that angles appear on a continuous span uninterrupted by the joint limit)
        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        desired_yaw = factory_utils.wrap_yaw(desired_yaw)

        # (2.b.iii) Clip motion in the correct direction.
        self.delta_yaw = desired_yaw - curr_yaw  # Used later for action_penalty.
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        # (2.b.iv) Clip roll and pitch.
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

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def _get_rewards(self):
        """FORGE reward includes a contact penalty and success prediction error."""
        # Use same base rewards as Factory.
        rew_buf = super()._get_rewards()

        rew_dict, rew_scales = {}, {}
        # Calculate action penalty for the asset-relative action space.
        pos_error = torch.norm(self.delta_pos, p=2, dim=-1) / self.cfg.ctrl.pos_action_threshold[0]
        rot_error = torch.abs(self.delta_yaw) / self.cfg.ctrl.rot_action_threshold[0]
        # Contact penalty.
        contact_force = torch.norm(self.force_sensor_smooth[:, 0:3], p=2, dim=-1, keepdim=False)
        contact_penalty = torch.nn.functional.relu(contact_force - self.contact_penalty_thresholds)
        # Add success prediction rewards.
        check_rot = self.cfg_task.name == "nut_thread"
        true_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )
        policy_success_pred = (self.actions[:, 6] + 1) / 2  # rescale from [-1, 1] to [0, 1]
        success_pred_error = (true_successes.float() - policy_success_pred).abs()
        # Delay success prediction penalty until some successes have occurred.
        if true_successes.float().mean() >= self.cfg_task.delay_until_ratio:
            self.success_pred_scale = 1.0

        # Add new FORGE reward terms.
        rew_dict = {
            "action_penalty_asset": pos_error + rot_error,
            "contact_penalty": contact_penalty,
            "success_pred_error": success_pred_error,
        }
        rew_scales = {
            "action_penalty_asset": -self.cfg_task.action_penalty_asset_scale,
            "contact_penalty": -self.cfg_task.contact_penalty_scale,
            "success_pred_error": -self.success_pred_scale,
        }
        if getattr(self, "_tactile_reward_enabled", False):
            rew_dict["tactile_progress"] = self._compute_tactile_reward()
            # `_tactile_reward_scale` already baked in inside the helper.
            rew_scales["tactile_progress"] = 1.0
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self._log_forge_metrics(rew_dict, policy_success_pred)
        return rew_buf

    def _reset_idx(self, env_ids):
        """Perform additional randomizations."""
        super()._reset_idx(env_ids)

        # Compute initial action for correct EMA computation.
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        pos_actions = self.fingertip_midpoint_pos - fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

        # Relative yaw to bolt.
        unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )

        fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.fingertip_midpoint_quat)
        fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
        )

        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action
        self.actions[:, 6] = self.prev_actions[:, 6] = -1.0

        # EMA randomization.
        ema_rand = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device)
        ema_lower, ema_upper = self.cfg.ctrl.ema_factor_range
        self.ema_factor = ema_lower + ema_rand * (ema_upper - ema_lower)

        # Set initial gains for the episode.
        prop_gains = self.default_gains.clone()
        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()
        prop_gains = forge_utils.get_random_prop_gains(
            prop_gains, self.cfg.ctrl.task_prop_gains_noise_level, self.num_envs, self.device
        )
        self.pos_threshold = forge_utils.get_random_prop_gains(
            self.pos_threshold, self.cfg.ctrl.pos_threshold_noise_level, self.num_envs, self.device
        )
        self.rot_threshold = forge_utils.get_random_prop_gains(
            self.rot_threshold, self.cfg.ctrl.rot_threshold_noise_level, self.num_envs, self.device
        )
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(prop_gains)

        contact_rand = torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
        contact_lower, contact_upper = self.cfg.task.contact_penalty_threshold_range
        self.contact_penalty_thresholds = contact_lower + contact_rand * (contact_upper - contact_lower)

        self.dead_zone_thresholds = (
            torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device) * self.default_dead_zone
        )

        self.force_sensor_world_smooth[:, :] = 0.0

        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        rand_flips = torch.rand(self.num_envs) > 0.5
        self.flip_quats[rand_flips] = -1.0

    def _reset_buffers(self, env_ids):
        """Reset additional logging metrics."""
        super()._reset_buffers(env_ids)
        # Reset success pred metrics.
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh][env_ids] = 0
        # Clear tactile reward buffer for envs that just reset.
        if getattr(self, "_tactile_reward_enabled", False):
            self._tactile_buffer[env_ids] = 0

    def _log_forge_metrics(self, rew_dict, policy_success_pred):
        """Log metrics to evaluate success prediction performance."""
        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        for thresh, first_success_tx in self.first_pred_success_tx.items():
            curr_predicted_success = policy_success_pred > thresh
            first_success_idxs = torch.logical_and(curr_predicted_success, first_success_tx == 0)

            first_success_tx[:] = torch.where(first_success_idxs, self.episode_length_buf, first_success_tx)

            # Only log at the end.
            if torch.any(self.reset_buf):
                # Log prediction delay.
                delay_ids = torch.logical_and(self.ep_success_times != 0, first_success_tx != 0)
                delay_times = (first_success_tx[delay_ids] - self.ep_success_times[delay_ids]).sum() / delay_ids.sum()
                if delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_all/{thresh}"] = delay_times

                correct_delay_ids = torch.logical_and(delay_ids, first_success_tx > self.ep_success_times)
                correct_delay_times = (
                    first_success_tx[correct_delay_ids] - self.ep_success_times[correct_delay_ids]
                ).sum() / correct_delay_ids.sum()
                if correct_delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_correct/{thresh}"] = correct_delay_times.item()

                # Log early-term success rate (for all episodes we have "stopped", did we succeed?).
                pred_success_idxs = first_success_tx != 0  # Episodes which we have predicted success.

                true_success_preds = torch.logical_and(
                    self.ep_success_times[pred_success_idxs] > 0,  # Success has actually occurred.
                    self.ep_success_times[pred_success_idxs]
                    < first_success_tx[pred_success_idxs],  # Success occurred before we predicted it.
                )

                num_pred_success = pred_success_idxs.sum().item()
                et_prec = true_success_preds.sum() / num_pred_success
                if num_pred_success > 0:
                    self.extras[f"early_term_precision/{thresh}"] = et_prec

                true_success_idxs = self.ep_success_times > 0
                num_true_success = true_success_idxs.sum().item()
                et_recall = true_success_preds.sum() / num_true_success
                if num_true_success > 0:
                    self.extras[f"early_term_recall/{thresh}"] = et_recall

        # Per-episode success rate: logged to wandb only after ALL envs have reset.
        if torch.any(self.reset_buf):
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            # Record each resetting env's success result and advance its episode counter.
            self.pending_episode_successes[reset_env_ids] = self.ep_succeeded[reset_env_ids].long()
            self.env_episode_index[reset_env_ids] += 1

            # Only log once every env has reported for this episode.
            if (self.pending_episode_successes >= 0).all():
                episode_success_rate = self.pending_episode_successes.float().mean()
                episode_idx = int(self.env_episode_index.min().item()) - 1
                self.pending_episode_successes.fill_(-1)

                if _wandb is not None and _wandb.run is not None:
                    _wandb.log(
                        {
                            "episode_success_rate": episode_success_rate.item(),
                            "episode_index": episode_idx,
                        }
                    )

    def close(self):
        """Flush any buffered tactile episode before tearing down the environment."""
        self._flush_tactile_episode()
        super().close()
