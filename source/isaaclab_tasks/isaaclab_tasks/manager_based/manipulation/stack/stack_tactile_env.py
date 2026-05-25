
import os
import sys
import copy
import time
import torch
import numpy as np
from typing import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

class StackTactileEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: StackEnvCfg, render_mode: str | None = None, **kwargs):
        if os.environ.get("FORGE_FIXED_OBJECT_POS", "0") == "1":
            if hasattr(cfg, "events") and hasattr(cfg.events, "randomize_cube_positions"):
                delattr(cfg.events, "randomize_cube_positions")
                cfg.events.randomize_cube_positions_1 = EventTerm(
                    func=franka_stack_events.randomize_object_pose,
                    mode="reset",
                    params={
                        "pose_range": {"x": (0.46, 0.46), "y": (-0.05, -0.05), "yaw": (-1.0, 1, 0)},
                        "asset_cfgs": [SceneEntityCfg("stack_object"),],
                    },
                )
                cfg.events.randomize_cube_positions_2 = EventTerm(
                    func=franka_stack_events.randomize_object_pose,
                    mode="reset",
                    params={
                        "pose_range": {"x": (0.54, 0.54), "y": (0.05, 0.05), "yaw": (-1.0, 1, 0)},
                        "asset_cfgs": [SceneEntityCfg("target_cube")],
                    },
                )


        super().__init__(cfg, render_mode, **kwargs)

        # Set friction on stack_object at runtime (UsdFileCfg does not support physics_material)
        if hasattr(self.cfg, 'set_friction') and "stack_object" in self.scene.keys():
            materials = self.scene["stack_object"].root_physx_view.get_material_properties()
            materials[..., 0] = 1.0  # static friction
            materials[..., 1] = 1.2 # dynamic friction
            materials[..., 2] = 0.0  # restitution
            env_ids = torch.arange(self.num_envs, device="cpu")
            self.scene["stack_object"].root_physx_view.set_material_properties(materials, env_ids)
        # Success tracking
        self.ep_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pending_episode_successes = torch.ones(self.num_envs, dtype=torch.long, device=self.device) * -1
        self.env_episode_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Tactile saving settings (mirrored from ForgeEnv)
        self._save_tactile_force_field = os.environ.get("FORGE_SAVE_TACTILE_FORCE_FIELD", "0") == "1"
        self._save_tactile_all_envs = os.environ.get("FORGE_SAVE_TACTILE_ALL_ENVS", "0") == "1"
        self._tactile_save_dir = os.environ.get("FORGE_TACTILE_SAVE_DIR", "./tactile_dataset/data")
        self._tactile_save_interval = int(os.environ.get("FORGE_TACTILE_SAVE_INTERVAL", "1"))
        self._tactile_reward_instruction = os.environ.get("FORGE_TACTILE_REWARD_INSTRUCTION", "stack an object on a box")
        self._tactile_max_buffer_frames = int(os.environ.get("FORGE_TACTILE_MAX_BUFFER_FRAMES", "500000"))
        self._tactile_episodes_per_env = int(os.environ.get("FORGE_TACTILE_EPISODES_PER_ENV", "0"))  # 0 = unlimited

        self._save_front_cam = self._save_tactile_force_field and ("front_cam" in self.scene.keys())

        if self._save_tactile_force_field:
            os.makedirs(self._tactile_save_dir, exist_ok=True)
            if self._save_tactile_all_envs:
                self._tactile_episode_frames = [[] for _ in range(self.num_envs)]
                self._tactile_step_in_episode_per_env = [0] * self.num_envs
                if self._save_front_cam:
                    self._camera_episode_frames = [[] for _ in range(self.num_envs)]
            else:
                self._tactile_episode_frames = []
                if self._save_front_cam:
                    self._camera_episode_frames = []
            self._tactile_saved_episode_count = 0
            self._tactile_step_in_episode = 0
            self._tactile_saved_per_env = [0] * self.num_envs  # episodes saved per env
            print(f"[StackTactileEnv] Saving tactile force field to: {self._tactile_save_dir}")
            if self._save_front_cam:
                print(f"[StackTactileEnv] Saving front camera to: {self._tactile_save_dir}")

        # Optional Tactile-ReWiND progress reward.
        self._init_tactile_reward()
    
    def _get_tactile_vector_field(self, sensor_name: str):
        """Return the GelSight force field for a given sensor as (N, H, W, 3)."""
        if sensor_name not in self.scene.sensors:
            return None
        sensor = self.scene.sensors[sensor_name]
        
        nrows, ncols = sensor.cfg.tactile_array_size
        normal_force = sensor.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
        shear_force = sensor.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        return torch.cat((normal_force, shear_force), dim=-1)

    def _init_tactile_reward(self):
        """Optional dense reward bonus from a Tactile-ReWiND ckpt.

        Activated when env var FORGE_TACTILE_REWARD_CKPT points at a .pth.
        Knobs:
            FORGE_TACTILE_REWARD_CKPT         (str path; empty = disabled)
            FORGE_TACTILE_REWARD_SCALE        (float, default 1.0)
            FORGE_TACTILE_REWARD_INSTRUCTION  (default "stack an object on a box")
            FORGE_TACTILE_REWARD_ROOT         (path to Tactile-ReWiND repo for sys.path)
            FORGE_TACTILE_REWARD_HISTORY      (int, default = self.max_episode_length):
                                              rolling-buffer length; linspace-subsampled
                                              down to max_length when fed to the model —
                                              matches training's `_sample_forward +
                                              _resize` stride behavior. Default matches
                                              the full episode so a slice that ends at
                                              episode-end approximates the `start=0,
                                              end=N` training case (progress → 1).
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
        
        cfg_shear = cfg.get("shear_channels", None)
        if cfg_shear:
            shear_channels = tuple(cfg_shear)
        else:
            ic = int(cfg.get("in_channels", 2))
            shear_channels = (0, 1, 2) if ic == 3 else (1, 2)
        in_channels = len(shear_channels)
        self._tactile_model = TactileReWiNDTransformer(
            max_length=cfg.get("max_length", 16),
            text_dim=384,
            hidden_dim=cfg.get("hidden_dim", 512),
            num_heads=cfg.get("num_heads", 8),
            num_layers=cfg.get("num_layers", 4),
            per_hand_dim=cfg.get("per_hand_dim", 384),
            num_strided_layers=num_strided,
            bimanual_axis=bimanual_axis,
            in_channels=in_channels,
        ).to(self.device)
        self._tactile_model.load_state_dict(state["model_state_dict"])
        self._tactile_model.eval()
        self._tactile_model_max_length = cfg.get("max_length", 16)
        self._tactile_in_channels = in_channels
        self._tactile_shear_channels = shear_channels
        
        norm_mode = cfg.get("normalize_mode", None)
        if norm_mode is None:
            norm_mode = "per_channel" if cfg.get("normalize_per_channel") else "off"
        self._tactile_normalize_mode = norm_mode

        # Encode instruction once via MiniLM, then drop the encoder.
        instruction = os.getenv(
            "FORGE_TACTILE_REWARD_INSTRUCTION",
            "stack an object on a box",
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

        # Per-env rolling history buffer.
        default_history = int(getattr(self, "max_episode_length", 150)) #TODO: Why 150?
        env_history = os.getenv("FORGE_TACTILE_REWARD_HISTORY", "").strip()
        self._tactile_history_length = int(env_history) if env_history else default_history
        if self._tactile_history_length < self._tactile_model_max_length:
            self._tactile_history_length = self._tactile_model_max_length
        self._tactile_buffer = torch.zeros(
            self.num_envs, self._tactile_history_length, 40, 25, in_channels,
            device=self.device, dtype=torch.float32,
        )
        # Per-env count of valid frames since last reset (clamped to H).
        self._tactile_step_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long,
        )
        self._tactile_reward_scale = float(
            os.getenv("FORGE_TACTILE_REWARD_SCALE", "1.0"))
        self._tactile_reward_smooth_alpha = float(
            os.getenv("FORGE_TACTILE_REWARD_SMOOTH_ALPHA", "1.0"))
        self._tactile_smoothed_progress = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32,
        )
        self._tactile_reward_enabled = True
        
        self._tactile_target_env = int(os.getenv("FORGE_TACTILE_REWARD_LOG_ENV", "0"))
        default_curve_dir = os.path.expanduser(
            f"~/tactile_isaaclab/logs/tactile_curves/{int(time.time())}"
        )
        self._tactile_curve_log_dir = os.getenv("FORGE_TACTILE_REWARD_LOG_DIR",
                                                default_curve_dir)
        self._tactile_progress_history = []
        self._tactile_curve_episode_idx = 0
        try:
            os.makedirs(self._tactile_curve_log_dir, exist_ok=True)
            curve_log_msg = f"  curve_log={self._tactile_curve_log_dir}"
        except Exception:
            self._tactile_curve_log_dir = None
            curve_log_msg = "  curve_log=DISABLED"
        print(f"[TactileReward] enabled  ckpt={ckpt}  scale={self._tactile_reward_scale}  "
              f"instruction={instruction!r}  history={self._tactile_history_length}  "
              f"normalize={self._tactile_normalize_mode}  "
              f"smooth_alpha={self._tactile_reward_smooth_alpha}{curve_log_msg}")

    def compute_tactile_reward(self) -> torch.Tensor:
        """(num_envs,) predicted progress as a dense reward bonus."""
        if not getattr(self, "_tactile_reward_enabled", False):
            return torch.zeros(self.num_envs, device=self.device)

        if "left_tactile_sensor" not in self.scene.sensors or "right_tactile_sensor" not in self.scene.sensors:
            return torch.zeros(self.num_envs, device=self.device)

        left = self.scene.sensors["left_tactile_sensor"]
        right = self.scene.sensors["right_tactile_sensor"]
        nrows, ncols = left.cfg.tactile_array_size
        
        l_shear = left.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        r_shear = right.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        l_normal = left.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
        r_normal = right.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
        l_full = torch.cat([l_normal, l_shear], dim=-1)
        r_full = torch.cat([r_normal, r_shear], dim=-1)
        full = torch.cat([l_full, r_full], dim=1).float()
        current = full[..., list(self._tactile_shear_channels)]

        H = self._tactile_history_length
        T = self._tactile_model_max_length
        self._tactile_buffer = torch.roll(self._tactile_buffer, shifts=-1, dims=1)
        self._tactile_buffer[:, -1] = current
        self._tactile_step_count = torch.clamp(self._tactile_step_count + 1, max=H)
        valid = self._tactile_step_count.clamp(min=1)
        start = (H - valid).long()

        device = self._tactile_buffer.device
        t_grid = torch.arange(T, device=device)
        if T > 1:
            frac = t_grid.float() / float(T - 1)
        else:
            frac = torch.zeros(T, device=device)
        span = (H - 1 - start).float().unsqueeze(1)
        long_idx = (start.float().unsqueeze(1) + span * frac.unsqueeze(0)).round().long()
        long_idx.clamp_(0, H - 1)
        
        short_idx = (start.unsqueeze(1) + torch.minimum(t_grid.unsqueeze(0), (valid - 1).unsqueeze(1)))
        short_idx.clamp_(0, H - 1)
        
        is_long = (valid >= T).unsqueeze(1)
        sel = torch.where(is_long, long_idx, short_idx)

        _, _, Hh, Ww, Cc = self._tactile_buffer.shape
        gather_idx = sel[:, :, None, None, None].expand(-1, -1, Hh, Ww, Cc)
        slc = torch.gather(self._tactile_buffer, 1, gather_idx)

        if self._tactile_normalize_mode == "global":
            denom = slc.abs().amax(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
            slc = slc / denom
        elif self._tactile_normalize_mode == "per_channel":
            denom = slc.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            slc = slc / denom

        x = slc.permute(0, 1, 4, 2, 3).contiguous()
        text = self._tactile_text_emb.expand(self.num_envs, -1)
        with torch.no_grad():
            progress = self._tactile_model(x, text).squeeze(-1)
        latest = progress[:, -1]

        alpha = self._tactile_reward_smooth_alpha
        if alpha < 1.0:
            self._tactile_smoothed_progress = (
                alpha * latest + (1.0 - alpha) * self._tactile_smoothed_progress
            )
            out = self._tactile_smoothed_progress
        else:
            self._tactile_smoothed_progress = latest
            out = latest

        if (self._tactile_curve_log_dir is not None
                and self._tactile_target_env < self.num_envs):
            self._tactile_progress_history.append((
                latest[self._tactile_target_env].item(),
                out[self._tactile_target_env].item(),
            ))
        return out * self._tactile_reward_scale

    def _save_tactile_progress_curve(self) -> None:
        """Dump the current `_tactile_progress_history` to disk as a PNG."""
        if (not self._tactile_curve_log_dir
                or not self._tactile_progress_history):
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[TactileReward] matplotlib unavailable, skipping curve dump: {e}")
            return
        history = self._tactile_progress_history
        scale = self._tactile_reward_scale
        alpha = self._tactile_reward_smooth_alpha
        raw_series = [t[0] for t in history]
        sm_series = [t[1] for t in history]
        fig, ax = plt.subplots(figsize=(8, 4))
        steps = list(range(len(history)))
        ax.plot(steps, raw_series, color="C0", linewidth=1.0, alpha=0.45,
                label="raw progress")
        if alpha < 1.0:
            ax.plot(steps, sm_series, color="C2", linewidth=1.5,
                    label=f"EMA smoothed (α={alpha})")
        ax.plot(steps, [v * scale for v in sm_series], color="C1",
                linewidth=1.2, linestyle="--",
                label=f"× scale={scale} (rew contribution)")
        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_xlabel("env step within episode")
        ax.set_ylabel("tactile progress")
        ax.set_title(
            f"env {self._tactile_target_env} | episode {self._tactile_curve_episode_idx} "
            f"| steps={len(history)}")
        ax.set_ylim(-0.1, 1.2)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        out_path = os.path.join(
            self._tactile_curve_log_dir,
            f"env{self._tactile_target_env}_ep{self._tactile_curve_episode_idx:06d}.png",
        )
        try:
            fig.tight_layout()
            fig.savefig(out_path, dpi=110)
        except Exception as e:
            print(f"[TactileReward] failed to save curve PNG: {e}")
        plt.close(fig)
        self._tactile_curve_episode_idx += 1

    def _flush_tactile_episode(self, success: int = 0, env_id: int | None = None):
        """Write the buffered target-env tactile tensors for the current episode."""
        if not self._save_tactile_force_field:
            return

        frames = (
            self._tactile_episode_frames if env_id is None
            else self._tactile_episode_frames[env_id]
        )
        if not frames:
            return

        if env_id is None:
            fname = f"ep{self._tactile_saved_episode_count}.npy"
        else:
            fname = f"ep{self._tactile_saved_episode_count}_env{env_id:03d}.npy"

        episode_path = os.path.join(self._tactile_save_dir, fname)
        episode_tensor = np.stack(frames, axis=0).astype(np.float16, copy=False)
        payload = {
            "Task": self._tactile_reward_instruction,
            "Tactile": episode_tensor,
            "Success": int(success),
        }
        np.save(episode_path, payload, allow_pickle=True)
        frames.clear()

        # Flush front camera if enabled and we have frames
        if self._save_front_cam:
            cam_frames = (
                self._camera_episode_frames if env_id is None
                else self._camera_episode_frames[env_id]
            )
            if cam_frames:
                if env_id is None:
                    cam_fname = f"ep{self._tactile_saved_episode_count}_camera.npy"
                else:
                    cam_fname = f"ep{self._tactile_saved_episode_count}_env{env_id:03d}_camera.npy"
                cam_path = os.path.join(self._tactile_save_dir, cam_fname)
                cam_tensor = np.stack(cam_frames, axis=0).astype(np.uint8, copy=False)
                np.save(cam_path, cam_tensor)
                cam_frames.clear()

        self._tactile_saved_episode_count += 1
        if env_id is not None:
            self._tactile_saved_per_env[env_id] += 1

    def _save_env0_tactile_force_field(self):
        """Buffer target-env tactile tensors and flush one .npy file per episode."""
        if not self._save_tactile_force_field:
            return

        if self._save_tactile_all_envs:
            self._save_all_envs_tactile_force_field()
            return

        # Follow ForgeEnv logic: target env 0 (or some fixed env)
        target_env_id = 0 

        # Detect episode boundary: target env just reset this step.
        # ManagerBasedRLEnv uses reset_buf
        if self.reset_buf[target_env_id]:
            success = int(self.ep_succeeded[target_env_id].item())
            self._flush_tactile_episode(success=success)
            self._tactile_step_in_episode = 0

        # Respect save interval.
        if self._tactile_step_in_episode % self._tactile_save_interval != 0:
            self._tactile_step_in_episode += 1
            return

        left_field = self._get_tactile_vector_field("left_tactile_sensor")
        right_field = self._get_tactile_vector_field("right_tactile_sensor")

        if left_field is not None and right_field is not None:
            # (H, W, 6)
            combined_field = torch.cat((left_field[target_env_id], right_field[target_env_id]), dim=-1)
            self._tactile_episode_frames.append(combined_field.cpu().numpy())

            # Fetch front camera frame if enabled
            if self._save_front_cam:
                cam_data = self.scene["front_cam"].data.output["rgb"][target_env_id, ..., :3]
                self._camera_episode_frames.append(cam_data.cpu().numpy().astype(np.uint8, copy=False))
        
        self._tactile_step_in_episode += 1

    def _save_all_envs_tactile_force_field(self):
        """Multi-env variant: every env keeps its own per-episode buffer."""
        quota = self._tactile_episodes_per_env
        reset_envs = torch.nonzero(self.reset_buf, as_tuple=False).flatten().tolist()
        for env_id in reset_envs:
            if quota > 0 and self._tactile_saved_per_env[env_id] >= quota:
                self._tactile_episode_frames[env_id].clear()
                if self._save_front_cam:
                    self._camera_episode_frames[env_id].clear()
            else:
                success = int(self.ep_succeeded[env_id].item())
                self._flush_tactile_episode(success=success, env_id=env_id)
            self._tactile_step_in_episode_per_env[env_id] = 0

        if quota > 0 and all(c >= quota for c in self._tactile_saved_per_env):
            print(
                f"[TactileSave] All {self.num_envs} envs reached {quota} episodes "
                f"({self._tactile_saved_episode_count} total)."
            )
            return

        total_frames = sum(len(buf) for buf in self._tactile_episode_frames)
        if total_frames >= self._tactile_max_buffer_frames:
            if not getattr(self, "_tactile_overflow_warned", False):
                print(
                    f"[TactileSave] WARNING: per-env buffers hold {total_frames} "
                    f"frames (cap {self._tactile_max_buffer_frames}); pausing "
                    f"appends until next flush."
                )
                self._tactile_overflow_warned = True
            for env_id in range(self.num_envs):
                self._tactile_step_in_episode_per_env[env_id] += 1
            return
        self._tactile_overflow_warned = False

        left_field = self._get_tactile_vector_field("left_tactile_sensor")
        right_field = self._get_tactile_vector_field("right_tactile_sensor")

        if left_field is not None and right_field is not None:
            # (B, H, W, 6)
            combined_fields = torch.cat((left_field, right_field), dim=-1).detach().cpu().numpy()
            
            # Fetch all front camera frames at once to avoid env-by-env GPU-CPU transfers.
            if self._save_front_cam:
                # Shape: (B, H, W, :3)
                cam_fields = self.scene["front_cam"].data.output["rgb"][..., :3].detach().cpu().numpy().astype(np.uint8, copy=False)
            
            for env_id in range(self.num_envs):
                step = self._tactile_step_in_episode_per_env[env_id]
                if step % self._tactile_save_interval == 0:
                    self._tactile_episode_frames[env_id].append(combined_fields[env_id].astype(np.float16, copy=False))
                    if self._save_front_cam:
                        self._camera_episode_frames[env_id].append(cam_fields[env_id])
                self._tactile_step_in_episode_per_env[env_id] = step + 1

    def step(self, action: torch.Tensor):
        # ManagerBasedRLEnv.step() calls _step_impl()
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update ep_succeeded. We consider success if it ever succeeded during the episode.
        if "stack_success" in self.reward_manager.active_terms:
             term_idx = self.reward_manager.active_terms.index("stack_success")
             success_reward = self.reward_manager._step_reward[:, term_idx]
             self.ep_succeeded |= (success_reward > 0)

        self._save_env0_tactile_force_field()

        if torch.any(self.reset_buf):
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.pending_episode_successes[reset_env_ids] = self.ep_succeeded[reset_env_ids].long()
            self.env_episode_index[reset_env_ids] += 1

            if (self.pending_episode_successes >= 0).all():
                episode_success_rate = self.pending_episode_successes.float().mean()
                self.extras["episode_success_rate"] = episode_success_rate.item()
                self.pending_episode_successes.fill_(-1)
            
            # Reset ep_succeeded for next episode
            self.ep_succeeded[reset_env_ids] = False

        return obs, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        obs, info = super().reset(seed=seed, env_ids=env_ids, options=options)
        
        # Sim has stepped by now, robot is at initial pose, no contact
        if 'left_tactile_sensor' in self.scene.keys() and self.scene['left_tactile_sensor']._nominal_tactile is None:  # Only on first reset
            self.scene['left_tactile_sensor'].get_initial_render()
        if 'right_tactile_sensor' in self.scene.keys() and self.scene['right_tactile_sensor']._nominal_tactile is None:
            self.scene['right_tactile_sensor'].get_initial_render()
        
        self.ep_succeeded[env_ids] = False

        # Clear tactile reward buffer and step counts for resetting envs.
        if getattr(self, "_tactile_reward_enabled", False):
            # Save curve when target env resets
            if self._tactile_target_env in env_ids:
                self._save_tactile_progress_curve()
                self._tactile_progress_history.clear()
            
            # Reset rolling buffer and step counters
            self._tactile_buffer[env_ids] = 0.0
            self._tactile_step_count[env_ids] = 0
            self._tactile_smoothed_progress[env_ids] = 0.0

        return obs, info

    def close(self):
        if self._save_tactile_force_field:
            if self._save_tactile_all_envs:
                # In multi-env mode, we typically only save episodes that completed (hit reset).
                # Partial episodes at shutdown are discarded to avoid incomplete trajectories.
                pass
            else:
                self._flush_tactile_episode(success=int(self.ep_succeeded[0].item()))
        super().close()
    
    def _post_physics_step(self):
        # Force update tactile sensors after each physics step
        if 'left_tactile_sensor' in self.scene.keys():
            self.scene['left_tactile_sensor'].update(
                dt=self.physics_dt, force_recompute=True
            )
        if 'right_tactile_sensor' in self.scene.keys():
            self.scene['right_tactile_sensor'].update(
                dt=self.physics_dt, force_recompute=True
            )
        
        super()._post_physics_step()  # or your existing post-physics logic
