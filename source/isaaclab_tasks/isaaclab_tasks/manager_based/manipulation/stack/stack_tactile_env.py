
import os
import sys
import copy
import torch
import numpy as np
from pathlib import Path
from typing import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events


def _import_tactile_reward_model():
    """Import ``TactileRewardModel``, adding the repo root to ``sys.path`` if needed.

    ``tactile_reward_model/`` lives at the repository root, which is not on
    ``sys.path`` when ``isaaclab_tasks`` is used as an installed package.
    """
    try:
        from tactile_reward_model import TactileRewardModel
        return TactileRewardModel
    except ImportError:
        pass
    for parent in Path(__file__).resolve().parents:
        if (parent / "tactile_reward_model" / "tactile_reward_model.py").is_file():
            sys.path.insert(0, str(parent))
            from tactile_reward_model import TactileRewardModel
            return TactileRewardModel
    raise ImportError("could not locate the tactile_reward_model package")


class StackTactileEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: StackEnvCfg, render_mode: str | None = None, **kwargs):
        if os.environ.get("FORGE_FIXED_OBJECT_POS", "0") == "1":
            if hasattr(cfg, "events") and hasattr(cfg.events, "randomize_cube_positions"):
                delattr(cfg.events, "randomize_cube_positions")
                cfg.events.randomize_cube_positions_1 = EventTerm(
                    func=franka_stack_events.randomize_object_pose,
                    mode="reset",
                    params={
                        "pose_range": {"x": (0.46, 0.46), "y": (-0.05, -0.05), "yaw": (-1.0, -1.0)},
                        "asset_cfgs": [SceneEntityCfg("stack_object"),],
                    },
                )
                cfg.events.randomize_cube_positions_2 = EventTerm(
                    func=franka_stack_events.randomize_object_pose,
                    mode="reset",
                    params={
                        "pose_range": {"x": (0.54, 0.54), "y": (0.05, 0.05), "yaw": (-1.0, -1.0)},
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
        self.pending_episode_successes_at_end = torch.ones(self.num_envs, dtype=torch.long, device=self.device) * -1
        self.env_episode_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_episode_success_rate = 0.0

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

        # Optional frozen tactile-AE encoder for the `tactile_embedding` obs.
        self._init_tactile_encoder()
    
    def _get_tactile_vector_field(self, sensor_name: str):
        """Return the GelSight force field for a given sensor as (N, H, W, 3)."""
        if sensor_name not in self.scene.sensors:
            return None
        sensor = self.scene.sensors[sensor_name]
        
        nrows, ncols = sensor.cfg.tactile_array_size
        normal_force = sensor.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
        shear_force = sensor.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        return torch.cat((normal_force, shear_force), dim=-1)

    def _tactile_row_stacked_field(self):
        """(N, 2*rows, cols, 3) force field with the hands stacked on the ROW axis.

        This is the layout both the ReWiND reward model and the AE encoder are
        trained on: channels = (normal, shear_x, shear_y), rows 0-19 = left
        finger, 20-39 = right finger. Note it differs from the dataset dump in
        `_flush_tactile_episode`, which stacks the hands on the CHANNEL axis —
        `train_tactile_ae.py` converts to this form on load.

        Returns None when the sensors are absent or have not rendered yet.
        """
        if ("left_tactile_sensor" not in self.scene.sensors
                or "right_tactile_sensor" not in self.scene.sensors):
            return None
        left = self.scene.sensors["left_tactile_sensor"]
        right = self.scene.sensors["right_tactile_sensor"]
        for sensor in (left, right):
            if getattr(sensor, "_nominal_tactile", None) is None:
                return None

        nrows, ncols = left.cfg.tactile_array_size
        l_shear = left.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        r_shear = right.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        l_normal = left.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
        r_normal = right.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
        l_full = torch.cat([l_normal, l_shear], dim=-1)
        r_full = torch.cat([r_normal, r_shear], dim=-1)
        return torch.cat([l_full, r_full], dim=1).float()

    @staticmethod
    def _rewind_root() -> str:
        """Directory holding `tools/tactile_model.py`.

        The env-var overrides win, but the repo-relative path is the fallback
        that is correct by construction — the historical default
        (`~/tactile_isaaclab/...`) silently misses on checkouts that live
        anywhere else.
        """
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 6))
        candidates = [
            os.getenv("FORGE_TACTILE_ENCODER_ROOT", "").strip(),
            os.getenv("FORGE_TACTILE_REWARD_ROOT", "").strip(),
            os.path.join(repo_root, "external", "third-party", "Tactile-ReWiND"),
        ]
        for cand in candidates:
            if cand and os.path.isfile(
                os.path.join(os.path.expanduser(cand), "tools", "tactile_model.py")
            ):
                return os.path.expanduser(cand)
        return os.path.expanduser(candidates[-1])

    def _init_tactile_encoder(self):
        """Frozen tactile-AE latent exposed as the `tactile_embedding` obs key.

        Activated when FORGE_TACTILE_ENCODER_CKPT points at a checkpoint from
        `external/third-party/Tactile-ReWiND/train_tactile_ae.py`. Unlike the
        ReWiND reward model, this latent is trained purely for reconstruction —
        it carries "what the sensor feels", not task progress — so it gives the
        policy tactile-as-state without leaking a reward signal.

        Knobs:
            FORGE_TACTILE_ENCODER_CKPT  (str path; empty = disabled)
            FORGE_TACTILE_ENCODER_DIM   (int; must equal 2*per_hand_dim of the
                                        ckpt — the obs term's width is fixed
                                        from this var before the ckpt is read)
            FORGE_TACTILE_ENCODER_ROOT  (path to Tactile-ReWiND for sys.path)
        """
        self._tactile_encoder_enabled = False
        self._tactile_encoder_dim = 0
        ckpt = os.getenv("FORGE_TACTILE_ENCODER_CKPT", "").strip()
        if not ckpt:
            return

        rewind_root = self._rewind_root()
        if rewind_root not in sys.path:
            sys.path.insert(0, rewind_root)
        try:
            from tools.tactile_model import TactileCNNEncoder
        except Exception as e:
            print(f"[TactileEncoder] FAILED import (rewind_root={rewind_root}): {e}")
            return

        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        cfg = state.get("args", {})
        in_channels = int(cfg.get("in_channels", 3))
        per_hand_dim = int(cfg.get("per_hand_dim", 64))
        dim = 2 * per_hand_dim

        # The obs term sizes itself from the env var at manager-construction
        # time, long before this runs — a mismatch would hand the policy a
        # zero-filled column of the wrong width for the whole run.
        declared = os.getenv("FORGE_TACTILE_ENCODER_DIM", "").strip()
        if declared and int(declared) != dim:
            raise ValueError(
                f"FORGE_TACTILE_ENCODER_DIM={declared} but {ckpt} has "
                f"per_hand_dim={per_hand_dim} (latent dim {dim}). Set "
                f"FORGE_TACTILE_ENCODER_DIM={dim}."
            )

        self._tactile_encoder = TactileCNNEncoder(
            in_channels=in_channels,
            per_hand_dim=per_hand_dim,
            output_dim=dim,
            num_strided_layers=int(cfg.get("num_strided_layers", 3)),
            bimanual_axis=cfg.get("bimanual_axis", None) or "height",
        ).to(self.device)
        # The AE ckpt holds encoder.* + decoder.*; the decoder is training-only.
        prefix = "encoder."
        enc_state = {k[len(prefix):]: v for k, v in state["model_state_dict"].items()
                     if k.startswith(prefix)}
        self._tactile_encoder.load_state_dict(enc_state)
        self._tactile_encoder.eval()
        for param in self._tactile_encoder.parameters():
            param.requires_grad_(False)

        # Same fixed dataset-wide scale the AE was trained with — per-frame
        # normalization here would destroy the grip-strength information the
        # latent encodes.
        scale = float(cfg.get("global_scale", 1.0))
        self._tactile_encoder_scale = scale if scale > 0 else 1.0
        self._tactile_encoder_channels = (0, 1, 2) if in_channels == 3 else (1, 2)
        self._tactile_encoder_dim = dim
        self._tactile_encoder_enabled = True
        print(f"[TactileEncoder] enabled  ckpt={ckpt}  dim={dim}  "
              f"in_channels={in_channels}  global_scale={self._tactile_encoder_scale:.6g}")

    def compute_tactile_embedding(self):
        """(num_envs, 2*per_hand_dim) frozen AE latent, or None if disabled."""
        if not getattr(self, "_tactile_encoder_enabled", False):
            return None

        field = self._tactile_row_stacked_field()
        if field is None:
            return torch.zeros(self.num_envs, self._tactile_encoder_dim, device=self.device)

        x = field[..., list(self._tactile_encoder_channels)] / self._tactile_encoder_scale
        x = x.permute(0, 3, 1, 2).contiguous()                # (N, C, 40, 25)
        with torch.no_grad():
            z = self._tactile_encoder(x)
        return z.detach().float()

    def _init_tactile_reward(self):
        """Build the optional Tactile-ReWiND progress-reward model.

        All the machinery lives in `tactile_reward_model.TactileRewardModel`;
        this env supplies the per-step force field and owns the reward shaping.
        Activated when env var FORGE_TACTILE_REWARD_CKPT points at a .pth — see
        `TactileRewardModel.from_env` for the full list of knobs.

        FORGE_TACTILE_REWARD_SCALE is read here rather than in the model: it is
        a per-run sweep knob (the same task config is trained at different
        scales), so it cannot live in the static RewTerm weight either.
        """
        self._tactile_reward_model = None
        self._tactile_reward_scale = float(os.getenv("FORGE_TACTILE_REWARD_SCALE", "1.0"))
        try:
            TactileRewardModel = _import_tactile_reward_model()
        except ImportError as e:
            print(f"[TactileReward] disabled: {e}")
            return
        self._tactile_reward_model = TactileRewardModel.from_env(
            num_envs=self.num_envs,
            device=self.device,
            max_episode_length=int(getattr(self, "max_episode_length", 150)),
        )
        if self._tactile_reward_model is not None:
            print(f"[TactileReward] reward scale={self._tactile_reward_scale}")

    def compute_tactile_reward(self) -> torch.Tensor:
        """(num_envs,) scaled predicted progress as a dense reward bonus."""
        if self._tactile_reward_model is None:
            return torch.zeros(self.num_envs, device=self.device)

        left = self._get_tactile_vector_field("left_tactile_sensor")
        right = self._get_tactile_vector_field("right_tactile_sensor")
        if left is None or right is None:
            return torch.zeros(self.num_envs, device=self.device)

        # Stack both pads along the row dim -> (num_envs, 2*rows, cols, 3).
        frame = torch.cat([left, right], dim=1)
        progress = self._tactile_reward_model.compute(frame)
        return progress * self._tactile_reward_scale

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
             curr_successes = (success_reward > 0)
        else:
             curr_successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._save_env0_tactile_force_field()

        if torch.any(self.reset_buf):
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.pending_episode_successes[reset_env_ids] = self.ep_succeeded[reset_env_ids].long()
            self.pending_episode_successes_at_end[reset_env_ids] = curr_successes[reset_env_ids].long()
            self.env_episode_index[reset_env_ids] += 1

            if (self.pending_episode_successes >= 0).all():
                episode_success_rate = self.pending_episode_successes.float().mean()
                self.extras["episode_success_rate"] = episode_success_rate.item()
                self.max_episode_success_rate = max(self.max_episode_success_rate, episode_success_rate.item())
                self.pending_episode_successes.fill_(-1)

            if (self.pending_episode_successes_at_end >= 0).all():
                episode_success_rate_at_end = self.pending_episode_successes_at_end.float().mean()
                self.extras["episode_success_rate_at_end"] = episode_success_rate_at_end.item()
                self.pending_episode_successes_at_end.fill_(-1)
            
            # Reset ep_succeeded for next episode
            self.ep_succeeded[reset_env_ids] = False

        return obs, reward, terminated, truncated, info

    def get_env_state(self):
        """Serializable env state persisted inside the rl_games checkpoint.

        rl_games stores this dict under ``state['env_state']`` in the .pth via
        ``get_full_state_weights`` and hands it back through ``set_env_state`` on
        resume. We persist the monotonic curriculum state so the tactile-reward
        fade (see ``rewind_tactile_reward``) does not restart from full strength.
        """
        return {"max_episode_success_rate": float(self.max_episode_success_rate)}

    def set_env_state(self, env_state):
        """Restore persisted env state when resuming from a checkpoint."""
        if not env_state:
            return
        if "max_episode_success_rate" in env_state:
            self.max_episode_success_rate = float(env_state["max_episode_success_rate"])
            print(
                f"[StackTactileEnv] Restored max_episode_success_rate="
                f"{self.max_episode_success_rate:.4f} from checkpoint"
            )

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

        return obs, info

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        
        # Clear tactile reward history (and dump the progress curve) for resetting envs.
        if getattr(self, "_tactile_reward_model", None) is not None:
            self._tactile_reward_model.reset_idx(env_ids)

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
