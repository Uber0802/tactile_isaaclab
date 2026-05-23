
import os
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
        if "stack_object" in self.scene.keys():
            materials = self.scene["stack_object"].root_physx_view.get_material_properties()
            materials[..., 0] = 0.8  # static friction
            materials[..., 1] = 1.0  # dynamic friction
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
    
    def _get_tactile_vector_field(self, sensor_name: str):
        """Return the GelSight force field for a given sensor as (N, H, W, 3)."""
        if sensor_name not in self.scene.sensors:
            return None
        sensor = self.scene.sensors[sensor_name]
        
        nrows, ncols = sensor.cfg.tactile_array_size
        normal_force = sensor.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
        shear_force = sensor.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        return torch.cat((normal_force, shear_force), dim=-1)

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
