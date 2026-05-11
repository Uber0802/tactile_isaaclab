
import os
import torch
import numpy as np
from typing import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg

class StackTactileEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: StackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Success tracking
        self.ep_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pending_episode_successes = torch.ones(self.num_envs, dtype=torch.long, device=self.device) * -1
        self.env_episode_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Tactile saving settings (mirrored from ForgeEnv)
        self._save_tactile_force_field = os.environ.get("FORGE_SAVE_TACTILE_FORCE_FIELD", "0") == "1"
        self._tactile_save_dir = os.environ.get("FORGE_TACTILE_SAVE_DIR", "./tactile_dataset/data")
        self._tactile_save_interval = int(os.environ.get("FORGE_TACTILE_SAVE_INTERVAL", "1"))
        
        if self._save_tactile_force_field:
            os.makedirs(self._tactile_save_dir, exist_ok=True)
            self._tactile_episode_frames = []
            self._tactile_saved_episode_count = 0
            self._tactile_step_in_episode = 0
            print(f"[StackTactileEnv] Saving tactile force field to: {self._tactile_save_dir}")
    
    def _get_tactile_vector_field(self, sensor_name: str):
        """Return the GelSight force field for a given sensor as (N, H, W, 3)."""
        if sensor_name not in self.scene.sensors:
            return None
        sensor = self.scene.sensors[sensor_name]
        
        nrows, ncols = sensor.cfg.tactile_array_size
        normal_force = sensor.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
        shear_force = sensor.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        return torch.cat((normal_force, shear_force), dim=-1)

    def _flush_tactile_episode(self, success: int = 0):
        """Write the buffered target-env tactile tensors for the current episode."""
        if not self._save_tactile_force_field or not self._tactile_episode_frames:
            return

        episode_path = os.path.join(self._tactile_save_dir, f"ep{self._tactile_saved_episode_count}.npy")
        episode_tensor = np.stack(self._tactile_episode_frames, axis=0).astype(np.float16, copy=False)
        payload = {
            "Task": "stack cube_2 on cube_1",
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
        
        self._tactile_step_in_episode += 1

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
        if self.scene['left_tactile_sensor']._nominal_tactile is None:  # Only on first reset
            self.scene['left_tactile_sensor'].get_initial_render()
            self.scene['right_tactile_sensor'].get_initial_render()
        
        self.ep_succeeded[env_ids] = False
        return obs, info

    def close(self):
        self._flush_tactile_episode()
        super().close()
