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

        # Speed escape hatch: skip GelSight sensor creation when set. Saves the
        # per-step rendering cost (~30-40% of step time) for runs that don't
        # need tactile — baseline A no-tactile-reward variants, debugging, etc.
        # Code paths that later read sensor outputs (tactile reward model,
        # baseline B/B2 obs, FORGE_SAVE_TACTILE_FORCE_FIELD) all guard on
        # `"left_tactile_sensor" in self.scene.sensors`, so skipping is safe
        # so long as those features aren't enabled in the same run.
        # The cfg has already been nulled in ForgeEnvCfg.__post_init__ for the
        # scene auto-detection; here we additionally skip the manual fallback.
        if os.getenv("FORGE_SKIP_TACTILE_SENSORS", "0") == "1":
            return

        if getattr(self.cfg, "left_tactile_sensor", None) is not None:
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

        if getattr(self.cfg, "right_tactile_sensor", None) is not None:
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

        save_cfg = self.cfg.tactile_save
        self._save_tactile_force_field = bool(save_cfg.force_field)
        # Opt-in multi-env tactile capture: every env maintains its own episode
        # buffer instead of only buffering the single hard-coded target env.
        # False keeps the legacy behavior so existing baseline-A/B/B2 runs are
        # bit-identical. Used by the curriculum-rollout scripts to multiply
        # trajectory yield by num_envs.
        self._save_tactile_all_envs = bool(save_cfg.all_envs)
        self._tactile_save_interval = max(1, int(save_cfg.save_interval))
        self._tactile_save_dir = save_cfg.save_dir or "./tactile_dataset"
        # Safety cap for total buffered frames across per-env buffers (multi-env
        # mode only). 5e5 frames * 6 KB ≈ 3 GB upper bound.
        self._tactile_max_buffer_frames = int(save_cfg.max_buffer_frames)
        self._tactile_saved_episode_count = 0
        self._tactile_step_in_episode = 0
        # Per-env episode quota. 0 = unlimited (default). When > 0, each env
        # stops appending to its buffer once it has saved this many complete
        # episodes — useful for curriculum rollout where you want exactly
        # `num_envs * quota` trajectories per ckpt regardless of iter count.
        self._tactile_episodes_per_env = int(save_cfg.episodes_per_env)
        self._tactile_saved_per_env = [0] * self.num_envs

        # If FORGE_SKIP_TACTILE_SENSORS=1 took the GelSight sensors out, tactile
        # data collection has nothing to read — silently downgrade the tactile
        # save flag so the code path below doesn't crash. Camera-only saving
        # still works via tactile_save.camera below.
        if self._save_tactile_force_field and "left_tactile_sensor" not in self.scene.sensors:
            print("[TactileSave] tactile_save.force_field=True but tactile sensors absent; "
                  "disabling tactile save (camera-only is still allowed via tactile_save.camera).")
            self._save_tactile_force_field = False

        # RGB front-camera save (independent of tactile save). Triggered when
        # the camera is in the scene AND tactile_save.camera is set (or the
        # legacy combined behavior: tactile save on + camera attached).
        camera_present = "front_cam" in self.scene.sensors and getattr(self.cfg, "enable_front_cam", False)
        self._save_front_cam = camera_present and (
            bool(save_cfg.camera) or self._save_tactile_force_field
        )

        self._save_any_trajectory = self._save_tactile_force_field or self._save_front_cam

        if self._save_any_trajectory:
            os.makedirs(self._tactile_save_dir, exist_ok=True)

        if self._save_tactile_all_envs:
            self._tactile_step_in_episode_per_env = [0] * self.num_envs
            self._tactile_episode_frames = (
                [[] for _ in range(self.num_envs)] if self._save_tactile_force_field else []
            )
            self._camera_episode_frames = (
                [[] for _ in range(self.num_envs)] if self._save_front_cam else []
            )
        else:
            self._tactile_episode_frames = []
            self._camera_episode_frames = []

        # Optional Tactile-ReWiND progress reward.
        self._init_tactile_reward()

        # Optional ReWiND visual reward (RGB → DINOv2 → ReWiNDTransformer).
        # Mirrors the tactile reward but reads the front_cam sensor. Activated
        # by cfg.visual_reward.ckpt; requires FORGE_ENABLE_FRONT_CAM=1 plus
        # --enable_cameras. Opt-in — disabled runs see no overhead.
        self._init_visual_reward()

        # Optional Tactile-ReWiND CNN encoder for Baseline B2 (frozen 768-dim
        # tactile embedding fed to the policy in place of the raw 3000-dim
        # force fields). Independent from the reward model above.
        self._init_tactile_encoder()

    def _init_tactile_reward(self):
        """Optional dense reward bonus from a Tactile-ReWiND ckpt.

        Configured through ``cfg.tactile_reward`` (a ``TactileRewardCfg``), so
        the knobs go through Hydra like the rest of the config::

            env.tactile_reward.ckpt=assets/TactileModel/gear_scratch_epoch18.pth
            env.tactile_reward.scale=0.1
            env.tactile_reward.smooth_alpha=0.2

        An empty ``ckpt`` disables the reward entirely.
        """
        self._tactile_reward_enabled = False
        self._tactile_reward_model = None

        rew_cfg = self.cfg.tactile_reward
        if not (rew_cfg.ckpt or "").strip():
            return

        from isaaclab_tasks.utils.tactile_reward_import import TactileRewardModel

        # Ckpt loading, channel selection, history subsampling, per-slice
        # normalization and EMA smoothing all live in the shared model. This env
        # keeps only the reward shaping (scale + annealing) below.
        self._tactile_reward_model = TactileRewardModel.from_cfg(
            rew_cfg,
            num_envs=self.num_envs,
            device=self.device,
            max_episode_length=int(getattr(self, "max_episode_length", 128)),
            default_instruction="grasp peg and insert to another hole",
        )
        if self._tactile_reward_model is None:
            return

        # Optional linear annealing of the tactile reward scale over training.
        # Lets the tactile bonus bootstrap early learning, then fade so the
        # policy converges on the task reward alone (helps when tactile speeds
        # up learning but caps final success — see nut A_hard_success).
        # scale(t) = start + (end-start) * clamp(t / anneal_steps, 0, 1)
        # 1 PPO iter = horizon_length control steps (nut horizon=256), so
        # annealing over 3000 iters means anneal_steps = 3000*256.
        self._tactile_reward_scale = float(rew_cfg.scale)
        self._tactile_reward_scale_start = self._tactile_reward_scale
        self._tactile_reward_anneal_steps = int(rew_cfg.anneal_steps)
        # scale_end only matters while annealing; hold at `scale` otherwise so
        # the startup banner reports the constant the policy actually sees.
        self._tactile_reward_scale_end = (
            float(rew_cfg.scale_end) if self._tactile_reward_anneal_steps > 0
            else self._tactile_reward_scale
        )
        self._tactile_anneal_step = 0
        # "linear" ramps from step 0; "success" holds the start scale until the
        # running success rate first crosses the threshold, then ramps from
        # there. The latter keeps the bonus bootstrapping until the policy is
        # just starting to solve the task, instead of fading on a fixed clock
        # that may expire before anything works.
        self._tactile_anneal_mode = (rew_cfg.anneal_mode or "linear").strip().lower()
        self._tactile_anneal_success_thresh = float(rew_cfg.anneal_success_thresh)
        self._tactile_anneal_success_ema_alpha = float(rew_cfg.anneal_success_ema_alpha)
        self._tactile_anneal_success_ema = 0.0
        self._tactile_anneal_triggered = False
        self._tactile_reward_enabled = True

        if self._tactile_reward_anneal_steps > 0:
            anneal_msg = (f"  ANNEAL[{self._tactile_anneal_mode}] "
                          f"{self._tactile_reward_scale_start}->"
                          f"{self._tactile_reward_scale_end} over "
                          f"{self._tactile_reward_anneal_steps} steps")
            if self._tactile_anneal_mode == "success":
                anneal_msg += (f" (trigger@success>="
                               f"{self._tactile_anneal_success_thresh})")
        else:
            anneal_msg = ""
        print(f"[TactileReward] scale={self._tactile_reward_scale}{anneal_msg}")

    def _init_visual_reward(self):
        """Optional dense reward bonus from a ReWiND visual model ckpt.

        Mirrors `_init_tactile_reward` but reads RGB from the `front_cam`
        sensor and runs frames through DINOv2 + ReWiNDTransformer.

        Configured through ``cfg.visual_reward`` (a ``VisualRewardCfg``)::

            env.visual_reward.ckpt=/path/to/rewind.pth
            env.visual_reward.scale=0.3
            env.visual_reward.dino_interval=4     # if DINOv2 is the bottleneck

        An empty ``ckpt`` disables the reward. Still requires
        FORGE_ENABLE_FRONT_CAM=1 + --enable_cameras for the camera to be in the
        scene: that flag is consumed in the config's __post_init__, before Hydra
        applies overrides, so it cannot move onto the config.
        """
        self._visual_reward_enabled = False
        vis_cfg = self.cfg.visual_reward
        ckpt = (vis_cfg.ckpt or "").strip()
        if not ckpt:
            return
        if "front_cam" not in self.scene.sensors:
            print("[VisualReward] front_cam not attached — set FORGE_ENABLE_FRONT_CAM=1 "
                  "and pass --enable_cameras to enable it. Visual reward DISABLED.")
            return

        # Load ReWiND's model.py by explicit path — DON'T just sys.path.insert
        # + `from model import ...`, because Tactile-ReWiND ALSO ships a
        # top-level `model.py`. If both reward heads are active the second
        # `from model import ...` resolves to whichever module was imported
        # first (cached in sys.modules), so we'd silently call the WRONG
        # transformer class. importlib.util.spec_from_file_location avoids
        # that by giving each visual model its own unique module name.
        import importlib.util
        rewind_root = os.path.expanduser(vis_cfg.root or "~/ReWiND")
        model_path = os.path.join(rewind_root, "model.py")
        try:
            spec = importlib.util.spec_from_file_location("rewind_visual_model", model_path)
            mod = importlib.util.module_from_spec(spec)
            # Make `from training.X import Y` style imports inside model.py
            # still resolve — push rewind_root onto sys.path for the local
            # module-load scope. We can't avoid this if model.py has its own
            # imports relative to its repo root.
            if rewind_root not in sys.path:
                sys.path.insert(0, rewind_root)
            spec.loader.exec_module(mod)
            ReWiNDTransformer = mod.ReWiNDTransformer
        except Exception as e:
            print(f"[VisualReward] FAILED loading ReWiNDTransformer from {model_path}: {e}")
            return

        # Load DINOv2 backbone (frozen).
        backbone_name = vis_cfg.backbone or "dinov2_vitb14"
        try:
            backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        except Exception as e:
            print(f"[VisualReward] FAILED loading backbone {backbone_name}: {e}")
            return
        backbone = backbone.to(self.device).eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self._visual_backbone = backbone

        # Load ReWiND model.
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        cfg = state.get("args", None)
        max_length = getattr(cfg, "max_length", 16) if cfg is not None else 16
        self._visual_model = ReWiNDTransformer(
            args=cfg, video_dim=768, text_dim=384, hidden_dim=512,
        ).to(self.device).eval()
        self._visual_model.load_state_dict(state["model_state_dict"])
        for p in self._visual_model.parameters():
            p.requires_grad = False
        self._visual_max_length = max_length

        # Encode instruction via MiniLM (same as tactile reward).
        instruction = vis_cfg.instruction or "pick the peg and insert it into the hole"
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        minilm = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        ).to(self.device).eval()
        with torch.no_grad():
            enc = tok([instruction], padding=True, return_tensors="pt").to(self.device)
            out = minilm(**enc)
            tok_emb = out[0]
            mask = enc["attention_mask"].unsqueeze(-1).expand(tok_emb.size()).float()
            text_emb = (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            # ReWiND training applied L2 normalize; match it.
            text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=1)
        del minilm, tok
        self._visual_text_emb = text_emb.float()

        # ImageNet normalization for DINOv2.
        self._dino_mean = torch.tensor([0.485, 0.456, 0.406],
                                        device=self.device).view(1, 3, 1, 1)
        self._dino_std = torch.tensor([0.229, 0.224, 0.225],
                                       device=self.device).view(1, 3, 1, 1)

        # Per-env feature buffer (B, H, 768).
        default_history = int(getattr(self, "max_episode_length", 150))
        self._visual_history_length = int(vis_cfg.history) or default_history
        if self._visual_history_length < max_length:
            self._visual_history_length = max_length
        self._visual_buffer = torch.zeros(
            self.num_envs, self._visual_history_length, 768,
            device=self.device, dtype=torch.float32,
        )
        self._visual_step_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long,
        )
        self._visual_reward_scale = float(vis_cfg.scale)
        # Annealing state, mirroring _init_tactile_reward's block; each head
        # keeps its own trigger/ramp state so both can anneal in the same run.
        self._visual_reward_scale_start = self._visual_reward_scale
        self._visual_reward_anneal_steps = int(vis_cfg.anneal_steps)
        self._visual_reward_scale_end = (
            float(vis_cfg.scale_end) if self._visual_reward_anneal_steps > 0
            else self._visual_reward_scale
        )
        self._visual_anneal_step = 0
        self._visual_anneal_mode = (vis_cfg.anneal_mode or "linear").strip().lower()
        self._visual_anneal_success_thresh = float(vis_cfg.anneal_success_thresh)
        self._visual_anneal_success_ema_alpha = float(vis_cfg.anneal_success_ema_alpha)
        self._visual_anneal_success_ema = 0.0
        self._visual_anneal_triggered = False
        self._visual_reward_smooth_alpha = float(vis_cfg.smooth_alpha)
        self._visual_smoothed_progress = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32,
        )
        self._visual_dino_interval = max(1, int(vis_cfg.dino_interval))
        self._visual_dino_step_counter = 0
        self._visual_last_features = torch.zeros(
            self.num_envs, 768, device=self.device, dtype=torch.float32,
        )

        self._visual_reward_enabled = True
        if self._visual_reward_anneal_steps > 0:
            anneal_msg = (f"  ANNEAL[{self._visual_anneal_mode}] "
                          f"{self._visual_reward_scale_start}->"
                          f"{self._visual_reward_scale_end} over "
                          f"{self._visual_reward_anneal_steps} steps")
            if self._visual_anneal_mode == "success":
                anneal_msg += (f" (trigger@success>="
                               f"{self._visual_anneal_success_thresh})")
        else:
            anneal_msg = ""
        print(f"[VisualReward] enabled  ckpt={ckpt}  scale={self._visual_reward_scale}{anneal_msg}  "
              f"backbone={backbone_name}  instruction={instruction!r}  "
              f"history={self._visual_history_length}  max_length={max_length}  "
              f"smooth_alpha={self._visual_reward_smooth_alpha}  "
              f"dino_interval={self._visual_dino_interval}")

    def _compute_visual_reward(self) -> torch.Tensor:
        """(num_envs,) predicted visual progress as a dense reward bonus."""
        if not getattr(self, "_visual_reward_enabled", False):
            return torch.zeros(self.num_envs, device=self.device)

        # Get RGB (B, 224, 224, 3) uint8 from front_cam.
        rgb_raw = self.scene.sensors["front_cam"].data.output["rgb"][..., :3]

        # Run DINOv2 only every `dino_interval` steps to amortise cost.
        self._visual_dino_step_counter += 1
        if self._visual_dino_step_counter % self._visual_dino_interval == 0:
            with torch.no_grad():
                rgb = rgb_raw.permute(0, 3, 1, 2).float() / 255.0
                rgb = (rgb - self._dino_mean) / self._dino_std
                features = self._visual_backbone(rgb)        # (B, 768)
            self._visual_last_features = features
        else:
            features = self._visual_last_features

        # Push to rolling buffer.
        H = self._visual_history_length
        T = self._visual_max_length
        self._visual_buffer = torch.roll(self._visual_buffer, shifts=-1, dims=1)
        self._visual_buffer[:, -1] = features
        self._visual_step_count = torch.clamp(self._visual_step_count + 1, max=H)
        valid = self._visual_step_count.clamp(min=1)
        start = (H - valid).long()

        # Sample T frames per env (linspace stride, same rule as tactile).
        device = self._visual_buffer.device
        t_grid = torch.arange(T, device=device)
        if T > 1:
            frac = t_grid.float() / float(T - 1)
        else:
            frac = torch.zeros(T, device=device)
        span = (H - 1 - start).float().unsqueeze(1)
        long_idx = (start.float().unsqueeze(1)
                    + span * frac.unsqueeze(0)).round().long()
        long_idx.clamp_(0, H - 1)
        short_idx = (start.unsqueeze(1)
                     + torch.minimum(t_grid.unsqueeze(0),
                                     (valid - 1).unsqueeze(1)))
        short_idx.clamp_(0, H - 1)
        is_long = (valid >= T).unsqueeze(1)
        sel = torch.where(is_long, long_idx, short_idx)
        gather_idx = sel[:, :, None].expand(-1, -1, 768)
        slc = torch.gather(self._visual_buffer, 1, gather_idx)   # (B, T, 768)

        text = self._visual_text_emb.expand(self.num_envs, -1)
        with torch.no_grad():
            progress = self._visual_model(slc, text)
            if progress.ndim == 3:
                progress = progress.squeeze(-1)
        latest = progress[:, -1]

        alpha = self._visual_reward_smooth_alpha
        if alpha < 1.0:
            self._visual_smoothed_progress = (
                alpha * latest + (1.0 - alpha) * self._visual_smoothed_progress
            )
            out = self._visual_smoothed_progress
        else:
            self._visual_smoothed_progress = latest
            out = latest

        # Anneal the scale (no-op when anneal_steps <= 0 or end == start).
        # Same schedule as compute_tactile_reward's block, on independent state.
        if self._visual_reward_anneal_steps > 0:
            ramping = True
            if self._visual_anneal_mode == "success" and not self._visual_anneal_triggered:
                if self._visual_anneal_success_ema >= self._visual_anneal_success_thresh:
                    self._visual_anneal_triggered = True
                    print(f"[VisualReward] success anneal TRIGGERED "
                          f"(success_ema={self._visual_anneal_success_ema:.4f} >= "
                          f"{self._visual_anneal_success_thresh}); decaying "
                          f"{self._visual_reward_scale_start}->"
                          f"{self._visual_reward_scale_end} over "
                          f"{self._visual_reward_anneal_steps} steps")
                else:
                    ramping = False
            if ramping:
                frac = min(1.0, self._visual_anneal_step / self._visual_reward_anneal_steps)
                self._visual_reward_scale = (
                    self._visual_reward_scale_start
                    + (self._visual_reward_scale_end - self._visual_reward_scale_start) * frac
                )
                self._visual_anneal_step += 1
        return out * self._visual_reward_scale

    def _compute_tactile_reward(self) -> torch.Tensor:
        """(num_envs,) predicted progress as a dense reward bonus.

        The rolling history, training-matched stride subsampling, per-slice
        normalization and EMA smoothing live in `TactileRewardModel`; this
        method only feeds it the current force field and applies the reward
        shaping (annealed scale) that is specific to this env.
        """
        if not getattr(self, "_tactile_reward_enabled", False):
            return torch.zeros(self.num_envs, device=self.device)

        left = self.scene.sensors["left_tactile_sensor"]
        right = self.scene.sensors["right_tactile_sensor"]
        nrows, ncols = left.cfg.tactile_array_size           # (20, 25)
        # Build the full (B, 40, 25, 3) tensor in the (normal, shear_x, shear_y)
        # layout that `get_left_tactile_vector_field` writes to disk. The model
        # applies the ckpt's own channel selection to it.
        l_shear = left.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        r_shear = right.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        l_normal = left.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
        r_normal = right.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
        l_full = torch.cat([l_normal, l_shear], dim=-1)          # (B, 20, 25, 3)
        r_full = torch.cat([r_normal, r_shear], dim=-1)
        frame = torch.cat([l_full, r_full], dim=1).float()       # (B, 40, 25, 3)

        out = self._tactile_reward_model.compute(frame)          # (B,) in [0, 1]

        # Anneal the scale (no-op when anneal_steps <= 0 or end == start).
        if self._tactile_reward_anneal_steps > 0:
            # In "success" mode the ramp is frozen until the running success rate
            # crosses the trigger; once tripped it behaves like the linear ramp,
            # counting from the moment it fired.
            ramping = True
            if self._tactile_anneal_mode == "success" and not self._tactile_anneal_triggered:
                if self._tactile_anneal_success_ema >= self._tactile_anneal_success_thresh:
                    self._tactile_anneal_triggered = True
                    print(f"[TactileReward] success anneal TRIGGERED "
                          f"(success_ema={self._tactile_anneal_success_ema:.4f} >= "
                          f"{self._tactile_anneal_success_thresh}); decaying "
                          f"{self._tactile_reward_scale_start}->"
                          f"{self._tactile_reward_scale_end} over "
                          f"{self._tactile_reward_anneal_steps} steps")
                else:
                    ramping = False
            if ramping:
                frac = min(1.0, self._tactile_anneal_step / self._tactile_reward_anneal_steps)
                self._tactile_reward_scale = (
                    self._tactile_reward_scale_start
                    + (self._tactile_reward_scale_end - self._tactile_reward_scale_start) * frac
                )
                self._tactile_anneal_step += 1
        return out * self._tactile_reward_scale

    @staticmethod
    def _encoder_rewind_root(configured: str = "") -> str:
        """Directory holding ``tools/tactile_model.py``.

        The configured path wins, but the repo-relative fallback is correct by
        construction — the historical default (``~/tactile_isaaclab/...``)
        silently misses on checkouts that live anywhere else.
        """
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
        candidates = [
            (configured or "").strip(),
            os.path.join(repo_root, "external", "third-party", "Tactile-ReWiND"),
        ]
        for cand in candidates:
            if cand and os.path.isfile(
                os.path.join(os.path.expanduser(cand), "tools", "tactile_model.py")
            ):
                return os.path.expanduser(cand)
        return os.path.expanduser(candidates[-1])

    def _init_tactile_encoder(self):
        """Optional frozen tactile embedding appended to the obs/state vectors.

        Configured through ``cfg.tactile_encoder`` (a ``TactileEncoderCfg``)::

            env.tactile_encoder.ckpt=/path/to/model.pth env.tactile_encoder.dim=32

        Loads only the ``TactileCNNEncoder`` submodule (``encoder.*``), freezes
        it, and runs it once per env-step on the current force-field frame. Two
        ckpt families are supported, distinguished by the ``args`` they carry:

          * ReWiND progress ckpts (baseline B2): no ``in_channels`` recorded, so
            the default reproduces the original behavior — shear-only
            ``(B, 2, 40, 25)`` at raw scale.
          * AE ckpts from ``train_tactile_ae.py`` (baseline ``tactile_state``):
            record ``in_channels`` and ``global_scale``. A 3-channel ckpt
            consumes ``(normal, shear_x, shear_y)`` in the same layout the AE was
            trained on, divided by the ckpt's fixed dataset-wide scale.

        The AE latent is trained for reconstruction only, so unlike B2's
        progress-trained encoder it carries no task/reward information — which is
        what makes it a clean "tactile as state" ablation against "tactile as
        reward" (TacReward).

        An empty ``ckpt`` disables it.
        """
        self._tactile_encoder_enabled = False
        enc_cfg = self.cfg.tactile_encoder
        ckpt = (enc_cfg.ckpt or "").strip()
        if not ckpt:
            return

        rewind_root = self._encoder_rewind_root(enc_cfg.root)
        if rewind_root not in sys.path:
            sys.path.insert(0, rewind_root)
        try:
            from tools.tactile_model import TactileCNNEncoder
        except Exception as e:
            print(f"[TactileEncoder] FAILED import (rewind_root={rewind_root}): {e}")
            return

        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        cfg = state.get("args", {})
        num_strided = cfg.get("num_strided_layers", None) or 3
        bimanual_axis = cfg.get("bimanual_axis", None) or "height"
        per_hand_dim = cfg.get("per_hand_dim", 384)
        output_dim = 2 * per_hand_dim   # matches TactileReWiNDTransformer.video_dim
        # AE ckpts record these; ReWiND reward ckpts predate them, and the
        # defaults reproduce the original B2 behavior (shear-only, raw scale).
        in_channels = int(cfg.get("in_channels", None) or 2)
        global_scale = float(cfg.get("global_scale", 0.0) or 0.0)

        # The env cfg sized the obs/state vectors from ``dim`` before this ckpt
        # was read; a mismatch would hand the policy a column of the wrong width
        # for the entire run. Fail here rather than deep inside obs assembly.
        declared = int(enc_cfg.dim)
        if declared and declared != output_dim:
            raise ValueError(
                f"[TactileEncoder] env.tactile_encoder.dim={declared} but {ckpt} has "
                f"per_hand_dim={per_hand_dim} (latent dim {output_dim}). Set "
                f"env.tactile_encoder.dim={output_dim}, or retrain the AE with "
                f"--per_hand_dim {declared // 2}."
            )

        encoder = TactileCNNEncoder(
            in_channels=in_channels,       # 2 = shear-only (B2), 3 = normal+shear (AE)
            per_hand_dim=per_hand_dim,
            output_dim=output_dim,
            num_strided_layers=num_strided,
            bimanual_axis=bimanual_axis,
        ).to(self.device)

        # Filter the full ckpt state_dict down to just the encoder submodule.
        full_sd = state["model_state_dict"]
        enc_sd = {
            k[len("encoder."):]: v for k, v in full_sd.items()
            if k.startswith("encoder.")
        }
        missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
        if missing or unexpected:
            print(f"[TactileEncoder] load_state_dict missing={missing} unexpected={unexpected}")
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

        self._tactile_encoder = encoder
        self._tactile_encoder_dim = output_dim
        self._tactile_encoder_in_channels = in_channels
        # Same fixed dataset-wide scale the AE was trained with. Per-frame
        # normalization here would destroy the grip-strength information the
        # latent encodes, so this is a constant divisor, not a running max.
        self._tactile_encoder_scale = global_scale if global_scale > 0 else None
        self._tactile_encoder_enabled = True
        print(f"[TactileEncoder] enabled  ckpt={ckpt}  out_dim={output_dim}  "
              f"in_ch={in_channels}  global_scale={self._tactile_encoder_scale}  "
              f"axis={bimanual_axis}  strided={num_strided}")

    def _compute_tactile_embedding(self) -> torch.Tensor:
        """(num_envs, D) per-step embedding of the current force-field frame.

        Both hands stacked on the row axis, then passed through the frozen CNN:

          * 2-channel ckpts (B2 / ReWiND): shear-only -> ``(B, 2, 40, 25)`` at
            raw scale, mirroring the ReWiND training layout.
          * 3-channel AE ckpts: ``(normal, shear_x, shear_y)`` ->
            ``(B, 3, 40, 25)``, matching the layout ``train_tactile_ae.py`` was
            trained on, divided by the ckpt's fixed ``global_scale``.

        Returns zeros if the encoder is not enabled (so callers can populate
        the obs dict unconditionally).
        """
        if not getattr(self, "_tactile_encoder_enabled", False):
            return torch.zeros(
                self.num_envs,
                getattr(self, "_tactile_encoder_dim", 768),
                device=self.device,
            )
        left = self.scene.sensors["left_tactile_sensor"]
        right = self.scene.sensors["right_tactile_sensor"]
        nrows, ncols = left.cfg.tactile_array_size           # (20, 25)
        l_shear = left.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        r_shear = right.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
        if getattr(self, "_tactile_encoder_in_channels", 2) == 3:
            l_normal = left.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
            r_normal = right.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
            l_frame = torch.cat([l_normal, l_shear], dim=-1)     # (B, 20, 25, 3)
            r_frame = torch.cat([r_normal, r_shear], dim=-1)
        else:
            l_frame, r_frame = l_shear, r_shear
        # (B, 40, 25, C) -> (B, C, 40, 25) for the CNN.
        frame = torch.cat([l_frame, r_frame], dim=1).float().permute(0, 3, 1, 2).contiguous()
        scale = getattr(self, "_tactile_encoder_scale", None)
        if scale:
            frame = frame / scale
        with torch.no_grad():
            return self._tactile_encoder(frame)              # (B, D)

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

    def _flush_tactile_episode(self, success: int = 0, env_id: int | None = None):
        """Write the buffered tactile tensors for one completed episode.

        Saved file is a dict (np.save with allow_pickle=True):
            {"Task":     <fixed task description>,
             "Tactile":  np.ndarray (T, H, W, C) float16,
             "Success":  int 0 / 1}
        Load with `np.load(path, allow_pickle=True).item()`.

        env_id=None  → legacy single-target-env mode; `_tactile_episode_frames`
                       is a flat list.
        env_id=int   → multi-env mode; flush `_tactile_episode_frames[env_id]`.
        """
        if not self._save_any_trajectory:
            return

        tactile_frames = (
            self._tactile_episode_frames if env_id is None
            else self._tactile_episode_frames[env_id]
        ) if self._save_tactile_force_field else None
        cam_frames = (
            self._camera_episode_frames if env_id is None
            else self._camera_episode_frames[env_id]
        ) if self._save_front_cam else None

        # Bail if both buffers are empty — nothing to write for this episode.
        if not tactile_frames and not cam_frames:
            return

        # Legacy single-env path keeps the plain `ep{N}.npy` name; multi-env
        # path appends the source env id so downstream tooling can group / dedupe.
        if env_id is None:
            base_fname = f"ep{self._tactile_saved_episode_count}"
        else:
            base_fname = f"ep{self._tactile_saved_episode_count}_env{env_id:03d}"

        if tactile_frames:
            episode_path = os.path.join(self._tactile_save_dir, f"{base_fname}.npy")
            episode_tensor = np.stack(tactile_frames, axis=0).astype(np.float16, copy=False)
            payload = {
                "Task": "grasp peg and insert to another hole",
                "Tactile": episode_tensor,
                "Success": int(success),
            }
            np.save(episode_path, payload, allow_pickle=True)
            tactile_frames.clear()

        if cam_frames:
            cam_path = os.path.join(self._tactile_save_dir, f"{base_fname}_camera.npy")
            cam_tensor = np.stack(cam_frames, axis=0).astype(np.uint8, copy=False)
            cam_payload = {
                "Camera": cam_tensor,
                "Success": int(success),
            }
            np.save(cam_path, cam_payload, allow_pickle=True)
            cam_frames.clear()

        self._tactile_saved_episode_count += 1
        if env_id is not None:
            self._tactile_saved_per_env[env_id] += 1

    def _save_env0_tactile_force_field(self):
        """Buffer target-env tactile / camera tensors and flush one .npy per episode."""
        if not self._save_any_trajectory:
            return

        if self._save_tactile_all_envs:
            self._save_all_envs_tactile_force_field()
            return

        target_env_id = min(71, self.num_envs - 1)

        # Detect episode boundary: target env just reset this step.
        if self.reset_buf[target_env_id]:
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

        # Tactile branch — only when sensors are present AND tactile save is on.
        if self._save_tactile_force_field and "left_tactile_sensor" in self.scene.sensors:
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

        # Camera branch — runs independently of tactile.
        if self._save_front_cam:
            cam_frame = (
                self.scene.sensors["front_cam"].data.output["rgb"][target_env_id, ..., :3]
                .detach().cpu().numpy().astype(np.uint8, copy=False)
            )
            self._camera_episode_frames.append(cam_frame)

        self._tactile_step_in_episode += 1

    def _save_all_envs_tactile_force_field(self):
        """Multi-env variant: every env keeps its own per-episode buffer.

        On each call:
          1. Flush the buffer of any env that just hit reset (one .npy per ep).
          2. Honour `FORGE_TACTILE_MAX_BUFFER_FRAMES` as a global safety cap
             (sum of all per-env buffers) — if exceeded, pause appends until
             flushes free space.
          3. Do one GPU→CPU transfer for the whole batch, then per-env append.
        """
        # 1) Episode-boundary flushes for any env that reset this step.
        quota = self._tactile_episodes_per_env
        reset_envs = torch.nonzero(self.reset_buf, as_tuple=False).flatten().tolist()
        for env_id in reset_envs:
            if quota > 0 and self._tactile_saved_per_env[env_id] >= quota:
                # Quota already met for this env — discard partial buffer
                # without writing a file. Subsequent appends are also blocked
                # in step 4 so disk usage stays at exactly quota * num_envs.
                if self._save_tactile_force_field:
                    self._tactile_episode_frames[env_id].clear()
                if self._save_front_cam:
                    self._camera_episode_frames[env_id].clear()
            else:
                success = (
                    int(self.ep_succeeded[env_id].item())
                    if hasattr(self, "ep_succeeded") else 0
                )
                self._flush_tactile_episode(success=success, env_id=env_id)
            self._tactile_step_in_episode_per_env[env_id] = 0

        # Early exit when every env has met its quota — saves DINOv2 / sensor
        # work for the remaining iterations of this rollout.
        if quota > 0 and all(c >= quota for c in self._tactile_saved_per_env):
            return

        # 2) Memory safety: total frames across both tactile + camera buffers.
        tactile_buf_len = (
            sum(len(buf) for buf in self._tactile_episode_frames)
            if self._save_tactile_force_field else 0
        )
        camera_buf_len = (
            sum(len(buf) for buf in self._camera_episode_frames)
            if self._save_front_cam else 0
        )
        total_frames = tactile_buf_len + camera_buf_len
        if total_frames >= self._tactile_max_buffer_frames:
            if not getattr(self, "_tactile_overflow_warned", False):
                print(
                    f"[TactileSave] WARNING: per-env buffers hold {total_frames} "
                    f"frames (cap {self._tactile_max_buffer_frames}); pausing "
                    f"appends until next flush. Raise FORGE_TACTILE_MAX_BUFFER_FRAMES "
                    f"or increase save_interval if this is hot."
                )
                self._tactile_overflow_warned = True
            for env_id in range(self.num_envs):
                self._tactile_step_in_episode_per_env[env_id] += 1
            return
        self._tactile_overflow_warned = False

        # 3) Batch GPU→CPU transfers (tactile is gated by sensor presence,
        # camera is independent — either may be active alone).
        all_frames = None
        cam_all = None

        if self._save_tactile_force_field and "left_tactile_sensor" in self.scene.sensors:
            left_sensor = self.scene.sensors["left_tactile_sensor"]
            right_sensor = self.scene.sensors["right_tactile_sensor"]
            nrows, ncols = left_sensor.cfg.tactile_array_size
            left_normal = left_sensor.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
            left_shear = left_sensor.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
            right_normal = right_sensor.data.tactile_normal_force.view(self.num_envs, nrows, ncols, 1)
            right_shear = right_sensor.data.tactile_shear_force.view(self.num_envs, nrows, ncols, 2)
            left_full = torch.cat([left_normal, left_shear], dim=-1)
            right_full = torch.cat([right_normal, right_shear], dim=-1)
            all_frames = (
                torch.cat([left_full, right_full], dim=1)
                .detach().cpu().numpy().astype(np.float16, copy=False)
            )

        if self._save_front_cam:
            cam_all = (
                self.scene.sensors["front_cam"].data.output["rgb"][..., :3]
                .detach().cpu().numpy().astype(np.uint8, copy=False)
            )

        # 4) Per-env append, respecting save_interval against this env's
        #    step-in-episode counter. Envs that already hit their per-env quota
        #    skip appending (no point buffering frames that won't be saved).
        for env_id in range(self.num_envs):
            if quota > 0 and self._tactile_saved_per_env[env_id] >= quota:
                continue
            step = self._tactile_step_in_episode_per_env[env_id]
            if step % self._tactile_save_interval == 0:
                if all_frames is not None:
                    self._tactile_episode_frames[env_id].append(all_frames[env_id])
                if cam_all is not None:
                    self._camera_episode_frames[env_id].append(cam_all[env_id])
            self._tactile_step_in_episode_per_env[env_id] = step + 1

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
        # Trajectory save (tactile and/or camera) runs once per step, before
        # the optional tactile-obs branch — works even when sensors are absent.
        self._save_env0_tactile_force_field()
        if "left_tactile_sensor" in self.scene.sensors:
            left_normal_force, left_shear_force = self._get_tactile_force_tensors("left_tactile_sensor")
            right_normal_force, right_shear_force = self._get_tactile_force_tensors("right_tactile_sensor")
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
                # Absolute positions of bolt (fixed_asset) and nut (held_asset).
                # Must be in obs_dict (not just state_dict) so tasks that include
                # them in `obs_order` — e.g. NutThread baseline — can see them.
                "fixed_pos": self.fixed_pos,
                "held_pos": self.held_pos,
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
        if getattr(self, "_visual_reward_enabled", False):
            rew_dict["visual_progress"] = self._compute_visual_reward()
            # `_visual_reward_scale` already baked in inside the helper.
            rew_scales["visual_progress"] = 1.0
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
        # Clear tactile reward history + step counter for envs that just reset.
        if getattr(self, "_tactile_reward_enabled", False):
            # Clears the rolling history and the EMA state (so a fresh episode
            # doesn't inherit the previous episode's tail value), and dumps the
            # target env's progress curve when it is in this reset batch.
            self._tactile_reward_model.reset_idx(env_ids)

        if getattr(self, "_visual_reward_enabled", False):
            self._visual_buffer[env_ids] = 0
            self._visual_step_count[env_ids] = 0
            self._visual_smoothed_progress[env_ids] = 0
            self._visual_last_features[env_ids] = 0

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

            # Feed the success-triggered tactile annealer a running estimate of
            # the episode success rate (EMA over each reset batch's mean). Read
            # every step by the ramp, so it cannot wait for the all-envs log below.
            if getattr(self, "_tactile_reward_enabled", False):
                batch_rate = self.ep_succeeded[reset_env_ids].float().mean().item()
                a = self._tactile_anneal_success_ema_alpha
                self._tactile_anneal_success_ema = (
                    a * batch_rate + (1.0 - a) * self._tactile_anneal_success_ema
                )

            # Same feed for the visual head's annealer (independent EMA state,
            # so both reward heads can anneal in the same run).
            if getattr(self, "_visual_reward_enabled", False):
                batch_rate = self.ep_succeeded[reset_env_ids].float().mean().item()
                a = self._visual_anneal_success_ema_alpha
                self._visual_anneal_success_ema = (
                    a * batch_rate + (1.0 - a) * self._visual_anneal_success_ema
                )

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
        """Flush any buffered tactile / camera episode before tearing down."""
        if self._save_any_trajectory:
            if self._save_tactile_all_envs:
                # Multi-env mode: complete episodes were already flushed at
                # each env's reset boundary. The remaining per-env buffers hold
                # *partial* episodes that didn't end before the rollout cap —
                # writing them at shutdown produces truncated trajectories that
                # pollute the dataset (and worse, lumped together via np.stack
                # they easily exceed pickle's 4 GiB protocol-3 limit). Drop them.
                pass
            else:
                success = (
                    int(self.ep_succeeded[0].item()) if hasattr(self, "ep_succeeded") else 0
                )
                self._flush_tactile_episode(success=success)
        super().close()
