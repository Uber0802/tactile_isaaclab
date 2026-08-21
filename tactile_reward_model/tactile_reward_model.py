"""Standalone Tactile-ReWiND progress-reward model.

This module owns everything about the tactile reward: checkpoint loading, the
instruction embedding, the per-env rolling force-field history, the model
forward pass, EMA smoothing and progress-curve logging. It knows nothing about
IsaacLab environments — the caller feeds it a per-step tactile frame and gets a
reward tensor back.

Typical usage from an env::

    self._tactile_reward_model = TactileRewardModel.from_cfg(
        self.cfg.tactile_reward,
        num_envs=self.num_envs,
        device=self.device,
        max_episode_length=self.max_episode_length,
    )

    # every step, frame is (num_envs, rows, cols, 3) = (normal, shear_x, shear_y)
    reward = self._tactile_reward_model.compute(frame)

    # on reset
    self._tactile_reward_model.reset_idx(env_ids)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

__all__ = ["TactileRewardCfg", "TactileRewardModel"]

# Library convention: emit records, never configure handlers or levels — that is
# the host application's job. Messages at WARNING and above still reach stderr
# via logging.lastResort even when nothing configures logging at all; the
# startup banner is INFO and therefore quiet by default.
logger = logging.getLogger(__name__)

# This module lives at <repo_root>/tactile_reward_model/, so the vendored
# Tactile-ReWiND checkout is resolved relative to it rather than hardcoded to a
# particular home directory. Override with the config's `rewind_root`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REWIND_ROOT = str(_REPO_ROOT / "external" / "third-party" / "Tactile-ReWiND")
_DEFAULT_INSTRUCTION = "stack an object on a box"


@dataclass
class TactileRewardCfg:
    """Knobs for the dense tactile progress reward. Empty ``ckpt`` = disabled.

    A plain stdlib dataclass, deliberately not an IsaacLab ``@configclass``, so
    this package stays importable without IsaacLab. It still nests inside a
    ``@configclass`` env config: ``configclass`` wraps it in a
    ``default_factory`` (so instances don't share state), ``class_to_dict``
    recurses into any object with a ``__dict__``, and ``update_class_from_dict``
    applies overrides via ``setattr``. That makes
    ``env.tactile_reward.scale=0.1`` work on the CLI like any other field.

    Every field defaults to a value of its own declared type — never ``None``.
    IsaacLab's ``update_class_from_dict`` type-checks an override against
    ``type(current_value)``, so a ``None`` default makes the field reject every
    override with "Expected: <class 'NoneType'>". Empty string / 0 are the
    "unset" sentinels instead.
    """

    ckpt: str = ""
    """Path to the Tactile-ReWiND ``.pth``. Empty disables the reward entirely."""

    scale: float = 1.0
    """Multiplier on the predicted progress. Reward shaping, applied by the env."""

    scale_end: float = 0.0
    """Target scale for linear annealing. Only read when ``anneal_steps > 0``."""

    anneal_steps: int = 0
    """Env control-steps to ramp ``scale`` -> ``scale_end``. 0 disables annealing."""

    instruction: str = ""
    """Task string encoded by MiniLM. Empty keeps the env's own default wording."""

    history: int = 0
    """Rolling-buffer length. 0 = the episode length."""

    smooth_alpha: float = 1.0
    """EMA coefficient on the predicted progress. 1.0 disables smoothing."""

    rewind_root: str = ""
    """Path to the Tactile-ReWiND checkout. Empty = the vendored copy."""

    log_env: int = 0
    """Index of the env whose per-episode progress curve is plotted."""

    curve_log_dir: str = ""
    """Directory for progress-curve PNGs. Empty = a timestamped default."""


class TactileRewardModel:
    """Task progress in ``[0, 1]`` predicted by a Tactile-ReWiND transformer.

    Returns raw progress: reward shaping (scaling, curriculum fades, clipping)
    is the caller's job, so the same predictor can be reused across tasks and
    reward formulations without carrying RL-specific knobs.

    Args:
        ckpt_path: Path to the Tactile-ReWiND ``.pth`` checkpoint.
        num_envs: Number of parallel environments (leading batch dim).
        device: Torch device the model and buffers live on.
        instruction: Natural-language task string encoded once by MiniLM.
        history_length: Rolling-buffer length; the slice fed to the model is
            linspace-subsampled down to the checkpoint's ``max_length`` —
            matching training's ``_sample_forward + _resize`` stride behavior.
            Defaults to ``max_episode_length`` so a slice ending at episode-end
            approximates the ``start=0, end=N`` training case (progress -> 1).
        max_episode_length: Fallback for ``history_length``.
        smooth_alpha: EMA coefficient; ``1.0`` disables smoothing.
        rewind_root: Path to the Tactile-ReWiND repo (added to ``sys.path``).
        curve_log_dir: Directory for per-episode progress-curve PNGs.
            ``None`` disables curve logging.
        log_env: Index of the env whose progress curve is logged.
    """

    def __init__(
        self,
        ckpt_path: str,
        num_envs: int,
        device: torch.device | str,
        instruction: str = _DEFAULT_INSTRUCTION,
        history_length: int | None = None,
        max_episode_length: int = 150,
        smooth_alpha: float = 1.0,
        rewind_root: str | None = None,
        curve_log_dir: str | None = None,
        log_env: int = 0,
    ):
        self.ckpt_path = ckpt_path
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.instruction = instruction
        self.smooth_alpha = float(smooth_alpha)
        self.log_env = int(log_env)

        model_cls = self._import_model_class(rewind_root)
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = state.get("args", {})

        cfg_shear = cfg.get("shear_channels", None)
        if cfg_shear:
            self.shear_channels = tuple(cfg_shear)
        else:
            ic = int(cfg.get("in_channels", 2))
            self.shear_channels = (0, 1, 2) if ic == 3 else (1, 2)
        self.in_channels = len(self.shear_channels)

        self.max_length = cfg.get("max_length", 16)
        # The encoder halves the frame along this axis to get the two pads, so
        # it constrains the layout the caller must hand us. See TactileCNNEncoder.
        self.bimanual_axis = cfg.get("bimanual_axis", None) or "height"
        self.model = model_cls(
            max_length=self.max_length,
            text_dim=384,
            hidden_dim=cfg.get("hidden_dim", 512),
            num_heads=cfg.get("num_heads", 8),
            num_layers=cfg.get("num_layers", 4),
            per_hand_dim=cfg.get("per_hand_dim", 384),
            num_strided_layers=cfg.get("num_strided_layers", None) or 3,
            bimanual_axis=self.bimanual_axis,
            in_channels=self.in_channels,
        ).to(self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

        norm_mode = cfg.get("normalize_mode", None)
        if norm_mode is None:
            norm_mode = "per_channel" if cfg.get("normalize_per_channel") else "off"
        self.normalize_mode = norm_mode

        self.text_emb = self._encode_instruction(instruction)  # (1, 384)

        # Rolling history buffer is allocated lazily on the first frame, since
        # the sensor resolution is only known then.
        self.history_length = int(history_length or max_episode_length)
        self.history_length = max(self.history_length, self.max_length)
        self._buffer: torch.Tensor | None = None
        self._step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._smoothed_progress = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self.curve_log_dir = curve_log_dir
        self._progress_history: list[tuple[float, float]] = []
        self._curve_episode_idx = 0
        if self.curve_log_dir:
            try:
                os.makedirs(self.curve_log_dir, exist_ok=True)
                curve_log_msg = f"  curve_log={self.curve_log_dir}"
            except Exception:
                self.curve_log_dir = None
                curve_log_msg = "  curve_log=DISABLED"
        else:
            curve_log_msg = "  curve_log=DISABLED"

        logger.info(
            "[TactileReward] enabled  ckpt=%s  instruction=%r  history=%s  "
            "normalize=%s  smooth_alpha=%s%s",
            ckpt_path, instruction, self.history_length,
            self.normalize_mode, self.smooth_alpha, curve_log_msg,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_cfg(
        cls,
        cfg: TactileRewardCfg,
        num_envs: int,
        device: torch.device | str,
        max_episode_length: int = 150,
        default_instruction: str = _DEFAULT_INSTRUCTION,
    ) -> "TactileRewardModel | None":
        """Build from a :class:`TactileRewardCfg`, or ``None`` if ``ckpt`` is empty.

        The shaping fields on that config (``scale``, ``scale_end``,
        ``anneal_steps``) are deliberately ignored here: the caller applies them
        to the progress this model returns.
        """
        ckpt = (cfg.ckpt or "").strip()
        if not ckpt:
            return None

        default_curve_dir = os.path.expanduser(
            f"~/tactile_isaaclab/logs/tactile_curves/{int(time.time())}"
        )
        # The config uses empty-string / 0 rather than None as its "unset"
        # sentinels (see TactileRewardCfg), so normalize them here.
        try:
            return cls(
                ckpt_path=ckpt,
                num_envs=num_envs,
                device=device,
                instruction=cfg.instruction or default_instruction,
                history_length=int(cfg.history) or None,
                max_episode_length=max_episode_length,
                smooth_alpha=float(cfg.smooth_alpha),
                rewind_root=cfg.rewind_root or None,
                curve_log_dir=cfg.curve_log_dir or default_curve_dir,
                log_env=int(cfg.log_env),
            )
        except ImportError as e:
            logger.warning("[TactileReward] disabled: %s", e)
            return None

    @staticmethod
    def _import_model_class(rewind_root: str | None):
        """Make ``Tactile-ReWiND/tools/`` importable and return the transformer."""
        root = os.path.expanduser(rewind_root or _DEFAULT_REWIND_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from tools.tactile_model import TactileReWiNDTransformer
        except Exception as e:
            raise ImportError(
                f"[TactileReward] FAILED import (rewind_root={root}): {e}"
            ) from e
        return TactileReWiNDTransformer

    def _encode_instruction(self, instruction: str) -> torch.Tensor:
        """Mean-pool MiniLM token embeddings once, then drop the encoder."""
        from transformers import AutoTokenizer, AutoModel

        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        minilm = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(self.device)
        minilm.eval()
        with torch.no_grad():
            enc = tok([instruction], padding=True, return_tensors="pt").to(self.device)
            tok_emb = minilm(**enc)[0]
            mask = enc["attention_mask"].unsqueeze(-1).expand(tok_emb.size()).float()
            text_emb = (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        del minilm, tok
        return text_emb.float()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def compute(self, frame: torch.Tensor) -> torch.Tensor:
        """Push one tactile frame into the history and return the scaled reward.

        Args:
            frame: ``(num_envs, rows, cols, 3)`` force field, channel order
                ``(normal, shear_x, shear_y)``. For a two-finger gripper the
                caller concatenates the two pads along the row dim.

        Returns:
            ``(num_envs,)`` predicted progress in ``[0, 1]``, EMA-smoothed if
            ``smooth_alpha < 1``. Unscaled — apply reward shaping in the caller.
        """
        self._validate_frame(frame)
        current = frame.float()[..., list(self.shear_channels)].detach()
        buffer = self._ensure_buffer(current)

        buffer = torch.roll(buffer, shifts=-1, dims=1)
        buffer[:, -1] = current
        self._buffer = buffer
        self._step_count = torch.clamp(self._step_count + 1, max=self.history_length)

        slc = self._gather_slice(buffer)

        if self.normalize_mode == "global":
            denom = slc.abs().amax(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
            slc = slc / denom
        elif self.normalize_mode == "per_channel":
            denom = slc.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            slc = slc / denom

        x = slc.permute(0, 1, 4, 2, 3).contiguous()
        text = self.text_emb.expand(self.num_envs, -1)
        with torch.no_grad():
            progress = self.model(x, text).squeeze(-1)
        latest = progress[:, -1]

        alpha = self.smooth_alpha
        if alpha < 1.0:
            self._smoothed_progress = alpha * latest + (1.0 - alpha) * self._smoothed_progress
            out = self._smoothed_progress
        else:
            self._smoothed_progress = latest
            out = latest

        if self.curve_log_dir is not None and self.log_env < self.num_envs:
            self._progress_history.append(
                (latest[self.log_env].item(), out[self.log_env].item())
            )

        # Copy: with smoothing on, `out` IS the internal EMA buffer, and
        # reset_idx() zeroes it in place. A caller holding the returned tensor
        # across an episode reset would otherwise see its values change
        # underneath it. Costs num_envs floats.
        return out.clone()

    def _validate_frame(self, frame: torch.Tensor) -> None:
        """Fail fast, and in the caller's terms, on a malformed tactile frame.

        Without this a bad layout surfaces either as a broadcast error inside
        the rolling buffer or as a complaint from the CNN encoder about a
        reshaped ``(B*T, C, H, W)`` tensor the caller never constructed.
        """
        if frame.ndim != 4:
            raise ValueError(
                f"frame must be 4D (num_envs, rows, cols, channels); "
                f"got shape {tuple(frame.shape)}"
            )
        n, rows, cols, ch = frame.shape

        if n != self.num_envs:
            raise ValueError(
                f"frame has batch size {n} but this model was built for "
                f"num_envs={self.num_envs}"
            )

        needed = max(self.shear_channels) + 1
        if ch < needed:
            raise ValueError(
                f"ckpt {os.path.basename(self.ckpt_path)} selects channels "
                f"{self.shear_channels}, so frames need at least {needed} channels; "
                f"got {ch}. Expected channel order (normal, shear_x, shear_y)."
            )

        # The encoder splits the frame in half to recover the two pads.
        split_len, axis_name = (
            (rows, "rows") if self.bimanual_axis == "height" else (cols, "cols")
        )
        if split_len % 2 != 0:
            raise ValueError(
                f"bimanual_axis={self.bimanual_axis!r} splits the frame in half along "
                f"{axis_name}, so {axis_name} must be even; got {split_len}. Concatenate "
                f"the two sensor pads along that axis before calling compute()."
            )

        if frame.device.type != self.device.type:
            raise ValueError(
                f"frame is on device {frame.device} but the model is on {self.device}"
            )

        # The history buffer fixes the layout at the first frame; a later change
        # would silently corrupt the rolling window.
        if self._buffer is not None:
            expected = tuple(self._buffer.shape[2:])
            got = (rows, cols, self.in_channels)
            if got != expected:
                raise ValueError(
                    f"frame layout changed mid-episode: history holds "
                    f"(rows, cols, channels)={expected}, got {got}. Construct a new "
                    f"TactileRewardModel if the sensor layout changes."
                )

    def _ensure_buffer(self, current: torch.Tensor) -> torch.Tensor:
        if self._buffer is None:
            rows, cols, ch = current.shape[1:]
            self._buffer = torch.zeros(
                self.num_envs, self.history_length, rows, cols, ch,
                device=self.device, dtype=torch.float32,
            )
        return self._buffer

    def _gather_slice(self, buffer: torch.Tensor) -> torch.Tensor:
        """Select ``max_length`` frames out of the valid history window.

        Once the window holds at least ``max_length`` valid frames we
        linspace-subsample it (matching training's stride); before that we take
        consecutive frames and clamp-repeat the newest one.
        """
        H, T = self.history_length, self.max_length
        valid = self._step_count.clamp(min=1)
        start = (H - valid).long()

        t_grid = torch.arange(T, device=self.device)
        frac = t_grid.float() / float(T - 1) if T > 1 else torch.zeros(T, device=self.device)
        span = (H - 1 - start).float().unsqueeze(1)
        long_idx = (start.float().unsqueeze(1) + span * frac.unsqueeze(0)).round().long()
        long_idx.clamp_(0, H - 1)

        short_idx = start.unsqueeze(1) + torch.minimum(t_grid.unsqueeze(0), (valid - 1).unsqueeze(1))
        short_idx.clamp_(0, H - 1)

        sel = torch.where((valid >= T).unsqueeze(1), long_idx, short_idx)

        _, _, rows, cols, ch = buffer.shape
        gather_idx = sel[:, :, None, None, None].expand(-1, -1, rows, cols, ch)
        return torch.gather(buffer, 1, gather_idx)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def reset_idx(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Clear the rolling history for the given envs (call from ``_reset_idx``)."""
        if self._is_log_env_resetting(env_ids):
            self._save_progress_curve()
            self._progress_history.clear()

        if self._buffer is not None:
            self._buffer[env_ids] = 0.0
        self._step_count[env_ids] = 0
        self._smoothed_progress[env_ids] = 0.0

    def _is_log_env_resetting(self, env_ids: Sequence[int] | torch.Tensor) -> bool:
        if isinstance(env_ids, torch.Tensor):
            return bool((env_ids == self.log_env).any().item())
        return self.log_env in env_ids

    def _save_progress_curve(self) -> None:
        """Dump the buffered progress history for ``log_env`` to a PNG."""
        if not self.curve_log_dir or not self._progress_history:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            logger.warning("[TactileReward] matplotlib unavailable, skipping curve dump: %s", e)
            return

        history = self._progress_history
        alpha = self.smooth_alpha
        raw_series = [t[0] for t in history]
        sm_series = [t[1] for t in history]
        steps = list(range(len(history)))

        # Model output only — the downstream reward scale is not plotted, since
        # scaling now happens in the caller.
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, raw_series, color="C0", linewidth=1.0, alpha=0.45, label="raw progress")
        if alpha < 1.0:
            ax.plot(steps, sm_series, color="C2", linewidth=1.5,
                    label=f"EMA smoothed (α={alpha})")
        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_xlabel("env step within episode")
        ax.set_ylabel("tactile progress")
        ax.set_title(
            f"env {self.log_env} | episode {self._curve_episode_idx} | steps={len(history)}"
        )
        ax.set_ylim(-0.1, 1.2)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        out_path = os.path.join(
            self.curve_log_dir, f"env{self.log_env}_ep{self._curve_episode_idx:06d}.png"
        )
        try:
            fig.tight_layout()
            fig.savefig(out_path, dpi=110)
        except Exception as e:
            logger.warning("[TactileReward] failed to save curve PNG: %s", e)
        plt.close(fig)
        self._curve_episode_idx += 1
