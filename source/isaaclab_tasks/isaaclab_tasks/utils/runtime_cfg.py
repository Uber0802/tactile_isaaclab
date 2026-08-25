"""Typed config for the optional runtime features that were env-var driven.

Covers trajectory saving, the frozen tactile encoder, and the ReWiND visual
reward. Companion to ``tactile_reward_model.TactileRewardCfg``, which lives with
its model; these have no standalone package, so they live here.

Each replaces a family of ``FORGE_*`` environment variables with fields that go
through Hydra like the rest of the config::

    env.tactile_save.force_field=true env.tactile_save.save_dir=/data/peg
    env.visual_reward.ckpt=/path/to/rewind.pth env.visual_reward.scale=0.3

Unknown keys are rejected instead of silently falling back to a default, and the
resolved values land in the run's ``params/env.yaml``.

As with ``TactileRewardCfg``, no field defaults to ``None``: IsaacLab's
``update_class_from_dict`` type-checks an override against
``type(current_value)``, so a ``None`` default would reject every override.
Empty string / 0 are the "unset" sentinels.

Note these are all consumed in the env's ``__init__``, i.e. after Hydra has
applied CLI overrides. Flags consumed inside a config's ``__post_init__``
(``FORGE_SKIP_TACTILE_SENSORS``, ``FORGE_ENABLE_FRONT_CAM``,
``FORGE_ENABLE_SENSOR``, ``FORGE_DISABLE_YAW_DIFF_OBS``) cannot move here:
``register_task_to_hydra`` instantiates the config before ``hydra_main`` applies
overrides, so a CLI value would arrive too late to affect scene construction.
"""

from __future__ import annotations

from isaaclab.utils import configclass

__all__ = ["TactileEncoderCfg", "TactileSaveCfg", "VisualRewardCfg"]


@configclass
class TactileSaveCfg:
    """Per-episode tactile / camera trajectory dumping. ``force_field`` gates it."""

    force_field: bool = False
    """Save the GelSight force field each step (was FORGE_SAVE_TACTILE_FORCE_FIELD)."""

    all_envs: bool = False
    """Buffer every env rather than only the single target env. Multiplies yield."""

    camera: bool = False
    """Also save front-camera RGB. Forge additionally enables this whenever
    ``force_field`` is set, preserving the legacy combined behavior."""

    save_dir: str = ""
    """Output directory. Empty = the env's own default, which differs per env."""

    save_interval: int = 1
    """Keep every Nth step. Clamped to >= 1."""

    max_buffer_frames: int = 500000
    """Safety cap on total buffered frames across per-env buffers (all_envs mode)."""

    episodes_per_env: int = 0
    """Per-env episode quota. 0 = unlimited."""


@configclass
class TactileEncoderCfg:
    """Frozen tactile CNN encoder producing an embedding obs. Empty ``ckpt`` = off."""

    ckpt: str = ""
    """Checkpoint holding the ``encoder.*`` weights. Empty disables the encoder."""

    dim: int = 0
    """Declared embedding width. 0 = infer from the checkpoint."""

    root: str = ""
    """Path to the Tactile-ReWiND checkout. Empty = the vendored copy."""


@configclass
class VisualRewardCfg:
    """Dense progress reward from RGB via DINOv2 + ReWiND. Empty ``ckpt`` = off.

    Needs the front camera in the scene, which is still an ``__post_init__``-time
    decision — see the module docstring.
    """

    ckpt: str = ""
    """Path to the ReWiND ``.pth``. Empty disables the reward entirely."""

    scale: float = 1.0
    """Multiplier on the predicted progress."""

    # -- auxiliary shaping group: read by ForgeEnv, not by the visual model.
    # Mirrors TactileRewardCfg's group so both heads anneal with the same knobs;
    # each head keeps independent trigger/ramp state at runtime.

    scale_end: float = 0.0
    """Target scale to anneal toward. Only read when ``anneal_steps > 0``."""

    anneal_steps: int = 0
    """Env control-steps to ramp ``scale`` -> ``scale_end``. 0 disables annealing."""

    anneal_mode: str = "linear"
    """``"linear"`` ramps from step 0; ``"success"`` holds ``scale`` until the
    running episode success rate first reaches ``anneal_success_thresh``, then
    ramps over ``anneal_steps`` from that moment."""

    anneal_success_thresh: float = 0.01
    """Success rate that fires the ramp in ``"success"`` mode. Ignored otherwise."""

    anneal_success_ema_alpha: float = 0.1
    """EMA coefficient on the per-reset success rate used as the trigger signal."""

    smooth_alpha: float = 1.0
    """EMA coefficient on the predicted progress. 1.0 disables smoothing."""

    instruction: str = ""
    """Task string encoded by MiniLM. Empty keeps the env's own default wording."""

    root: str = "~/ReWiND"
    """Path to the ReWiND repo, holding ``model.py``."""

    history: int = 0
    """Rolling-buffer length. 0 = the episode length."""

    backbone: str = "dinov2_vitb14"
    """torch.hub DINOv2 backbone name."""

    dino_interval: int = 1
    """Run DINOv2 every N sim steps, reusing features between. Clamped to >= 1."""
