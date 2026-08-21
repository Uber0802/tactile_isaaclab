"""Bridge to the repo-root ``tactile_reward_model`` package.

That package lives at the repository root, which is not on ``sys.path`` when
``isaaclab_tasks`` is used as an installed package. This module puts it there
and re-exports its public names, so callers inside isaaclab_tasks just do::

    from isaaclab_tasks.utils.tactile_reward_import import TactileRewardCfg
    from isaaclab_tasks.utils.tactile_reward_import import TactileRewardModel

The path fix has to happen at import time rather than inside a function,
because the env *config* modules need ``TactileRewardCfg`` while their class
bodies are being executed.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["TactileRewardCfg", "TactileRewardModel"]


def _ensure_importable() -> None:
    """Put the repository root on ``sys.path`` if the package isn't visible."""
    try:
        import tactile_reward_model  # noqa: F401
        return
    except ImportError:
        pass

    for parent in Path(__file__).resolve().parents:
        if (parent / "tactile_reward_model" / "tactile_reward_model.py").is_file():
            sys.path.insert(0, str(parent))
            return

    raise ImportError(
        "could not locate the tactile_reward_model package from "
        f"{Path(__file__).resolve()}"
    )


_ensure_importable()

from tactile_reward_model import TactileRewardCfg, TactileRewardModel  # noqa: E402
