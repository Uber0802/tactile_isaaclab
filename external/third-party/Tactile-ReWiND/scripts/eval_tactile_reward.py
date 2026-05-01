"""Evaluate a tactile ReWiND reward model on held-out trajectories.

For each task in the eval metadata H5, run every held-out trajectory through
the model with (a) the correct instruction and (b) a randomly sampled
mismatched instruction. Report:

  * Pearson correlation between predicted progress and a linear ground-truth
    ramp, conditioned on the correct instruction (higher = better)
  * Mean predicted progress under a mismatched instruction (lower = better,
    ideally near 0)
  * "Language gap" = correct mean progress - mismatched mean progress

These numbers tell us: is the model picking up real progress signal, and is it
actually conditioning on language?

Usage:
    python scripts/eval_tactile_reward.py \\
        --ckpt /mnt/tank/uber/Tactile-Reward/checkpoints/tactile_rewind_epoch19.pth \\
        --eval_metadata /mnt/tank/uber/Tactile-Reward/tactile_metadata_eval.h5
"""
from __future__ import annotations

import os
import sys
import json
import random
import argparse
from collections import defaultdict

import h5py
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.tactile_model import TactileReWiNDTransformer


def load_model(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("args", {})
    isaaclab_aligned = bool(cfg.get("isaaclab_aligned", False))
    num_strided = cfg.get("num_strided_layers", 0)
    if num_strided in (0, None):
        num_strided = 3 if isaaclab_aligned else 5
    bimanual_axis = "height" if isaaclab_aligned else "width"
    model = TactileReWiNDTransformer(
        max_length=cfg.get("max_length", 16),
        text_dim=384,
        hidden_dim=cfg.get("hidden_dim", 512),
        num_heads=cfg.get("num_heads", 8),
        num_layers=cfg.get("num_layers", 4),
        per_hand_dim=cfg.get("per_hand_dim", 384),
        num_strided_layers=num_strided,
        bimanual_axis=bimanual_axis,
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, cfg


def subsample_to_max_length(traj: np.ndarray, max_length: int) -> np.ndarray:
    """Pick `max_length` evenly-spaced frames from a (T, H, W, C) trajectory."""
    T = len(traj)
    if T <= max_length:
        idx = np.arange(T)
        pad = max_length - T
        if pad > 0:
            idx = np.concatenate([idx, np.full(pad, T - 1, dtype=int)])
    else:
        idx = np.linspace(0, T - 1, max_length).astype(int)
    return np.ascontiguousarray(traj[idx])


def to_chw_tensor(frames_thwc: np.ndarray, device, isaaclab_aligned: bool = False) -> torch.Tensor:
    """(T, H, W, C) float16 mmap → (1, T, C, H, W) float32 on device.

    If `isaaclab_aligned`, also runs the same split→transpose→pool→re-concat
    transform that `TactileReWiNDDataset(align_to_isaaclab=True)` applies.
    """
    x = torch.from_numpy(np.ascontiguousarray(frames_thwc, dtype=np.float32))
    x = x.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
    if isaaclab_aligned:
        from tools.tactile_dataset import TactileReWiNDDataset
        x = TactileReWiNDDataset._align_to_isaaclab(x)
    return x.unsqueeze(0).to(device)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom < 1e-12:
        return float("nan")
    return float((x * y).sum() / denom)


def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_model(args.ckpt, device)
    max_length = cfg.get("max_length", 16)
    isaaclab_aligned = bool(cfg.get("isaaclab_aligned", False))
    print(f"loaded {args.ckpt} (max_length={max_length}, "
          f"isaaclab_aligned={isaaclab_aligned})")

    with h5py.File(args.eval_metadata, "r") as h5:
        data_dir = args.data_dir_override or h5.attrs["data_dir"]
        tasks = sorted(h5.keys())
        print(f"eval set: {len(tasks)} tasks from {args.eval_metadata}")

        # Pre-load lang embeddings + traj file lists
        lang = {}
        files = {}
        for task in tasks:
            grp = h5[task]
            lang[task] = np.asarray(grp["minilm_lang_embedding"], dtype=np.float32)
            files[task] = [
                s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                for s in np.asarray(grp["trajectory_files"])
            ]

    per_task = {}
    correct_progress_all, ramp_all = [], []
    mismatched_progress_all = []

    for task in tasks:
        if not files[task]:
            continue
        correct_text = torch.from_numpy(lang[task][0]).float().unsqueeze(0).to(device)

        # Pick a mismatched task with a different lang embedding
        other = task
        while other == task and len(tasks) > 1:
            other = random.choice(tasks)
        mismatched_text = torch.from_numpy(lang[other][0]).float().unsqueeze(0).to(device)

        per_traj_corr = []
        per_traj_correct_mean = []
        per_traj_mismatched_mean = []

        for fname in files[task]:
            path = os.path.join(data_dir, fname)
            traj_full = np.load(path, mmap_mode="r")
            if len(traj_full) < 3:
                continue
            frames = subsample_to_max_length(traj_full, max_length)
            ramp = np.linspace(0, 1, max_length, dtype=np.float32)

            x = to_chw_tensor(frames, device, isaaclab_aligned=isaaclab_aligned)
            with torch.no_grad():
                pred_correct = model(x, correct_text).squeeze(-1).squeeze(0).cpu().numpy()
                pred_wrong = model(x, mismatched_text).squeeze(-1).squeeze(0).cpu().numpy()

            # Skip first frame (matches train-time loss)
            corr = pearson(pred_correct[1:], ramp[1:])
            per_traj_corr.append(corr)
            per_traj_correct_mean.append(float(pred_correct[1:].mean()))
            per_traj_mismatched_mean.append(float(pred_wrong[1:].mean()))

            correct_progress_all.append(pred_correct[1:])
            ramp_all.append(ramp[1:])
            mismatched_progress_all.append(pred_wrong[1:])

        per_task[task] = {
            "n_trajectories": len(per_traj_corr),
            "mean_pearson_correct": float(np.nanmean(per_traj_corr)) if per_traj_corr else float("nan"),
            "mean_progress_correct": float(np.nanmean(per_traj_correct_mean)) if per_traj_correct_mean else float("nan"),
            "mean_progress_mismatched": float(np.nanmean(per_traj_mismatched_mean)) if per_traj_mismatched_mean else float("nan"),
        }
        gap = per_task[task]["mean_progress_correct"] - per_task[task]["mean_progress_mismatched"]
        per_task[task]["language_gap"] = gap

    # Aggregate
    correct_progress_flat = np.concatenate(correct_progress_all) if correct_progress_all else np.zeros(0)
    ramp_flat = np.concatenate(ramp_all) if ramp_all else np.zeros(0)
    mismatched_flat = np.concatenate(mismatched_progress_all) if mismatched_progress_all else np.zeros(0)

    summary = {
        "n_tasks": len(per_task),
        "n_trajectories": int(sum(v["n_trajectories"] for v in per_task.values())),
        "global_pearson_correct": pearson(correct_progress_flat, ramp_flat),
        "global_progress_correct": float(correct_progress_flat.mean()) if len(correct_progress_flat) else float("nan"),
        "global_progress_mismatched": float(mismatched_flat.mean()) if len(mismatched_flat) else float("nan"),
        "global_language_gap": float(correct_progress_flat.mean() - mismatched_flat.mean())
            if len(correct_progress_flat) and len(mismatched_flat) else float("nan"),
        "per_task": per_task,
    }

    print()
    print("=" * 64)
    print(f"global Pearson (correct lang)        : {summary['global_pearson_correct']:.4f}")
    print(f"global mean progress (correct lang)  : {summary['global_progress_correct']:.4f}")
    print(f"global mean progress (mismatched)    : {summary['global_progress_mismatched']:.4f}")
    print(f"global language gap (correct - wrong): {summary['global_language_gap']:+.4f}")
    print("=" * 64)
    print()

    sorted_tasks = sorted(per_task.items(), key=lambda kv: kv[1]["mean_pearson_correct"], reverse=True)
    print(f"{'task':<40} {'corr':>7} {'p_corr':>8} {'p_wrong':>8} {'gap':>8}  n")
    for task, m in sorted_tasks:
        print(f"{task:<40} {m['mean_pearson_correct']:>7.3f} "
              f"{m['mean_progress_correct']:>8.3f} {m['mean_progress_mismatched']:>8.3f} "
              f"{m['language_gap']:>+8.3f}  {m['n_trajectories']}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nwrote per-task metrics to {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Path to trained tactile reward checkpoint (.pth).")
    ap.add_argument("--eval_metadata",
                    default="/mnt/tank/uber/Tactile-Reward/tactile_metadata_eval.h5")
    ap.add_argument("--data_dir_override", default=None,
                    help="Override the data_dir attribute stored in the metadata H5.")
    ap.add_argument("--output", default=None,
                    help="Optional JSON path to dump per-task metrics.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for picking mismatched task per query.")
    args = ap.parse_args()
    main(args)
