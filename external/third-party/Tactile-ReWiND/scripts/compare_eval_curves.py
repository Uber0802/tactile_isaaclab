"""Compare reward curves across multiple checkpoint epochs.

Reads JSON outputs from eval_reward_curves.py and plots a 1x3 subplot:
forward / rewind / mismatched, each with one line per epoch.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPOCH_RE = re.compile(r"epoch(\d+)")


def epoch_from_path(p: str) -> int:
    m = EPOCH_RE.search(os.path.basename(p))
    if not m:
        raise ValueError(f"can't extract epoch from {p}")
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_glob",
                    default="/mnt/lab-tank/uber/Tactile-Reward/eval_curves/*_curves.json",
                    help="Glob pattern for the per-epoch JSON outputs.")
    ap.add_argument("--output",
                    default="/mnt/lab-tank/uber/Tactile-Reward/eval_curves/epoch_compare.png")
    ap.add_argument("--epochs", default=None,
                    help="Comma-separated epochs to include (default: all matches).")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.json_glob), key=epoch_from_path)
    if args.epochs:
        wanted = {int(e) for e in args.epochs.split(",")}
        paths = [p for p in paths if epoch_from_path(p) in wanted]
    if not paths:
        raise SystemExit(f"no JSON files matched {args.json_glob}")

    runs = []
    for p in paths:
        with open(p) as fh:
            data = json.load(fh)
        runs.append({
            "epoch": epoch_from_path(p),
            "frames": np.array(data["frames"]),
            "forward_mean": np.array(data["forward_mean"]),
            "rewind_mean": np.array(data["rewind_mean"]),
            "mismatched_mean": np.array(data["mismatched_mean"]),
            "n_traj": data["n_trajectories"],
        })
    print(f"loaded {len(runs)} epochs: {[r['epoch'] for r in runs]}")

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(runs) - 1, 1)) for i in range(len(runs))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    panels = [
        ("forward (correct lang)", "forward_mean", "tab:green"),
        ("rewind (fwd→rev)",       "rewind_mean", "tab:red"),
        ("forward + mismatched lang", "mismatched_mean", "tab:purple"),
    ]
    max_len = max(r["frames"].shape[0] for r in runs)
    ideal_x = np.arange(max_len)
    ideal_y = np.linspace(0, 1, max_len)

    for ax, (title, key, _) in zip(axes, panels):
        for run, c in zip(runs, colors):
            ax.plot(run["frames"], run[key], color=c, linewidth=2,
                    label=f"epoch {run['epoch']}")
        if key == "forward_mean":
            ax.plot(ideal_x, ideal_y, color="gray", linestyle="--",
                    linewidth=1, label="ideal ramp")
        ax.set_title(title)
        ax.set_xlabel("frame index")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("predicted reward")
    axes[-1].legend(loc="upper right", fontsize=8)
    n_traj = runs[0]["n_traj"]
    fig.suptitle(f"reward curves across epochs (n={n_traj} eval trajectories)",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=130)
    plt.close(fig)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
