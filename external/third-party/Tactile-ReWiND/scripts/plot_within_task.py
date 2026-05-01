"""Plot within-task eval performance from one or more eval JSONs.

Each --json arg is `path:label`. Produces a single PNG with three panels:
  1. Per-task Pearson bar chart, sorted by best model's value
  2. Per-task language gap bar chart, same task ordering
  3. Distribution histograms of per-task Pearson (one per model)

Usage:
    python scripts/plot_within_task.py \\
        --json /mnt/tank/uber/Tactile-Reward/eval_within_task.json:320x480 \\
        --json /mnt/tank/uber/Tactile-Reward/eval_aligned_within.json:aligned \\
        --json /mnt/tank/uber/Tactile-Reward/eval_aligned_long2long_within.json:long2long \\
        --output /mnt/tank/uber/Tactile-Reward/within_task_compare.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_eval_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def main(args):
    runs = []
    for spec in args.json:
        if ":" in spec:
            path, label = spec.rsplit(":", 1)
        else:
            path, label = spec, Path(spec).stem
        runs.append((label, load_eval_json(path)))
    if not runs:
        raise SystemExit("need at least one --json arg")

    # Union of tasks across all runs
    all_tasks = sorted({t for _, d in runs for t in d.get("per_task", {}).keys()})
    n_tasks = len(all_tasks)

    # Build matrices: (n_tasks, n_runs)
    pearson = np.full((n_tasks, len(runs)), np.nan)
    gap = np.full((n_tasks, len(runs)), np.nan)
    for j, (_, d) in enumerate(runs):
        per_task = d.get("per_task", {})
        for i, t in enumerate(all_tasks):
            entry = per_task.get(t, {})
            p = entry.get("mean_pearson_correct")
            g = entry.get("language_gap")
            if p is not None and not np.isnan(p):
                pearson[i, j] = p
            if g is not None and not np.isnan(g):
                gap[i, j] = g

    # Sort tasks by the LAST run's Pearson (typically the most recent / best)
    sort_col = len(runs) - 1
    order = np.argsort(-np.where(np.isnan(pearson[:, sort_col]),
                                 -np.inf, pearson[:, sort_col]))
    tasks_sorted = [all_tasks[i] for i in order]
    pearson_sorted = pearson[order]
    gap_sorted = gap[order]

    # Figure
    fig = plt.figure(figsize=(20, 14), dpi=120)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 1.2],
                          left=0.06, right=0.99, top=0.95, bottom=0.06,
                          hspace=0.35)
    ax_p = fig.add_subplot(gs[0])
    ax_g = fig.add_subplot(gs[1])
    ax_h = fig.add_subplot(gs[2])

    palette = plt.get_cmap("tab10").colors
    width = 0.8 / len(runs)
    x = np.arange(n_tasks)

    # Pearson bars
    for j, (label, d) in enumerate(runs):
        col = palette[j % len(palette)]
        global_p = d.get("global_pearson_correct", float("nan"))
        ax_p.bar(x + (j - (len(runs) - 1) / 2) * width,
                 pearson_sorted[:, j], width=width, color=col,
                 label=f"{label}  (global={global_p:.3f})")
    ax_p.axhline(0, color="black", lw=0.5)
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(tasks_sorted, rotation=70, ha="right", fontsize=7)
    ax_p.set_ylabel("Pearson(pred, ramp)")
    ax_p.set_title("Per-task Pearson correlation (correct lang) — sorted by last run")
    ax_p.legend(loc="lower left", fontsize=10)
    ax_p.grid(axis="y", alpha=0.3)
    ax_p.set_ylim(-1.05, 1.05)

    # Language gap bars (same task order)
    for j, (label, d) in enumerate(runs):
        col = palette[j % len(palette)]
        global_gap = d.get("global_language_gap", float("nan"))
        ax_g.bar(x + (j - (len(runs) - 1) / 2) * width,
                 gap_sorted[:, j], width=width, color=col,
                 label=f"{label}  (global gap={global_gap:+.3f})")
    ax_g.axhline(0, color="black", lw=0.5)
    ax_g.set_xticks(x)
    ax_g.set_xticklabels(tasks_sorted, rotation=70, ha="right", fontsize=7)
    ax_g.set_ylabel("language gap (correct − mismatched mean progress)")
    ax_g.set_title("Per-task language gap — same task ordering")
    ax_g.legend(loc="upper right", fontsize=10)
    ax_g.grid(axis="y", alpha=0.3)

    # Histogram of Pearson
    bins = np.linspace(-1.0, 1.0, 21)
    for j, (label, _) in enumerate(runs):
        col = palette[j % len(palette)]
        valid = pearson_sorted[:, j][~np.isnan(pearson_sorted[:, j])]
        ax_h.hist(valid, bins=bins, alpha=0.5, color=col, label=label,
                  edgecolor="black", linewidth=0.5)
    ax_h.axvline(0, color="black", lw=0.5)
    ax_h.set_xlabel("Pearson")
    ax_h.set_ylabel("# tasks")
    ax_h.set_title("Distribution of per-task Pearson")
    ax_h.legend(fontsize=10)
    ax_h.grid(alpha=0.3)

    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="append", required=True,
                    help="path/to/eval.json[:label] — pass multiple times to compare runs.")
    ap.add_argument("--output",
                    default="/mnt/tank/uber/Tactile-Reward/within_task_compare.png")
    args = ap.parse_args()
    main(args)
