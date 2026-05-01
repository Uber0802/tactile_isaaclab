"""For each eval task, run two model inferences and overlay the resulting
predicted-progress curves so you can eyeball whether the model recognises
rewinds.

Per task:
  * forward  : 16 evenly-spaced frames over [0, T-1]            (progress should climb 0 -> 1)
  * rewind   : forward to peak frame at peak_ratio*T, then play
               backward to frame 0                              (progress should climb then fall)

Plots a grid of small panels — one per task — with both curves
overlaid plus a dashed reference ramp.

Usage:
    python scripts/eval_forward_vs_rewind.py \\
        --ckpt /mnt/tank/uber/Tactile-Reward/checkpoints_aligned_long2long/tactile_rewind_epoch19.pth \\
        --output /mnt/tank/uber/Tactile-Reward/forward_vs_rewind.png
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.tactile_model import TactileReWiNDTransformer
from tools.tactile_dataset import TactileReWiNDDataset


def load_model(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("args", {})
    isaaclab_aligned = bool(cfg.get("isaaclab_aligned", False))
    num_strided = cfg.get("num_strided_layers", 0) or (3 if isaaclab_aligned else 5)
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
    return model, cfg, isaaclab_aligned


def to_chw_tensor(frames_thwc: np.ndarray, device, isaaclab_aligned: bool) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(frames_thwc, dtype=np.float32))
    x = x.permute(0, 3, 1, 2).contiguous()
    if isaaclab_aligned:
        x = TactileReWiNDDataset._align_to_isaaclab(x)
    return x.unsqueeze(0).to(device)


def forward_indices(T: int, max_length: int) -> np.ndarray:
    """Evenly spaced 16 frames covering the whole traj."""
    return np.round(np.linspace(0, T - 1, max_length)).astype(int)


def rewind_indices(T: int, max_length: int, peak_ratio: float) -> np.ndarray:
    """Forward to frame round(peak_ratio*(T-1)) then backward to 0.

    Half (rounded up) of max_length goes to the forward leg, the rest is the reverse leg.
    """
    peak = int(round(peak_ratio * (T - 1)))
    peak = max(2, min(T - 1, peak))
    n_fwd = (max_length + 1) // 2          # 9 if max_length=16
    n_rev = max_length - n_fwd              # 7
    fwd = np.round(np.linspace(0, peak, n_fwd)).astype(int)
    rev = np.round(np.linspace(peak - 1, 0, n_rev)).astype(int) if n_rev > 0 else np.empty(0, dtype=int)
    return np.concatenate([fwd, rev])


def gt_progress_forward(max_length: int) -> np.ndarray:
    return np.linspace(0, 1, max_length, dtype=np.float32)


def gt_progress_rewind(max_length: int, peak_ratio: float) -> np.ndarray:
    """Ground-truth progress for the rewind sequence (rises then mirrors back)."""
    n_fwd = (max_length + 1) // 2
    n_rev = max_length - n_fwd
    peak_progress = peak_ratio
    fwd = np.linspace(0, peak_progress, n_fwd)
    rev = np.linspace(peak_progress * (n_fwd - 1) / n_fwd, 0, n_rev) if n_rev > 0 else np.empty(0)
    return np.concatenate([fwd, rev]).astype(np.float32)


def _save_single_panel(out_path, x_axis, gt_fwd, gt_rew,
                       pred_fwd, pred_rew, peak_ratio, title, subtitle=""):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    fig.patch.set_facecolor("white")
    ax.plot(x_axis, gt_fwd, "--", color="0.5", lw=1.0, alpha=0.7, label="GT forward")
    if gt_rew is not None:
        ax.plot(x_axis, gt_rew, ":", color="0.5", lw=1.0, alpha=0.7, label="GT rewind")
    ax.plot(x_axis, pred_fwd, "-", color="tab:blue", lw=2.2, label="forward")
    label_rew = (f"rewind (peak={peak_ratio:.2f})"
                 if peak_ratio is not None else "rewind")
    ax.plot(x_axis, pred_rew, "-", color="tab:orange", lw=2.2, label=label_rew)
    ax.set_xlabel("frame index")
    ax.set_ylabel("predicted progress")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, len(x_axis) - 0.5)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_per_task_pngs(results, max_length, fwd_mean, fwd_std, rew_mean, rew_std,
                       out_dir, ckpt_label=""):
    os.makedirs(out_dir, exist_ok=True)
    x_axis = np.arange(max_length)

    for task, r in results.items():
        gt_fwd = gt_progress_forward(max_length)
        gt_rew = gt_progress_rewind(max_length, r["rewind_peak_ratio"])
        sub = (f"T_traj={r['T_traj']}, peak={r['rewind_peak_ratio']:.2f}  |  "
               f"forward mean={np.mean(r['pred_forward']):.3f}, "
               f"rewind mean={np.mean(r['pred_rewind']):.3f}")
        _save_single_panel(
            os.path.join(out_dir, f"{task}.png"),
            x_axis, gt_fwd, gt_rew,
            np.array(r["pred_forward"]), np.array(r["pred_rewind"]),
            r["rewind_peak_ratio"],
            title=f"{task} ({ckpt_label})",
            subtitle=sub,
        )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    fig.patch.set_facecolor("white")
    ax.plot(x_axis, gt_progress_forward(max_length), "--", color="0.5", lw=1.0,
            alpha=0.7, label="GT ramp (forward)")
    ax.plot(x_axis, fwd_mean, "-", color="tab:blue", lw=2.4, label="forward (mean)")
    ax.fill_between(x_axis, fwd_mean - fwd_std, fwd_mean + fwd_std,
                    color="tab:blue", alpha=0.15, label="forward ±1σ")
    ax.plot(x_axis, rew_mean, "-", color="tab:orange", lw=2.4, label="rewind (mean)")
    ax.fill_between(x_axis, rew_mean - rew_std, rew_mean + rew_std,
                    color="tab:orange", alpha=0.15, label="rewind ±1σ")
    ax.set_xlabel("frame index")
    ax.set_ylabel("predicted progress")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, max_length - 0.5)
    ax.set_title(f"aggregate across {len(results)} eval tasks   |   {ckpt_label}",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "aggregate.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {len(results)} per-task PNGs + aggregate.png to {out_dir}/")


def main(args):
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, isaaclab_aligned = load_model(args.ckpt, device)
    max_length = cfg.get("max_length", 16)
    print(f"loaded {args.ckpt} (max_length={max_length}, isaaclab_aligned={isaaclab_aligned})")

    rng = np.random.default_rng(args.seed)

    with h5py.File(args.eval_metadata, "r") as h5:
        data_dir = args.data_dir_override or h5.attrs["data_dir"]
        tasks = sorted(h5.keys())
        meta = {}
        for task in tasks:
            grp = h5[task]
            meta[task] = {
                "lang": np.asarray(grp["minilm_lang_embedding"], dtype=np.float32),
                "files": [
                    s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in np.asarray(grp["trajectory_files"])
                ],
            }
    print(f"eval set: {len(tasks)} tasks from {args.eval_metadata}")

    results = {}
    for task in tasks:
        files = meta[task]["files"]
        if not files:
            continue
        path = os.path.join(data_dir, files[0])
        traj = np.load(path, mmap_mode="r")
        T = len(traj)
        if T < 4:
            continue

        text = torch.from_numpy(meta[task]["lang"][0]).float().unsqueeze(0).to(device)

        # Forward
        fwd_idx = forward_indices(T, max_length)
        x_fwd = to_chw_tensor(np.asarray(traj[fwd_idx]), device, isaaclab_aligned)
        with torch.no_grad():
            pred_fwd = model(x_fwd, text).squeeze(-1).squeeze(0).cpu().numpy()

        # Rewind: peak randomly in [peak_min, peak_max] for variety
        peak_ratio = float(rng.uniform(args.peak_min, args.peak_max))
        rew_idx = rewind_indices(T, max_length, peak_ratio)
        x_rew = to_chw_tensor(np.asarray(traj[rew_idx]), device, isaaclab_aligned)
        with torch.no_grad():
            pred_rew = model(x_rew, text).squeeze(-1).squeeze(0).cpu().numpy()

        results[task] = {
            "pred_forward": pred_fwd.astype(float).tolist(),
            "pred_rewind": pred_rew.astype(float).tolist(),
            "rewind_peak_ratio": peak_ratio,
            "T_traj": T,
        }

    # Aggregates
    fwd_curves = np.array([r["pred_forward"] for r in results.values()])
    rew_curves = np.array([r["pred_rewind"] for r in results.values()])
    fwd_mean, fwd_std = fwd_curves.mean(axis=0), fwd_curves.std(axis=0)
    rew_mean, rew_std = rew_curves.mean(axis=0), rew_curves.std(axis=0)

    # Plot grid
    n_cols = args.cols
    task_list = sorted(results.keys())
    n_panels = len(task_list) + 1   # +1 for the aggregate panel
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(args.col_size * n_cols,
                                                       args.row_size * n_rows),
                             dpi=120)
    axes = np.atleast_2d(axes).flatten()
    fig.patch.set_facecolor("white")

    x_axis = np.arange(max_length)
    for ax_idx, task in enumerate(task_list):
        ax = axes[ax_idx]
        r = results[task]
        gt_fwd = gt_progress_forward(max_length)
        gt_rew = gt_progress_rewind(max_length, r["rewind_peak_ratio"])
        ax.plot(x_axis, gt_fwd, "--", color="0.5", lw=0.8, alpha=0.6, zorder=0)
        ax.plot(x_axis, gt_rew, ":", color="0.5", lw=0.8, alpha=0.6, zorder=0)
        ax.plot(x_axis, r["pred_forward"], "-", color="tab:blue", lw=1.6, label="forward")
        ax.plot(x_axis, r["pred_rewind"], "-", color="tab:orange", lw=1.6,
                label=f"rewind@{r['rewind_peak_ratio']:.2f}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.5, max_length - 0.5)
        ax.set_title(task, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    # Aggregate panel
    ax_agg = axes[len(task_list)]
    gt_fwd = gt_progress_forward(max_length)
    ax_agg.plot(x_axis, gt_fwd, "--", color="0.5", lw=1.0, alpha=0.6, label="ramp")
    ax_agg.plot(x_axis, fwd_mean, "-", color="tab:blue", lw=2.2, label="forward mean")
    ax_agg.fill_between(x_axis, fwd_mean - fwd_std, fwd_mean + fwd_std,
                        color="tab:blue", alpha=0.15)
    ax_agg.plot(x_axis, rew_mean, "-", color="tab:orange", lw=2.2, label="rewind mean")
    ax_agg.fill_between(x_axis, rew_mean - rew_std, rew_mean + rew_std,
                        color="tab:orange", alpha=0.15)
    ax_agg.set_ylim(-0.05, 1.05)
    ax_agg.set_xlim(-0.5, max_length - 0.5)
    ax_agg.set_title("aggregate (mean ± 1σ across tasks)", fontsize=10, fontweight="bold")
    ax_agg.tick_params(labelsize=7)
    ax_agg.grid(alpha=0.25)
    ax_agg.legend(fontsize=7, loc="lower right")

    # Hide unused axes
    for k in range(len(task_list) + 1, len(axes)):
        axes[k].axis("off")

    handles = [
        plt.Line2D([0], [0], color="tab:blue", lw=2, label="forward"),
        plt.Line2D([0], [0], color="tab:orange", lw=2, label="rewind"),
        plt.Line2D([0], [0], color="0.5", lw=1, ls="--", label="GT (forward)"),
        plt.Line2D([0], [0], color="0.5", lw=1, ls=":", label="GT (rewind)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(f"Forward vs Rewind progress curves — {args.ckpt}",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")

    if args.json_output:
        with open(args.json_output, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote per-task curves JSON to {args.json_output}")

    if args.per_task_dir:
        save_per_task_pngs(
            results=results,
            max_length=max_length,
            fwd_mean=fwd_mean, fwd_std=fwd_std,
            rew_mean=rew_mean, rew_std=rew_std,
            out_dir=args.per_task_dir,
            ckpt_label=os.path.basename(os.path.dirname(args.ckpt)) or "ckpt",
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval_metadata",
                    default="/mnt/tank/uber/Tactile-Reward/tactile_metadata_eval.h5")
    ap.add_argument("--data_dir_override", default=None)
    ap.add_argument("--peak_min", type=float, default=0.5,
                    help="Lower bound on rewind peak ratio (fraction of trajectory).")
    ap.add_argument("--peak_max", type=float, default=0.7,
                    help="Upper bound on rewind peak ratio.")
    ap.add_argument("--cols", type=int, default=5,
                    help="Number of panels per row in the figure.")
    ap.add_argument("--col_size", type=float, default=3.5)
    ap.add_argument("--row_size", type=float, default=2.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output",
                    default="/mnt/tank/uber/Tactile-Reward/forward_vs_rewind.png")
    ap.add_argument("--json_output", default=None,
                    help="Optional: dump per-task curves as JSON.")
    ap.add_argument("--per_task_dir", default=None,
                    help="Optional: directory to write one PNG per task plus an "
                         "aggregate.png. The grid figure is still produced via --output.")
    args = ap.parse_args()
    main(args)
