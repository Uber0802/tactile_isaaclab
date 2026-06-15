"""Per-frame reward curves for forward vs rewind playback.

For every eval trajectory, runs the model twice — once on the trajectory
played **forward** (i.e. the "successful" playback the model should reward
with a rising 0→1 ramp), and once on a **rewind** trajectory (forward then
reverse, which a well-trained model should reward as up-then-down). The
two curves are the per-frame mean of the model's predicted reward across
all eval trajectories.

Usage:
    python scripts/eval_reward_curves.py \\
        --ckpt /mnt/lab-tank/uber/Tactile-Reward/checkpoints_3ch/tactile_rewind_epoch19.pth \\
        --eval_metadata /mnt/lab-tank/uber/Tactile-Reward/3ch/tactile_metadata_eval.h5 \\
        --output_dir /mnt/lab-tank/uber/Tactile-Reward/eval_curves
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.tactile_model import TactileReWiNDTransformer
from tools.tactile_dataset import TactileReWiNDDataset


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
        in_channels=cfg.get("in_channels", 2),
    ).to(device)
    # Tolerate state_dicts saved with a `_orig_mod.` prefix (torch.compile).
    sd = state["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, cfg


def resize_to_length(idx: np.ndarray, max_length: int) -> np.ndarray:
    """Match the training-time _resize_indices: linspace-subsample if longer,
    pad with the last index if shorter."""
    T = idx.shape[0]
    if T < max_length:
        pad = max_length - T
        return np.concatenate([idx, np.full(pad, idx[-1], dtype=idx.dtype)])
    if T > max_length:
        local = np.linspace(0, T - 1, max_length).astype(int)
        return idx[local]
    return idx


def build_forward_idx(N: int, max_length: int) -> np.ndarray:
    return resize_to_length(np.arange(N, dtype=np.int64), max_length)


def build_rewind_idx(N: int, max_length: int) -> np.ndarray:
    """Forward then reverse (matches training _sample_rewind with start=0,
    end=N, rev_end=fwd_len). Total length before resize = 2N - 1."""
    fwd = np.arange(N, dtype=np.int64)
    rev = np.arange(N - 2, -1, -1, dtype=np.int64)  # N-2, N-3, ..., 0
    full = np.concatenate([fwd, rev])
    return resize_to_length(full, max_length)


def frames_to_tensor(traj: np.ndarray, traj_idx: np.ndarray, *,
                     isaaclab_aligned: bool, data_already_aligned: bool,
                     in_channels: int, device: torch.device,
                     normalize_mode: str = "off") -> torch.Tensor:
    """(N, H, W, C) fp16 mmap + index list → (1, T, C, H', W') fp32 on device.

    Mirrors the dataset's two code paths:
      * data_already_aligned=True OR not isaaclab_aligned → plain THWC → CHW
      * isaaclab_aligned and not yet aligned → spatial sub-sample to (40, 25)
    Channel swap [2,0,1] is applied for C=3 to match training. Optional
    per-channel max-abs normalize matches training-time preprocessing.
    """
    do_spatial_align = isaaclab_aligned and not data_already_aligned
    if do_spatial_align:
        # Same orthogonal index math as TactileReWiNDDataset._to_aligned_frames.
        target_h, target_w = 20, 25  # eval always uses the canonical resolution
        H_in, W_in, C = 320, 480, traj.shape[-1]
        half_w = W_in // 2
        h_local = np.linspace(0, half_w - 1, target_h).round().astype(np.int64)
        w_local = np.linspace(0, H_in - 1, target_w).round().astype(np.int64)
        h_full = np.concatenate([h_local, h_local + half_w])
        sub = traj[np.ix_(traj_idx, w_local, h_full, np.arange(C))]
        # (T, target_w, 2*target_h, C) → (T, 2*target_h, target_w, C)
        sub = np.ascontiguousarray(sub.transpose(0, 2, 1, 3)).astype(np.float32, copy=False)
    else:
        sub = traj[traj_idx].astype(np.float32, copy=True)
    if sub.shape[-1] == 3:
        sub = sub[..., [2, 0, 1]]
    if normalize_mode == "global":
        denom = np.abs(sub).max(axis=None, keepdims=True)
        np.maximum(denom, 1e-6, out=denom)
        sub = sub / denom
    elif normalize_mode == "per_channel":
        denom = np.abs(sub).max(axis=(0, 1, 2), keepdims=True)
        np.maximum(denom, 1e-6, out=denom)
        sub = sub / denom
    elif normalize_mode != "off":
        raise ValueError(f"unknown normalize_mode={normalize_mode!r}")
    x = torch.from_numpy(sub).permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
    return x.unsqueeze(0).to(device)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)
    max_length = cfg.get("max_length", 16)
    isaaclab_aligned = bool(cfg.get("isaaclab_aligned", False))
    in_channels = cfg.get("in_channels", 2)
    # Read normalization mode; fall back to legacy boolean for old ckpts.
    normalize_mode = cfg.get("normalize_mode")
    if normalize_mode is None:
        normalize_mode = "per_channel" if cfg.get("normalize_per_channel") else "off"
    print(f"loaded {args.ckpt}")
    print(f"  max_length={max_length}, isaaclab_aligned={isaaclab_aligned}, "
          f"in_channels={in_channels}, normalize_mode={normalize_mode}")

    with h5py.File(args.eval_metadata, "r") as h5:
        data_dir = args.data_dir_override or h5.attrs["data_dir"]
        data_already_aligned = bool(h5.attrs.get("data_already_aligned", False))
        tasks = sorted(h5.keys())
        lang = {t: np.asarray(h5[t]["minilm_lang_embedding"], dtype=np.float32)
                for t in tasks}
        files = {t: [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                     for s in np.asarray(h5[t]["trajectory_files"])]
                 for t in tasks}
    print(f"eval set: {len(tasks)} tasks, data_dir={data_dir}, "
          f"data_already_aligned={data_already_aligned}")

    fwd_rewards: list[np.ndarray] = []
    rev_rewards: list[np.ndarray] = []
    mis_rewards: list[np.ndarray] = []
    per_traj_records = []
    skipped = 0

    # Deterministic mismatched-language pairing: each task uses the *next*
    # task's language embedding. Stable across runs without an RNG.
    task_order = list(tasks)
    mismatched_for = {t: task_order[(i + 1) % len(task_order)]
                      for i, t in enumerate(task_order)}

    for task in tasks:
        if not files[task]:
            continue
        text = torch.from_numpy(lang[task][0]).float().unsqueeze(0).to(device)
        mismatched_text = torch.from_numpy(
            lang[mismatched_for[task]][0]).float().unsqueeze(0).to(device)
        for fname in files[task]:
            path = os.path.join(data_dir, fname)
            traj = np.load(path, mmap_mode="r")
            N = len(traj)
            if N < 3:
                skipped += 1
                continue

            fwd_idx = build_forward_idx(N, max_length)
            rev_idx = build_rewind_idx(N, max_length)

            x_fwd = frames_to_tensor(traj, fwd_idx,
                                     isaaclab_aligned=isaaclab_aligned,
                                     data_already_aligned=data_already_aligned,
                                     in_channels=in_channels, device=device,
                                     normalize_mode=normalize_mode)
            x_rev = frames_to_tensor(traj, rev_idx,
                                     isaaclab_aligned=isaaclab_aligned,
                                     data_already_aligned=data_already_aligned,
                                     in_channels=in_channels, device=device,
                                     normalize_mode=normalize_mode)
            with torch.no_grad():
                pred_fwd = model(x_fwd, text).squeeze(-1).squeeze(0).float().cpu().numpy()
                pred_rev = model(x_rev, text).squeeze(-1).squeeze(0).float().cpu().numpy()
                # Same forward video, wrong language → should be ~0 if language conditioning works.
                pred_mis = model(x_fwd, mismatched_text).squeeze(-1).squeeze(0).float().cpu().numpy()

            fwd_rewards.append(pred_fwd)
            rev_rewards.append(pred_rev)
            mis_rewards.append(pred_mis)
            per_traj_records.append({
                "task": task, "file": fname, "N_frames": int(N),
                "mismatched_task": mismatched_for[task],
                "forward_curve": pred_fwd.tolist(),
                "rewind_curve": pred_rev.tolist(),
                "mismatched_curve": pred_mis.tolist(),
            })
            print(f"  [{len(fwd_rewards):3d}] {task}/{fname} N={N} "
                  f"fwd={pred_fwd.mean():.3f} rev={pred_rev.mean():.3f} "
                  f"mis={pred_mis.mean():.3f}")

    if not fwd_rewards:
        raise RuntimeError("no eval trajectories produced a curve")

    fwd_arr = np.stack(fwd_rewards, axis=0)   # (n_traj, max_length)
    rev_arr = np.stack(rev_rewards, axis=0)
    mis_arr = np.stack(mis_rewards, axis=0)
    fwd_mean = fwd_arr.mean(axis=0); fwd_std = fwd_arr.std(axis=0)
    rev_mean = rev_arr.mean(axis=0); rev_std = rev_arr.std(axis=0)
    mis_mean = mis_arr.mean(axis=0); mis_std = mis_arr.std(axis=0)

    n_traj = fwd_arr.shape[0]
    print()
    print("=" * 60)
    print(f"n_trajectories = {n_traj} (skipped {skipped})")
    print(f"forward    curve: min={fwd_mean.min():.3f} max={fwd_mean.max():.3f} final={fwd_mean[-1]:.3f}")
    print(f"rewind     curve: min={rev_mean.min():.3f} max={rev_mean.max():.3f} final={rev_mean[-1]:.3f}")
    print(f"mismatched curve: min={mis_mean.min():.3f} max={mis_mean.max():.3f} final={mis_mean[-1]:.3f}")
    print(f"  language gap (forward_final - mismatched_final): "
          f"{fwd_mean[-1] - mis_mean[-1]:+.3f}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    png_path = os.path.join(args.output_dir, f"{stem}_curves.png")
    json_path = os.path.join(args.output_dir, f"{stem}_curves.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = np.arange(max_length)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frames, fwd_mean, color="tab:green", label=f"forward (correct lang), n={n_traj}", linewidth=2)
    ax.fill_between(frames, fwd_mean - fwd_std, fwd_mean + fwd_std,
                    color="tab:green", alpha=0.18)
    ax.plot(frames, rev_mean, color="tab:red", label="rewind (fwd→rev)", linewidth=2)
    ax.fill_between(frames, rev_mean - rev_std, rev_mean + rev_std,
                    color="tab:red", alpha=0.18)
    ax.plot(frames, mis_mean, color="tab:purple", label="forward + mismatched lang",
            linewidth=2, linestyle=":")
    ax.fill_between(frames, mis_mean - mis_std, mis_mean + mis_std,
                    color="tab:purple", alpha=0.12)
    ax.plot(frames, np.linspace(0, 1, max_length), color="gray", linestyle="--",
            linewidth=1, label="ideal forward ramp")
    ax.set_xlabel("frame index")
    ax.set_ylabel("predicted reward")
    ax.set_title(f"{stem}: per-frame mean reward over {n_traj} eval trajectories")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"wrote plot:  {png_path}")

    with open(json_path, "w") as fh:
        json.dump({
            "ckpt": args.ckpt,
            "eval_metadata": args.eval_metadata,
            "max_length": max_length,
            "n_trajectories": n_traj,
            "skipped": skipped,
            "frames": frames.tolist(),
            "forward_mean": fwd_mean.tolist(),
            "forward_std": fwd_std.tolist(),
            "rewind_mean": rev_mean.tolist(),
            "rewind_std": rev_std.tolist(),
            "mismatched_mean": mis_mean.tolist(),
            "mismatched_std": mis_std.tolist(),
            "per_trajectory": per_traj_records,
        }, fh, indent=2)
    print(f"wrote json:  {json_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval_metadata",
                    default="/mnt/lab-tank/uber/Tactile-Reward/3ch/tactile_metadata_eval.h5")
    ap.add_argument("--data_dir_override", default=None)
    ap.add_argument("--output_dir",
                    default="/mnt/lab-tank/uber/Tactile-Reward/eval_curves")
    args = ap.parse_args()
    main(args)
