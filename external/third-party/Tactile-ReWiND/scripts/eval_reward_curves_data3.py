"""Eval reward curves on the IsaacLab Forge `data_3_baseline` rollout set.

Each .npy file is a pickled dict written by `forge_env._flush_tactile_episode`:
    {"Task": str,
     "Tactile": (T, 40, 25, 3) fp16, channels (Fz, Fx, Fy),
     "Success": 0 or 1}

For every episode the model predicts a per-frame reward curve (using max_length
sub-sampled frames of the forward playback). The per-frame mean is then taken
**separately over Success=1 and Success=0** episodes, so we can see whether the
reward model distinguishes the two.

The script accepts multiple `--ckpts` so that the .npy files only need to be
loaded from NFS once — each ckpt is then evaluated against the already-loaded
tensors.

Usage:
    python scripts/eval_reward_curves_data3.py \\
        --data_dir /mnt/tank/tactile/tactile_dataset/data_3_baseline \\
        --ckpts /mnt/lab-tank/uber/Tactile-Reward/checkpoints_3ch/tactile_rewind_epoch10.pth \\
                /mnt/lab-tank/uber/Tactile-Reward/checkpoints_3ch/tactile_rewind_epoch15.pth \\
                /mnt/lab-tank/uber/Tactile-Reward/checkpoints_3ch/tactile_rewind_epoch19.pth \\
        --output_dir /mnt/lab-tank/uber/Tactile-Reward/eval_curves_data3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.tactile_model import TactileReWiNDTransformer


EPOCH_RE = re.compile(r"epoch(\d+)")


def epoch_from_path(p: str) -> int:
    m = EPOCH_RE.search(os.path.basename(p))
    return int(m.group(1)) if m else -1


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
        in_channels=cfg.get("in_channels", 2),
    ).to(device)
    sd = state["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, cfg


def mean_pool(out, mask):
    tok = out[0]
    m = mask.unsqueeze(-1).expand(tok.size()).float()
    return (tok * m).sum(1) / m.sum(1).clamp(min=1e-9)


def embed_text(text: str, device: torch.device) -> torch.Tensor:
    """Match the embedding pipeline used by anytouch2_to_h5.py."""
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    enc = tok([text], padding=True, truncation=True, return_tensors="pt").to(device)
    m = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(device).eval()
    with torch.no_grad():
        out = m(**enc)
        e = mean_pool(out, enc["attention_mask"]).float().squeeze(0)
    del m, tok
    return e   # (384,)


def forward_indices(T: int, max_length: int) -> np.ndarray:
    if T >= max_length:
        return np.round(np.linspace(0, T - 1, max_length)).astype(np.int64)
    pad = max_length - T
    return np.concatenate([np.arange(T, dtype=np.int64),
                           np.full(pad, T - 1, dtype=np.int64)])


def load_episodes(data_dirs: list[str], limit: int | None,
                  include_files: set[str] | None = None) -> list[dict]:
    paths: list[str] = []
    for d in data_dirs:
        # Recursive globbing handles flat dirs AND curriculum-style ep_XXX subdirs.
        paths.extend(glob.glob(os.path.join(d, "**", "*.npy"), recursive=True))
    paths = sorted(paths)
    if include_files is not None:
        # Match on basename OR the path relative to any of the data_dirs, so
        # held-out lists written by finetune_data3.py (which use relpath) work.
        rels = {os.path.relpath(p) for p in paths}
        paths = [p for p in paths
                 if os.path.basename(p) in include_files or os.path.relpath(p) in include_files]
        print(f"  include_files filter: keeping {len(paths)}")
    if limit:
        paths = paths[:limit]
    eps = []
    t0 = time.time()
    for i, p in enumerate(paths):
        try:
            d = np.load(p, allow_pickle=True).item()
        except Exception as e:
            print(f"  skip {p}: {e}", file=sys.stderr)
            continue
        tac = d["Tactile"]
        if tac.ndim != 4 or tac.shape[1:] != (40, 25, 3):
            print(f"  skip {p}: unexpected shape {tac.shape}", file=sys.stderr)
            continue
        eps.append({
            "name": os.path.basename(p),
            "task": d["Task"],
            "success": int(d["Success"]),
            "tactile": tac,            # (T, 40, 25, 3) fp16
        })
        if (i + 1) % 200 == 0 or i + 1 == len(paths):
            print(f"  loaded {i + 1}/{len(paths)} ({time.time() - t0:.1f}s)")
    print(f"loaded {len(eps)} episodes in {time.time() - t0:.1f}s")
    return eps


def frames_to_tensor(tac: np.ndarray, idx: np.ndarray, device: torch.device,
                     scale: float = 1.0,
                     normalize_mode: str = "off") -> torch.Tensor:
    """Tactile (T_full, 40, 25, 3) fp16 + frame indices → (1, T, 3, 40, 25) fp32 on device.

    No channel swap, no spatial alignment — the data is already in IsaacLab
    (Fz, Fx, Fy) at (40, 25) bimanual-H layout. Optional `scale` multiplier
    bridges magnitude gaps vs the training distribution; `normalize_mode`
    mirrors the dataset normalization mode the ckpt was trained with.
    """
    sub = tac[idx].astype(np.float32, copy=True)            # (T, 40, 25, 3)
    if scale != 1.0:
        sub *= scale
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
    x = torch.from_numpy(sub).permute(0, 3, 1, 2).contiguous()  # (T, 3, 40, 25)
    return x.unsqueeze(0).to(device)


def eval_ckpt(ckpt: str, eps: list[dict], text_emb_cache: dict[str, torch.Tensor],
              device: torch.device, output_dir: str, input_scale: float = 1.0):
    model, cfg = load_model(ckpt, device)
    max_length = cfg.get("max_length", 16)
    in_channels = cfg.get("in_channels", 2)
    # Read normalization mode; fall back to legacy boolean for old ckpts.
    normalize_mode = cfg.get("normalize_mode")
    if normalize_mode is None:
        normalize_mode = "per_channel" if cfg.get("normalize_per_channel") else "off"
    scale_tag = "" if input_scale == 1.0 else f"  input_scale={input_scale:g}"
    norm_tag = "" if normalize_mode == "off" else f"  normalize={normalize_mode}"
    print(f"\n=== ckpt: {os.path.basename(ckpt)} "
          f"(max_length={max_length}, in_channels={in_channels})"
          f"{norm_tag}{scale_tag} ===")
    if in_channels != 3:
        print(f"  WARNING: ckpt has in_channels={in_channels}; data_3_baseline is "
              f"(Fz,Fx,Fy) 3-channel. Channels won't match.", file=sys.stderr)

    curves_by_success: dict[int, list[np.ndarray]] = defaultdict(list)

    for ep in eps:
        task = ep["task"]
        if task not in text_emb_cache:
            text_emb_cache[task] = embed_text(task, device)
        text = text_emb_cache[task].unsqueeze(0)            # (1, 384)
        idx = forward_indices(ep["tactile"].shape[0], max_length)
        x = frames_to_tensor(ep["tactile"], idx, device, scale=input_scale,
                             normalize_mode=normalize_mode)
        with torch.no_grad():
            pred = model(x, text).squeeze(-1).squeeze(0).float().cpu().numpy()
        curves_by_success[ep["success"]].append(pred)

    n_succ = len(curves_by_success[1])
    n_fail = len(curves_by_success[0])
    print(f"  n_success={n_succ}  n_fail={n_fail}")

    summary = {"ckpt": ckpt, "input_scale": input_scale, "max_length": max_length,
               "n_success": n_succ, "n_fail": n_fail,
               "frames": list(range(max_length))}
    for k, label in [(1, "success"), (0, "fail")]:
        if curves_by_success[k]:
            arr = np.stack(curves_by_success[k], axis=0)
            summary[f"{label}_mean"] = arr.mean(axis=0).tolist()
            summary[f"{label}_std"] = arr.std(axis=0).tolist()
            print(f"  {label}: final_mean={arr.mean(axis=0)[-1]:.3f}  "
                  f"peak_mean={arr.mean(axis=0).max():.3f}")
        else:
            summary[f"{label}_mean"] = None
            summary[f"{label}_std"] = None

    # Per-ckpt PNG (suffix the scale into the filename when > 1.0).
    os.makedirs(output_dir, exist_ok=True)
    stem_base = os.path.splitext(os.path.basename(ckpt))[0]
    stem = stem_base if input_scale == 1.0 else f"{stem_base}_scale{input_scale:g}"
    png = os.path.join(output_dir, f"{stem}_curves.png")
    json_p = os.path.join(output_dir, f"{stem}_curves.json")
    frames = np.arange(max_length)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, label, color in [(1, "success", "tab:green"), (0, "fail", "tab:red")]:
        if summary[f"{label}_mean"] is None:
            continue
        mean = np.array(summary[f"{label}_mean"])
        std = np.array(summary[f"{label}_std"])
        n = summary[f"n_{label}"]
        ax.plot(frames, mean, color=color, linewidth=2, label=f"{label} (n={n})")
        ax.fill_between(frames, mean - std, mean + std, color=color, alpha=0.15)
    ax.plot(frames, np.linspace(0, 1, max_length), color="gray", linestyle="--",
            linewidth=1, label="ideal forward ramp")
    ax.set_xlabel("frame index")
    ax.set_ylabel("predicted reward")
    ax.set_title(f"{stem} on data_3_baseline (n_succ={n_succ}, n_fail={n_fail})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"  wrote {png}")
    with open(json_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {json_p}")
    return summary


def make_combined_plot(summaries: list[dict], output_dir: str):
    """Side-by-side success/fail curves with one line per ckpt."""
    runs = sorted(summaries, key=lambda s: epoch_from_path(s["ckpt"]))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(runs) - 1, 1)) for i in range(len(runs))]

    max_len = runs[0]["max_length"]
    frames = np.arange(max_len)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    panels = [("success forward", "success_mean", "n_success"),
              ("failure forward", "fail_mean", "n_fail")]
    for ax, (title, key, n_key) in zip(axes, panels):
        for r, c in zip(runs, colors):
            if r.get(key) is None:
                continue
            ax.plot(frames, np.array(r[key]), color=c, linewidth=2,
                    label=f"epoch {epoch_from_path(r['ckpt'])} (n={r[n_key]})")
        if key == "success_mean":
            ax.plot(frames, np.linspace(0, 1, max_len), color="gray",
                    linestyle="--", linewidth=1, label="ideal ramp")
        ax.set_title(title)
        ax.set_xlabel("frame index")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("predicted reward")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle("data_3_baseline: success vs fail across checkpoints", fontsize=12)
    fig.tight_layout()
    combined = os.path.join(output_dir, "epoch_compare.png")
    fig.savefig(combined, dpi=130)
    plt.close(fig)
    print(f"\nwrote combined: {combined}")


def make_scale_sweep_plot(summaries: list[dict], output_dir: str):
    """For sweeps over input_scale: plot final-frame success/fail mean (and gap)
    against the scale factor, with one curve per checkpoint."""
    by_ckpt: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        by_ckpt[s["ckpt"]].append(s)
    for v in by_ckpt.values():
        v.sort(key=lambda s: s["input_scale"])

    cmap = plt.get_cmap("viridis")
    epochs = sorted({epoch_from_path(c) for c in by_ckpt})
    color_for_epoch = {e: cmap(i / max(len(epochs) - 1, 1)) for i, e in enumerate(epochs)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ck, runs in by_ckpt.items():
        ep = epoch_from_path(ck)
        c = color_for_epoch[ep]
        scales = [r["input_scale"] for r in runs]
        succ = [r["success_mean"][-1] for r in runs]
        fail = [r["fail_mean"][-1] for r in runs]
        gap = [s - f for s, f in zip(succ, fail)]
        axes[0].plot(scales, succ, color=c, marker="o", linewidth=2,
                     label=f"epoch {ep} success")
        axes[0].plot(scales, fail, color=c, marker="x", linewidth=2,
                     linestyle="--", label=f"epoch {ep} fail")
        axes[1].plot(scales, gap, color=c, marker="o", linewidth=2,
                     label=f"epoch {ep}")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("input scale (multiplier on data_3 tactile)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("final-frame mean reward")
    axes[0].set_title("success (solid) / fail (dashed) vs scale")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_ylabel("success_final − fail_final")
    axes[1].set_title("success-vs-fail gap (>0 = model distinguishes)")
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("data_3_baseline: input-scale sweep", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "scale_sweep.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nwrote sweep plot: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+",
                    default=["/mnt/tank/tactile/tactile_dataset/data_3_baseline"],
                    help="One or more dataset dirs (recursively globbed).")
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="One or more checkpoint paths.")
    ap.add_argument("--output_dir",
                    default="/mnt/lab-tank/uber/Tactile-Reward/eval_curves_data3")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit number of episodes (smoke test). 0 = all.")
    ap.add_argument("--input_scales", type=float, nargs="+", default=[1.0],
                    help="Multipliers applied to the data_3 tactile input before "
                         "feeding the model. Pass several to sweep, e.g. "
                         "`--input_scales 1 50 100 200 500 1000`.")
    ap.add_argument("--test_files_json", default=None,
                    help="Path to a JSON written by finetune_data3.py with a "
                         "'test_files' key. When set, eval only on the listed "
                         "held-out files. Overrides --include_files.")
    ap.add_argument("--include_files", nargs="+", default=None,
                    help="Explicit list of episode basenames (e.g. ep17.npy) to "
                         "include. Useful for custom subsets.")
    ap.add_argument("--task_text_override", default=None,
                    help="Replace the 'Task' string read from each .npy with this "
                         "instruction. Required when feeding non-peg data to a "
                         "multi-task model (the dataset's stored Task field is "
                         "hardcoded to the peg instruction regardless of source).")
    args = ap.parse_args()

    include_files: set[str] | None = None
    if args.test_files_json:
        with open(args.test_files_json) as f:
            include_files = set(json.load(f)["test_files"])
        print(f"loaded {len(include_files)} test filenames from {args.test_files_json}")
    elif args.include_files:
        include_files = set(args.include_files)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, data_dirs={args.data_dirs}, output_dir={args.output_dir}")
    print(f"input_scales={args.input_scales}")

    eps = load_episodes(args.data_dirs, args.limit if args.limit > 0 else None,
                        include_files=include_files)
    if not eps:
        raise SystemExit("no episodes loaded")
    n_succ = sum(1 for e in eps if e["success"])
    n_fail = len(eps) - n_succ
    print(f"  → {n_succ} success, {n_fail} fail")

    if args.task_text_override:
        print(f"override Task: replacing every episode's task with "
              f"{args.task_text_override!r}")
        for ep in eps:
            ep["task"] = args.task_text_override
    text_emb_cache: dict[str, torch.Tensor] = {}
    summaries: list[dict] = []
    for scale in args.input_scales:
        for ck in args.ckpts:
            summaries.append(eval_ckpt(ck, eps, text_emb_cache, device,
                                       args.output_dir, input_scale=scale))
    # Multi-ckpt comparison at scale=1.0 (the original combined plot).
    base = [s for s in summaries if s["input_scale"] == 1.0]
    if len(base) > 1:
        make_combined_plot(base, args.output_dir)
    # Scale sweep plot whenever > 1 scale was tried.
    if len(args.input_scales) > 1:
        make_scale_sweep_plot(summaries, args.output_dir)


if __name__ == "__main__":
    main()
