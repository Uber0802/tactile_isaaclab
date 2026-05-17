"""Inference-aligned K-sweep eval: simulate the per-step reward signal RL sees.

At inference time the patch fires once per env step. At step K it takes the
buffer's last K valid frames, linspace-samples 16 of them, runs the model, and
uses the LAST slot's predicted progress as the dense reward. The relevant
quantity is therefore:

    reward(K) = model(linspace(0, K-1, 16) frames, text)[15]

This script sweeps K from 16 to N for every held-out episode and plots the
average reward(K) curve, success vs failure, with optional cross-ckpt overlay.

Usage:
    python scripts/eval_k_sweep.py \\
        --ckpts random.pth fulltraj.pth hybrid.pth \\
        --data_dir /mnt/tank/tactile/tactile_dataset/data_3_baseline \\
        --test_files_json /path/to/test_files.json \\
        --task_text_override "grasp peg and insert to another hole" \\
        --output_dir /path/to/out
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


def load_model(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("args", {})
    isaaclab_aligned = bool(cfg.get("isaaclab_aligned", True))
    num_strided = cfg.get("num_strided_layers", 0) or (3 if isaaclab_aligned else 5)
    bimanual_axis = "height" if isaaclab_aligned else "width"
    model = TactileReWiNDTransformer(
        max_length=cfg.get("max_length", 16), text_dim=384,
        hidden_dim=cfg.get("hidden_dim", 512),
        num_heads=cfg.get("num_heads", 8),
        num_layers=cfg.get("num_layers", 4),
        per_hand_dim=cfg.get("per_hand_dim", 384),
        num_strided_layers=num_strided,
        bimanual_axis=bimanual_axis,
        in_channels=cfg.get("in_channels", 3),
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
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    enc = tok([text], padding=True, truncation=True, return_tensors="pt").to(device)
    m = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(device).eval()
    with torch.no_grad():
        out = m(**enc)
        e = mean_pool(out, enc["attention_mask"]).float().squeeze(0)
    del m, tok
    return e


def load_episodes(data_dir: str, include_files: set | None) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True))
    if include_files is not None:
        paths = [p for p in paths
                 if os.path.basename(p) in include_files or os.path.relpath(p) in include_files]
        print(f"  include_files filter: kept {len(paths)}")
    eps = []
    for p in paths:
        try:
            d = np.load(p, allow_pickle=True).item()
        except Exception:
            continue
        tac = d["Tactile"]
        if tac.ndim != 4 or tac.shape[1:] != (40, 25, 3):
            continue
        eps.append({"file": os.path.basename(p), "tactile": tac,
                    "success": int(d["Success"])})
    return eps


def normalize_in_place(frames_thwc: np.ndarray, mode: str):
    if mode == "global":
        denom = np.abs(frames_thwc).max(axis=None, keepdims=True)
    elif mode == "per_channel":
        denom = np.abs(frames_thwc).max(axis=(0, 1, 2), keepdims=True)
    elif mode in ("off", None):
        return frames_thwc
    else:
        raise ValueError(f"unknown normalize_mode={mode!r}")
    np.maximum(denom, 1e-6, out=denom)
    frames_thwc /= denom
    return frames_thwc


def k_sweep_one_episode(tactile: np.ndarray, k_values: np.ndarray, model,
                        text_emb: torch.Tensor, max_length: int,
                        normalize_mode: str, device: torch.device,
                        chunk: int = 64) -> np.ndarray:
    """Return reward(K) of shape (len(k_values),) — model output at slot 15 for
    each K. Batches the K-sweep through the model in chunks to keep mem bounded.

    Each entry samples 16 frame indices via linspace(0, K-1, 16), normalizes the
    sample (matching training), and reads pred[..., 15] as the reward at step K.
    """
    # Pre-build all (16, 3, 40, 25) fp32 tensors for every K, then send in chunks.
    out = np.empty(len(k_values), dtype=np.float32)
    text_seed = text_emb.unsqueeze(0)        # (1, 384)
    with torch.no_grad():
        for c0 in range(0, len(k_values), chunk):
            c_ks = k_values[c0:c0 + chunk]
            batch = np.empty((len(c_ks), max_length, 3, 40, 25), dtype=np.float32)
            for j, K in enumerate(c_ks):
                idx = np.round(np.linspace(0, K - 1, max_length)).astype(np.int64)
                frames = tactile[idx].astype(np.float32, copy=True)   # (16, 40, 25, 3)
                normalize_in_place(frames, normalize_mode)
                # → (16, 3, 40, 25)
                batch[j] = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
            x = torch.from_numpy(batch).to(device)
            txt = text_seed.expand(x.shape[0], -1)
            pred = model(x, txt).squeeze(-1).float().cpu().numpy()      # (chunk, 16)
            out[c0:c0 + chunk] = pred[:, max_length - 1]
    return out


EPOCH_RE = re.compile(r"epoch(\d+)")


def short_tag(ckpt_path: str) -> str:
    """Compact label for plots — pull run name + epoch out of the path."""
    base = os.path.basename(ckpt_path).replace(".pth", "")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--ckpt_labels", nargs="+", default=None,
                    help="One label per ckpt for the comparison plot legend "
                         "(default: stem of the ckpt filename).")
    ap.add_argument("--data_dir",
                    default="/mnt/tank/tactile/tactile_dataset/data_3_baseline")
    ap.add_argument("--test_files_json", default=None)
    ap.add_argument("--include_files", nargs="+", default=None)
    ap.add_argument("--task_text_override", default=None,
                    help="Override the Task string read from each .npy.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--k_min", type=int, default=16)
    ap.add_argument("--k_max", type=int, default=0, help="0 = use episode N.")
    ap.add_argument("--k_step", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=64,
                    help="K-values per model batch.")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.ckpt_labels and len(args.ckpt_labels) != len(args.ckpts):
        raise ValueError("--ckpt_labels length must match --ckpts")
    labels = args.ckpt_labels or [short_tag(c) for c in args.ckpts]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── load episodes once ─────────────────────────────────────────────────
    include = None
    if args.test_files_json:
        with open(args.test_files_json) as f:
            include = set(json.load(f)["test_files"])
        print(f"loaded {len(include)} test filenames from {args.test_files_json}")
    elif args.include_files:
        include = set(args.include_files)
    eps = load_episodes(args.data_dir, include)
    if not eps:
        raise SystemExit("no episodes loaded")
    n_succ = sum(1 for e in eps if e["success"])
    print(f"loaded {len(eps)} episodes ({n_succ} success, {len(eps) - n_succ} fail)")

    if args.task_text_override:
        task_text = args.task_text_override
    else:
        task_text = eps[0]["tactile"].shape and eps[0].get("task") or "grasp peg and insert to another hole"
    print(f"task text: {task_text!r}")
    text_emb = embed_text(task_text, device)

    # All episodes share N within a task; use the first to derive K-grid.
    N = eps[0]["tactile"].shape[0]
    k_max = args.k_max if args.k_max > 0 else N
    k_values = np.arange(args.k_min, min(k_max, N) + 1, args.k_step, dtype=np.int64)
    print(f"K-sweep: {len(k_values)} values from {k_values[0]} to {k_values[-1]} (N={N})")

    # ─── per-ckpt eval ──────────────────────────────────────────────────────
    results: dict[str, dict] = {}
    for ck, label in zip(args.ckpts, labels):
        print(f"\n=== {label}  ({ck})")
        model, cfg = load_model(ck, device)
        max_length = cfg.get("max_length", 16)
        normalize_mode = cfg.get("normalize_mode")
        if normalize_mode is None:
            normalize_mode = "per_channel" if cfg.get("normalize_per_channel") else "off"
        print(f"  max_length={max_length}  normalize_mode={normalize_mode}")
        succ_curves, fail_curves = [], []
        t0 = time.time()
        for i, ep in enumerate(eps):
            curve = k_sweep_one_episode(ep["tactile"], k_values, model, text_emb,
                                        max_length, normalize_mode, device, args.chunk)
            (succ_curves if ep["success"] else fail_curves).append(curve)
            if (i + 1) % 20 == 0 or i + 1 == len(eps):
                print(f"  episode {i + 1}/{len(eps)}  elapsed={time.time() - t0:.1f}s")
        succ_arr = np.stack(succ_curves, 0) if succ_curves else np.zeros((0, len(k_values)))
        fail_arr = np.stack(fail_curves, 0) if fail_curves else np.zeros((0, len(k_values)))
        results[label] = {
            "ckpt": ck, "normalize_mode": normalize_mode,
            "k_values": k_values.tolist(),
            "n_succ": succ_arr.shape[0], "n_fail": fail_arr.shape[0],
            "succ_mean": succ_arr.mean(axis=0).tolist() if succ_arr.size else None,
            "succ_std":  succ_arr.std(axis=0).tolist() if succ_arr.size else None,
            "fail_mean": fail_arr.mean(axis=0).tolist() if fail_arr.size else None,
            "fail_std":  fail_arr.std(axis=0).tolist() if fail_arr.size else None,
        }

    # ─── per-ckpt single plot ───────────────────────────────────────────────
    for label, r in results.items():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        k = np.array(r["k_values"])
        if r["succ_mean"] is not None:
            m, s = np.array(r["succ_mean"]), np.array(r["succ_std"])
            ax.plot(k, m, color="tab:green", lw=2, label=f"success (n={r['n_succ']})")
            ax.fill_between(k, m - s, m + s, color="tab:green", alpha=0.18)
        if r["fail_mean"] is not None:
            m, s = np.array(r["fail_mean"]), np.array(r["fail_std"])
            ax.plot(k, m, color="tab:red", lw=2, label=f"fail (n={r['n_fail']})")
            ax.fill_between(k, m - s, m + s, color="tab:red", alpha=0.18)
        ax.plot(k, k / N, color="gray", ls="--", lw=1, label="ideal K/N")
        ax.set_xlabel("step K within episode (inference)")
        ax.set_ylabel("predicted reward at step K  (pred[:, 15])")
        ax.set_title(f"{label}: reward(K) on N={N} eval episodes")
        ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3); ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"{label}_kcurve.png"), dpi=130)
        plt.close(fig)

    # ─── cross-ckpt comparison plot ─────────────────────────────────────────
    if len(results) > 1:
        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / max(len(results) - 1, 1)) for i in range(len(results))]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        k = np.array(next(iter(results.values()))["k_values"])
        for (label, r), c in zip(results.items(), colors):
            if r["succ_mean"] is not None:
                axes[0].plot(k, r["succ_mean"], color=c, lw=2, label=label)
            if r["fail_mean"] is not None:
                axes[1].plot(k, r["fail_mean"], color=c, lw=2, label=label)
        axes[0].plot(k, k / N, color="gray", ls="--", lw=1, label="ideal K/N")
        for ax, title in zip(axes, ("success episodes", "failure episodes")):
            ax.set_xlabel("step K within episode")
            ax.set_title(title)
            ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("predicted reward at step K")
        axes[-1].legend(loc="upper right", fontsize=9)
        fig.suptitle(f"K-sweep comparison (N={N}, {len(eps)} held-out episodes)")
        fig.tight_layout()
        cmp_path = os.path.join(args.output_dir, "kcurve_compare.png")
        fig.savefig(cmp_path, dpi=130)
        plt.close(fig)
        print(f"\nwrote comparison: {cmp_path}")

    out_json = os.path.join(args.output_dir, "k_sweep_results.json")
    with open(out_json, "w") as f:
        json.dump({"k_values": results[labels[0]]["k_values"],
                   "N": N, "task_text": task_text,
                   "results": results}, f, indent=2)
    print(f"wrote: {out_json}")


if __name__ == "__main__":
    main()
