"""Sanity-check generalisation: run the AnyTouch2-trained reward model on
your own ep*.npy tactile dataset.

Your dataset shape: (T, 40, 25, 3) float16
  - 3 channels = (Fx, Fy, |F|) — magnitude is dropped, only shear is kept
  - 40 x 25 spatial grid (vs AnyTouch2's 320 x 480 — the model's encoder ends
    in AdaptiveAvgPool2d, so spatial size is OK; just not great)

The CNN encoder splits its input along the width dimension into "left" and
"right" hands, so we duplicate the single-hand 40x25 view along width to
mimic the bimanual setup. This is hacky but the right thing to do for a
first sanity check; if it works at all, retrain a single-hand variant.

For each ep*.npy we report predicted progress curves (and Pearson correlation
with a linear ramp, as a *very* weak proxy for "looks like progress"). No
ground truth = no real metric here — eyeball the curves.

Usage:
    python scripts/eval_cross_dataset.py \\
        --ckpt /mnt/tank/uber/Tactile-Reward/checkpoints/tactile_rewind_epoch19.pth \\
        --data_dir /mnt/home/uber/tactile_isaaclab/tactile_dataset \\
        --instruction "grasp peg and insert to another hole" \\
        --output_csv /mnt/tank/uber/Tactile-Reward/cross_dataset_progress.csv
"""
from __future__ import annotations

import os
import re
import sys
import csv
import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.tactile_model import TactileReWiNDTransformer


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode_minilm_one(text: str, device: torch.device) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    minilm = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(device)
    minilm.eval()
    with torch.no_grad():
        enc = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(device)
        out = minilm(**enc)
        emb = mean_pooling(out, enc["attention_mask"]).float().squeeze(0)
    return emb


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


def adapt_frames(traj: np.ndarray, channels: tuple[int, int],
                 single_hand: bool, isaaclab_aligned: bool) -> np.ndarray:
    """Adapt a user IsaacLab trajectory to whatever shape the model expects.

    Input: (T, 40, 25, 3)  — channel order (normal, Fx, Fy) per forge_env.
    Output:
      * isaaclab_aligned=True:  (T, 40, 25, 2)         — model trained on this exact layout.
      * isaaclab_aligned=False: (T, 40, 50, 2)         — width-duplicated bimanual hack.
    """
    selected = traj[..., list(channels)]   # (T, H, W, 2)
    if not isaaclab_aligned and single_hand:
        # Old AnyTouch2-style model: needs bimanual-along-W hack.
        selected = np.concatenate([selected, selected], axis=2)
    return np.ascontiguousarray(selected, dtype=np.float32)


def subsample_to_max_length(traj: np.ndarray, max_length: int) -> np.ndarray:
    T = len(traj)
    if T <= max_length:
        idx = np.arange(T)
        pad = max_length - T
        if pad > 0:
            idx = np.concatenate([idx, np.full(pad, T - 1, dtype=int)])
    else:
        idx = np.linspace(0, T - 1, max_length).astype(int)
    return np.ascontiguousarray(traj[idx])


def to_chw_tensor(frames_thwc: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(frames_thwc)
    x = x.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
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


def episode_idx(filename: str) -> int:
    m = re.match(r"^ep(\d+)\.npy$", filename)
    return int(m.group(1)) if m else -1


def run_subset(model, files, args, text, text_neg, max_length, device,
               isaaclab_aligned: bool):
    """Run model on a list of files. Returns (rows, correct_curves, mismatched_curves).

    correct_curves / mismatched_curves are float32 arrays of shape (n_traj, max_length-1)
    holding the per-frame predicted progress with the leading frame dropped.
    """
    rows = []
    correct_curves: list[np.ndarray] = []
    mismatched_curves: list[np.ndarray] = []

    for fname in tqdm(files, desc="trajectories"):
        path = os.path.join(args.data_dir, fname)
        traj = np.load(path, mmap_mode="r")
        if traj.ndim != 4 or traj.shape[-1] < max(args.shear_channels) + 1:
            print(f"  skipping {fname}: unexpected shape {traj.shape}")
            continue

        frames = adapt_frames(np.array(traj), tuple(args.shear_channels),
                              args.single_hand, isaaclab_aligned)
        frames = subsample_to_max_length(frames, max_length)
        x = to_chw_tensor(frames, device)

        with torch.no_grad():
            pred = model(x, text).squeeze(-1).squeeze(0).cpu().numpy()
            pred_neg = (
                model(x, text_neg).squeeze(-1).squeeze(0).cpu().numpy()
                if text_neg is not None else None
            )

        pred = pred[1:]
        ramp = np.linspace(0, 1, len(pred), dtype=np.float32)
        corr = pearson(pred, ramp)
        correct_curves.append(pred.astype(np.float32))

        row = {
            "file": fname,
            "ep_idx": episode_idx(fname),
            "n_frames": int(traj.shape[0]),
            "mean_progress": float(pred.mean()),
            "pearson_vs_ramp": corr,
        }
        if pred_neg is not None:
            pred_neg = pred_neg[1:]
            mismatched_curves.append(pred_neg.astype(np.float32))
            row["mean_progress_mismatched"] = float(pred_neg.mean())
            row["language_gap"] = float(pred.mean() - pred_neg.mean())
        rows.append(row)

    correct_arr = np.stack(correct_curves) if correct_curves else np.zeros((0, max_length - 1))
    mismatched_arr = np.stack(mismatched_curves) if mismatched_curves else None
    return rows, correct_arr, mismatched_arr


def plot_subset_curves(results: dict, out_path: str):
    """Mean-progress curve per subset (correct=solid, mismatched=dashed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6), dpi=120)
    palette = plt.get_cmap("tab10").colors
    for i, (label, (rows, c_arr, m_arr)) in enumerate(results.items()):
        if c_arr.size == 0:
            continue
        T = c_arr.shape[1]
        x = np.arange(1, T + 1)
        c_mean = c_arr.mean(axis=0)
        c_std = c_arr.std(axis=0)
        color = palette[i % len(palette)]
        ax.plot(x, c_mean, "-", color=color, lw=2.2,
                label=f"{label}  correct  (n={len(rows)}, mean={c_mean.mean():.3f})")
        ax.fill_between(x, c_mean - c_std, c_mean + c_std, color=color, alpha=0.15)
        if m_arr is not None and m_arr.size > 0:
            m_mean = m_arr.mean(axis=0)
            ax.plot(x, m_mean, "--", color=color, lw=1.5,
                    label=f"{label}  mismatched  (mean={m_mean.mean():.3f})")

    ax.set_xlabel("frame index (skipping frame 0)")
    ax.set_ylabel("mean predicted progress")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    ax.set_title("Cross-dataset average progress curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved comparison plot to {out_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_model(args.ckpt, device)
    max_length = cfg.get("max_length", 16)
    isaaclab_aligned = bool(cfg.get("isaaclab_aligned", False))
    print(f"loaded {args.ckpt} (max_length={max_length}, "
          f"isaaclab_aligned={isaaclab_aligned})")

    text = encode_minilm_one(args.instruction, device).unsqueeze(0)
    print(f"instruction: {args.instruction!r}")
    if args.mismatched_instruction:
        text_neg = encode_minilm_one(args.mismatched_instruction, device).unsqueeze(0)
        print(f"mismatched : {args.mismatched_instruction!r}")
    else:
        text_neg = None

    # Numerical sort (ep0, ep1, ep2, ..., ep1749) — not lexicographic.
    files = sorted(
        (f for f in os.listdir(args.data_dir) if re.match(r"^ep\d+\.npy$", f)),
        key=episode_idx,
    )
    if not files:
        raise SystemExit(f"no ep*.npy under {args.data_dir}")
    print(f"discovered {len(files)} trajectories under {args.data_dir} "
          f"(ep{episode_idx(files[0])} ... ep{episode_idx(files[-1])})")

    # Decide which subsets to run.
    subsets: list[tuple[str, list[str]]] = []
    if args.first or args.last:
        if args.first:
            subsets.append((f"first_{args.first}", files[: args.first]))
        if args.last:
            subsets.append((f"last_{args.last}", files[-args.last :]))
    elif args.limit:
        subsets.append((f"first_{args.limit}", files[: args.limit]))
    else:
        subsets.append(("all", files))

    results: dict = {}
    for label, file_subset in subsets:
        print()
        print(f"=== subset: {label} ({len(file_subset)} traj) ===")
        rows, c_arr, m_arr = run_subset(model, file_subset, args, text, text_neg,
                                        max_length, device, isaaclab_aligned)
        results[label] = (rows, c_arr, m_arr)

    # Per-subset summary
    print()
    print("=" * 64)
    for label, (rows, c_arr, m_arr) in results.items():
        c_mean_overall = float(c_arr.mean()) if c_arr.size else float("nan")
        m_mean_overall = float(m_arr.mean()) if (m_arr is not None and m_arr.size) else None
        print(f"[{label}]  trajs={len(rows)}  mean_progress={c_mean_overall:.4f}", end="")
        if m_mean_overall is not None:
            print(f"  mismatched={m_mean_overall:.4f}  gap={c_mean_overall - m_mean_overall:+.4f}")
        else:
            print()
    print("=" * 64)

    # Per-trajectory CSV (one CSV per subset to keep things tidy)
    if args.output_csv:
        base, ext = os.path.splitext(args.output_csv)
        ext = ext or ".csv"
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        for label, (rows, _, _) in results.items():
            csv_path = f"{base}__{label}{ext}" if len(results) > 1 else args.output_csv
            cols = sorted({k for r in rows for k in r.keys()})
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=cols)
                writer.writeheader()
                writer.writerows(rows)
            print(f"wrote per-trajectory CSV to {csv_path}")

        plot_path = f"{base}__compare.png"
        plot_subset_curves(results, plot_path)

    # Optional: dump raw per-frame curves (one JSON per subset)
    if args.save_curves and args.output_csv:
        import json
        base, _ = os.path.splitext(args.output_csv)
        for label, (rows, c_arr, m_arr) in results.items():
            curves = {}
            for i, r in enumerate(rows):
                curves[r["file"]] = {
                    "correct": c_arr[i].tolist(),
                    "mismatched": m_arr[i].tolist() if m_arr is not None else None,
                }
            cur_path = f"{base}__{label}_curves.json"
            with open(cur_path, "w") as fh:
                json.dump(curves, fh)
            print(f"wrote per-frame curves to {cur_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Trained tactile reward checkpoint (.pth).")
    ap.add_argument("--data_dir",
                    default="/mnt/home/uber/tactile_isaaclab/tactile_dataset",
                    help="Directory with ep*.npy trajectory files.")
    ap.add_argument("--instruction", required=True,
                    help='Task description, e.g. "grasp peg and insert to another hole".')
    ap.add_argument("--mismatched_instruction", default=None,
                    help="Optional: a deliberately wrong instruction to compare against.")
    ap.add_argument("--shear_channels", type=int, nargs=2, default=[0, 1],
                    help="Indices of the 2 shear channels in (Fx, Fy, |F|).")
    ap.add_argument("--single_hand", action="store_true", default=True,
                    help="Duplicate the single-hand input along width to match the "
                         "bimanual layout (default: on).")
    ap.add_argument("--bimanual", dest="single_hand", action="store_false",
                    help="Override: data is already two-hand-concatenated.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap the number of trajectories processed (for fast smoke tests).")
    ap.add_argument("--first", type=int, default=None,
                    help="Run on the first N trajectories sorted by episode index.")
    ap.add_argument("--last", type=int, default=None,
                    help="Run on the last N trajectories sorted by episode index. "
                         "Combined with --first, a comparison plot is produced.")
    ap.add_argument("--output_csv",
                    default="/mnt/tank/uber/Tactile-Reward/cross_dataset_progress.csv")
    ap.add_argument("--save_curves", action="store_true",
                    help="Also dump per-frame progress curves as JSON.")
    args = ap.parse_args()
    main(args)
