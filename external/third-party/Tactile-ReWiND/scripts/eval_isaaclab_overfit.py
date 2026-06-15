"""IsaacLab overfit-eval: inference on every ep*.npy and plot average progress
curves grouped by (success / failure) × (forward / rewind).

Three curves overlaid on one PNG:
  1. Successful trajectories — forward sample        (should climb 0 -> 1)
  2. Failed trajectories — forward sample            (should stay near 0)
  3. Successful trajectories — rewind sample         (should climb then fall)

If you trained with `--synthetic_success_threshold N`, pass the same threshold
here so labels line up with the training run. If the npy files are dicts with
`Success` keys, the threshold is ignored.

Usage:
    python scripts/eval_isaaclab_overfit.py \\
        --ckpt /mnt/tank/uber/Tactile-Reward/checkpoints_isaaclab_overfit/isaaclab_overfit_epoch29.pth \\
        --data_dir /mnt/home/uber/tactile_isaaclab/tactile_dataset/data_2 \\
        --synthetic_success_threshold 970 \\
        --instruction "grasp peg and insert to another hole" \\
        --output /mnt/tank/uber/Tactile-Reward/isaaclab_overfit_avg.png
"""
from __future__ import annotations

import os
import re
import sys
import argparse
from typing import Optional, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def encode_minilm_one(text: str, device) -> torch.Tensor:
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    m = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(device)
    m.eval()
    with torch.no_grad():
        enc = tok([text], padding=True, truncation=True, return_tensors="pt").to(device)
        out = m(**enc)
        return mean_pooling(out, enc["attention_mask"]).float().squeeze(0)


def load_model(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("args", {})
    # IsaacLab overfit ckpts default to 3-layer height-split encoder. Older
    # ckpts that don't carry these fields explicitly fall back to that.
    num_strided = cfg.get("num_strided_layers", None) or 3
    bimanual_axis = cfg.get("bimanual_axis", None) or "height"
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


def forward_indices(T: int, max_length: int) -> np.ndarray:
    return np.round(np.linspace(0, T - 1, max_length)).astype(int)


def rewind_indices(T: int, max_length: int, peak_ratio: float) -> np.ndarray:
    peak = int(round(peak_ratio * (T - 1)))
    peak = max(2, min(T - 1, peak))
    n_fwd = (max_length + 1) // 2
    n_rev = max_length - n_fwd
    fwd = np.round(np.linspace(0, peak, n_fwd)).astype(int)
    rev = (np.round(np.linspace(peak - 1, 0, n_rev)).astype(int)
           if n_rev > 0 else np.empty(0, dtype=int))
    return np.concatenate([fwd, rev])


def gt_forward(max_length: int) -> np.ndarray:
    return np.linspace(0, 1, max_length, dtype=np.float32)


def gt_rewind(max_length: int, peak_ratio: float) -> np.ndarray:
    n_fwd = (max_length + 1) // 2
    n_rev = max_length - n_fwd
    fwd = np.linspace(0, peak_ratio, n_fwd)
    rev = (np.linspace(peak_ratio * (n_fwd - 1) / n_fwd, 0, n_rev)
           if n_rev > 0 else np.empty(0))
    return np.concatenate([fwd, rev]).astype(np.float32)


def to_chw_tensor(frames_thwc: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(frames_thwc, dtype=np.float32))
    x = x.permute(0, 3, 1, 2).contiguous()
    return x.unsqueeze(0).to(device)


def load_traj_and_label(path: str, shear_channels, threshold: Optional[int]):
    """Returns (traj_2ch, success_int) or None if unusable."""
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object:
        d = arr.item()
        if not (isinstance(d, dict) and "Tactile" in d and "Success" in d):
            return None
        traj = np.asarray(d["Tactile"])
        success = int(d["Success"])
    elif arr.ndim == 4:
        traj = np.asarray(arr)
        m = re.match(r"^ep(\d+)\.npy$", os.path.basename(path))
        if not m or threshold is None:
            return None
        success = 1 if int(m.group(1)) >= threshold else 0
    else:
        return None
    traj = traj[..., list(shear_channels)]
    return traj, success


def main(args):
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_model(args.ckpt, device)
    max_length = cfg.get("max_length", 16)
    print(f"loaded {args.ckpt} (max_length={max_length})")

    text = encode_minilm_one(args.instruction, device).unsqueeze(0)
    print(f"instruction: {args.instruction!r}")

    files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".npy"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no *.npy under {args.data_dir}")

    succ_fwd: List[np.ndarray] = []
    succ_rew: List[np.ndarray] = []
    succ_rew_peaks: List[float] = []
    fail_fwd: List[np.ndarray] = []

    for fn in tqdm(files):
        path = os.path.join(args.data_dir, fn)
        loaded = load_traj_and_label(path, tuple(args.shear_channels),
                                     args.synthetic_success_threshold)
        if loaded is None:
            continue
        traj, success = loaded
        T = len(traj)
        if T < 4:
            continue

        fwd_idx = forward_indices(T, max_length)
        x_fwd = to_chw_tensor(np.ascontiguousarray(traj[fwd_idx]), device)
        with torch.no_grad():
            pred_fwd = model(x_fwd, text).squeeze(-1).squeeze(0).cpu().numpy()
        if success:
            succ_fwd.append(pred_fwd)
        else:
            fail_fwd.append(pred_fwd)

        if success:
            peak = float(rng.uniform(args.peak_min, args.peak_max))
            rew_idx = rewind_indices(T, max_length, peak)
            x_rew = to_chw_tensor(np.ascontiguousarray(traj[rew_idx]), device)
            with torch.no_grad():
                pred_rew = model(x_rew, text).squeeze(-1).squeeze(0).cpu().numpy()
            succ_rew.append(pred_rew)
            succ_rew_peaks.append(peak)

    succ_fwd = np.stack(succ_fwd) if succ_fwd else np.zeros((0, max_length))
    fail_fwd = np.stack(fail_fwd) if fail_fwd else np.zeros((0, max_length))
    succ_rew = np.stack(succ_rew) if succ_rew else np.zeros((0, max_length))

    print()
    print("=" * 64)
    print(f"successful (forward): n = {len(succ_fwd)}, "
          f"mean progress = {float(succ_fwd[:, 1:].mean()) if len(succ_fwd) else float('nan'):.4f}")
    print(f"failed     (forward): n = {len(fail_fwd)}, "
          f"mean progress = {float(fail_fwd[:, 1:].mean()) if len(fail_fwd) else float('nan'):.4f}")
    print(f"successful (rewind):  n = {len(succ_rew)}, "
          f"mean progress = {float(succ_rew[:, 1:].mean()) if len(succ_rew) else float('nan'):.4f}")
    print("=" * 64)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    fig.patch.set_facecolor("white")
    x_axis = np.arange(max_length)

    ax.plot(x_axis, gt_forward(max_length), "--", color="0.55", lw=1.0, alpha=0.6,
            label="GT forward (success)")
    ax.axhline(0, color="0.55", lw=1.0, alpha=0.4, linestyle="--")
    if len(succ_rew_peaks) > 0:
        mean_peak = float(np.mean(succ_rew_peaks))
        ax.plot(x_axis, gt_rewind(max_length, mean_peak), ":", color="0.55",
                lw=1.0, alpha=0.6, label=f"GT rewind (peak={mean_peak:.2f})")

    def plot_band(arr, color, label):
        if arr.size == 0:
            return
        m, s = arr.mean(0), arr.std(0)
        ax.plot(x_axis, m, "-", color=color, lw=2.4, label=label)
        ax.fill_between(x_axis, m - s, m + s, color=color, alpha=0.15)

    plot_band(succ_fwd, "tab:green",  f"success forward (n={len(succ_fwd)})")
    plot_band(fail_fwd, "tab:red",    f"failure forward (n={len(fail_fwd)})")
    plot_band(succ_rew, "tab:orange", f"success rewind  (n={len(succ_rew)})")

    ax.set_xlabel("frame index")
    ax.set_ylabel("predicted progress")
    ax.set_xlim(-0.5, max_length - 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ckpt_label = os.path.basename(os.path.dirname(args.ckpt))
    ax.set_title(f"IsaacLab overfit eval — {ckpt_label}\n"
                 f"data_dir = {args.data_dir}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--instruction", default="grasp peg and insert to another hole")
    ap.add_argument("--shear_channels", type=int, nargs=2, default=[1, 2])
    ap.add_argument("--synthetic_success_threshold", type=int, default=None,
                    help="Same value used at training time for raw-ndarray files.")
    ap.add_argument("--peak_min", type=float, default=0.5)
    ap.add_argument("--peak_max", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output",
                    default="/mnt/tank/uber/Tactile-Reward/isaaclab_overfit_avg.png")
    args = ap.parse_args()
    main(args)
