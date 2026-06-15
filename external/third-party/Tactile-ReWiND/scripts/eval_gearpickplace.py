"""Evaluate a gearpickplace tactile reward ckpt on the holdout set written by
`train_gearpickplace.py` to `<ckpt_dir>/eval_holdout.json`.

Produces three predicted-progress curves on one PNG:
  * success forward — should climb 0 -> 1
  * failure forward — should stay near 0
  * success rewind  — should climb then fall

Usage:
    python scripts/eval_gearpickplace.py \\
        --ckpt /mnt/tank/tactile/Tactile-Reward/checkpoints_gearpickplace_balanced/gearpickplace_epoch99.pth \\
        --instruction "pick up the gear and mesh it onto the shaft"
"""
from __future__ import annotations

import os
import sys
import json
import random
import argparse
from typing import Dict, List

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


def scan_train_entries(data_dirs: List[str], holdout_keys: set) -> List[Dict]:
    """Walk data_dirs, return ep dicts EXCLUDING anything in holdout_keys."""
    entries: List[Dict] = []
    for d in data_dirs:
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".npy"):
                continue
            if (d, fn) in holdout_keys:
                continue
            try:
                arr = np.load(os.path.join(d, fn), allow_pickle=True)
            except Exception:
                continue
            if arr.dtype != object:
                continue
            dct = arr.item() if arr.ndim == 0 else None
            if not (isinstance(dct, dict) and "Success" in dct and "Tactile" in dct):
                continue
            entries.append({"dir": d, "file": fn, "success": int(dct["Success"])})
    return entries


def load_entry(entry: dict, shear_channels):
    path = os.path.join(entry["dir"], entry["file"])
    arr = np.load(path, allow_pickle=True)
    if arr.dtype != object:
        raise RuntimeError(f"unexpected ndarray format (not dict): {path}")
    d = arr.item()
    if not (isinstance(d, dict) and "Tactile" in d and "Success" in d):
        raise RuntimeError(f"missing Tactile/Success keys: {path}")
    traj = np.asarray(d["Tactile"])[..., list(shear_channels)]
    return traj, int(d["Success"])


def main(args):
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-resolve holdout JSON next to the ckpt unless overridden.
    holdout_path = args.holdout or os.path.join(
        os.path.dirname(args.ckpt), "eval_holdout.json")
    if not os.path.isfile(holdout_path):
        raise SystemExit(f"holdout JSON not found: {holdout_path}\n"
                         f"  pass --holdout explicitly")
    with open(holdout_path) as fh:
        holdout = json.load(fh)

    if args.mode == "holdout":
        entries = holdout["eval_entries"]
        print(f"loaded holdout from {holdout_path}: {len(entries)} entries")
    else:  # train
        holdout_keys = {(e["dir"], e["file"]) for e in holdout["eval_entries"]}
        all_train = scan_train_entries(holdout["data_dirs"], holdout_keys)
        train_succ = [e for e in all_train if e["success"]]
        train_fail = [e for e in all_train if not e["success"]]
        rng_pick = random.Random(args.seed)
        rng_pick.shuffle(train_succ)
        rng_pick.shuffle(train_fail)
        if len(train_succ) < args.n_train_succ:
            raise SystemExit(f"only {len(train_succ)} train success episodes, "
                             f"need {args.n_train_succ}")
        if len(train_fail) < args.n_train_fail:
            raise SystemExit(f"only {len(train_fail)} train fail episodes, "
                             f"need {args.n_train_fail}")
        entries = train_succ[:args.n_train_succ] + train_fail[:args.n_train_fail]
        print(f"sampled {len(entries)} train entries from "
              f"{len(all_train)} (succ pool={len(train_succ)}, "
              f"fail pool={len(train_fail)}, seed={args.seed})")

    n_succ_eval = sum(1 for e in entries if e["success"])
    n_fail_eval = len(entries) - n_succ_eval
    print(f"  → {n_succ_eval} success + {n_fail_eval} fail")

    model, cfg = load_model(args.ckpt, device)
    max_length = cfg.get("max_length", 16)
    print(f"loaded {args.ckpt} (max_length={max_length})")

    text = encode_minilm_one(args.instruction, device).unsqueeze(0)
    print(f"instruction: {args.instruction!r}")

    succ_fwd: List[np.ndarray] = []
    succ_rew: List[np.ndarray] = []
    succ_rew_peaks: List[float] = []
    fail_fwd: List[np.ndarray] = []

    for entry in tqdm(entries, desc="eval"):
        traj, success = load_entry(entry, tuple(args.shear_channels))
        T = len(traj)
        if T < 4:
            print(f"  skipping short traj: {entry['file']} (T={T})")
            continue

        fwd_idx = forward_indices(T, max_length)
        x_fwd = to_chw_tensor(np.ascontiguousarray(traj[fwd_idx]), device)
        with torch.no_grad():
            pred_fwd = model(x_fwd, text).squeeze(-1).squeeze(0).cpu().numpy()
        (succ_fwd if success else fail_fwd).append(pred_fwd)

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

    def _mean_last(arr):
        return float(arr[:, -1].mean()) if len(arr) else float("nan")

    def _mean_all(arr):
        return float(arr[:, 1:].mean()) if len(arr) else float("nan")

    print()
    print("=" * 72)
    print(f"success (fwd): n={len(succ_fwd)}  mean={_mean_all(succ_fwd):.4f}  "
          f"final-frame={_mean_last(succ_fwd):.4f}")
    print(f"failure (fwd): n={len(fail_fwd)}  mean={_mean_all(fail_fwd):.4f}  "
          f"final-frame={_mean_last(fail_fwd):.4f}")
    print(f"success (rew): n={len(succ_rew)}  mean={_mean_all(succ_rew):.4f}")
    if len(succ_fwd) and len(fail_fwd):
        sep = _mean_last(succ_fwd) - _mean_last(fail_fwd)
        print(f"success/fail separation at final frame: {sep:+.4f}")
    print("=" * 72)

    # Optional JSON dump of per-entry preds.
    if args.dump_json:
        rows = []
        for arr, label, kind in [(succ_fwd, 1, "forward"),
                                 (fail_fwd, 0, "forward"),
                                 (succ_rew, 1, "rewind")]:
            for i, p in enumerate(arr):
                rows.append({"success": label, "kind": kind, "pred": p.tolist()})
        out_json = args.dump_json
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as fh:
            json.dump({"ckpt": args.ckpt, "instruction": args.instruction,
                       "max_length": max_length, "rows": rows}, fh, indent=2)
        print(f"wrote {out_json}")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    fig.patch.set_facecolor("white")
    x_axis = np.arange(max_length)

    ax.plot(x_axis, gt_forward(max_length), "--", color="0.55", lw=1.0, alpha=0.6,
            label="GT forward (success)")
    ax.axhline(0, color="0.55", lw=1.0, alpha=0.4, linestyle="--")
    if succ_rew_peaks:
        mean_peak = float(np.mean(succ_rew_peaks))
        ax.plot(x_axis, gt_rewind(max_length, mean_peak), ":", color="0.55",
                lw=1.0, alpha=0.6, label=f"GT rewind (peak={mean_peak:.2f})")

    def plot_band(arr, color, label):
        if arr.size == 0:
            return
        m, s = arr.mean(0), arr.std(0)
        ax.plot(x_axis, m, "-", color=color, lw=2.6, label=label)
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
    ckpt_label = os.path.basename(args.ckpt).replace(".pth", "")
    mode_label = "holdout" if args.mode == "holdout" else "TRAIN-set"
    ax.set_title(f"GearPickPlace {mode_label} eval — {ckpt_label}\n"
                 f"{len(entries)} eps ({n_succ_eval} succ + {n_fail_eval} fail)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Path to a gearpickplace_epoch*.pth ckpt.")
    ap.add_argument("--holdout", default=None,
                    help="Path to eval_holdout.json. Default: alongside the ckpt.")
    ap.add_argument("--instruction",
                    default="pick up the gear and mesh it onto the shaft")
    ap.add_argument("--shear_channels", type=int, nargs=2, default=[1, 2])
    ap.add_argument("--peak_min", type=float, default=0.5)
    ap.add_argument("--peak_max", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None,
                    help="PNG path. Default: <ckpt_dir>/gearpickplace_eval[_train].png")
    ap.add_argument("--dump_json", default=None,
                    help="Optional path to dump per-entry predictions.")
    ap.add_argument("--mode", choices=["holdout", "train"], default="holdout",
                    help="holdout = the 20 ep eval set from training; "
                         "train = N random samples FROM the train set "
                         "(sanity check for overfit vs. underfit).")
    ap.add_argument("--n_train_succ", type=int, default=10,
                    help="(--mode train only) number of train success eps to sample.")
    ap.add_argument("--n_train_fail", type=int, default=10,
                    help="(--mode train only) number of train fail eps to sample.")
    args = ap.parse_args()
    if args.output is None:
        suffix = "_train" if args.mode == "train" else ""
        args.output = os.path.join(os.path.dirname(args.ckpt),
                                   f"gearpickplace_eval{suffix}.png")
    main(args)
