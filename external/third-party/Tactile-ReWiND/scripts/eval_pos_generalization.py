"""Paper-style position-generalization eval for a single tactile reward ckpt.

For every .npy episode in `--data_dirs`, runs a 16-frame rolling causal window
inference and aggregates per-frame mean / std of the predicted reward over
Success=1 and Success=0 trajectories. The output PNG matches the
TacReward_pos_generalization plotting style used in the paper:

  * figsize=(10, 5), no title
  * cyan #02bfbf success / purple #c109c1 fail, lw=3, ±1σ band
  * EMA-smoothed mean (alpha=0.15 by default)
  * xlabel="Frame Index", ylabel="predicted reward", bold size 18
  * legend "success trajectory" / "failure trajectory", upper left, bold

Episode .npy format:
    {"Task": str (ignored — use --task_text),
     "Tactile": (T, 40, 25, 3) or (T, 20, 25, 6),
     "Success": 0|1}

Example:
    python scripts/eval_pos_generalization.py \\
      --data_dirs /mnt/home/kimnai/research/tactile_isaaclab/tactile_dataset/stack_box/multipos \\
      --ckpt /mnt/lab-tank/uber/Tactile-Reward/box_kimnai_curriculum/box_kimnai_curr_epoch25.pth \\
      --task_text "grasp the blue box and stack it on the red box" \\
      --out_png  /tmp/box_kimnai_multipos.png \\
      --out_json /tmp/box_kimnai_multipos.json \\
      --limit 1000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve repo root (this script lives in <repo>/scripts/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.tactile_model import TactileReWiNDTransformer


# ---------- model + text ----------

def load_model(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("args", {}) or {}
    if hasattr(cfg, "__dict__"):  # argparse.Namespace -> dict-like
        cfg = vars(cfg)
    m = TactileReWiNDTransformer(
        max_length=cfg.get("max_length", 16), text_dim=384,
        hidden_dim=cfg.get("hidden_dim", 512),
        num_heads=cfg.get("num_heads", 8),
        num_layers=cfg.get("num_layers", 4),
        per_hand_dim=cfg.get("per_hand_dim", 384),
        num_strided_layers=cfg.get("num_strided_layers", 0) or 3,
        bimanual_axis="height" if cfg.get("isaaclab_aligned", True) else "width",
        in_channels=cfg.get("in_channels", 3),
    ).to(device).eval()
    sd = state["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m, cfg.get("normalize_mode", "global")


def embed_text(text: str, device: torch.device) -> torch.Tensor:
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    mdl = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(device).eval()
    enc = tok([text], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl(**enc); t_out = out[0]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        e = (t_out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    del mdl, tok
    return e.float().squeeze(0)


# ---------- data ----------

def reshape_if_needed(tac: np.ndarray) -> np.ndarray:
    """(T, 20, 25, 6) -> (T, 40, 25, 3) by concat L|R on H axis."""
    if tac.ndim == 4 and tac.shape[1:] == (20, 25, 6):
        return np.concatenate([tac[..., :3], tac[..., 3:]], axis=1)
    return tac


def load_episodes(data_dirs: list[str], limit: int | None, seed: int = 42):
    files: list[str] = []
    for d in data_dirs:
        f = sorted(glob.glob(os.path.join(d, "**/*.npy"), recursive=True))
        if not f:
            f = sorted(glob.glob(os.path.join(d, "*.npy")))
        files.extend(f)
    print(f"  found {len(files)} npy across {len(data_dirs)} dir(s)", flush=True)
    if limit and len(files) > limit:
        rng = random.Random(seed)
        files = rng.sample(files, limit); files.sort()
        print(f"  sampled to {limit}", flush=True)

    eps: list[dict] = []
    skipped = 0
    for p in files:
        try:
            raw = np.load(p, allow_pickle=True)
            d = raw.item() if (raw.dtype == object and raw.ndim == 0) else raw
            d["Task"] = "grasp the blue box and stack it on the red box"
        except Exception:
            skipped += 1; continue
        if not isinstance(d, dict):
            skipped += 1; continue
        tac = reshape_if_needed(np.asarray(d["Tactile"], dtype=np.float32))
        if tac.ndim != 4 or tac.shape[1:] != (40, 25, 3):
            skipped += 1; continue
        eps.append({"tactile": tac, "success": int(d["Success"])})
    if skipped:
        print(f"  skipped {skipped} malformed files", flush=True)
    return eps


# ---------- rolling inference ----------

def build_window(traj_len: int, t: int, max_length: int) -> np.ndarray:
    if t >= max_length:
        return np.round(np.linspace(0, t - 1, max_length)).astype(np.int64)
    return np.concatenate([
        np.arange(t, dtype=np.int64),
        np.full(max_length - t, t - 1, dtype=np.int64),
    ])


@torch.no_grad()
def rolling(traj: np.ndarray, model, text_emb: torch.Tensor, norm_mode: str,
            device: torch.device, max_length: int, batch: int) -> np.ndarray:
    N = traj.shape[0]
    windows = np.stack(
        [traj[build_window(N, t, max_length)] for t in range(1, N + 1)], axis=0
    )
    if norm_mode == "global":
        d = np.abs(windows).max(axis=(1, 2, 3, 4), keepdims=True)
        np.maximum(d, 1e-6, out=d)
        windows = windows / d
    x = torch.from_numpy(windows.astype(np.float32)).permute(0, 1, 4, 2, 3).contiguous().to(device)
    text = text_emb.unsqueeze(0).expand(x.size(0), -1)
    rs = []
    for i in range(0, N, batch):
        bx = x[i:i + batch]
        bt = text[:bx.size(0)]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
            out = model(bx, bt).squeeze(-1).float()
        rs.append(out[:, -1].cpu().numpy())
    return np.concatenate(rs)


# ---------- aggregation + plot ----------

def ema(y: np.ndarray, alpha: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y); out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def aggregate(pool: dict[int, list[float]]):
    if not pool:
        return None
    ts = sorted(pool)
    return {
        "frames": [int(t) for t in ts],
        "mean":   [float(np.mean(pool[t])) for t in ts],
        "std":    [float(np.std(pool[t]))  for t in ts],
        "n":      [len(pool[t])            for t in ts],
    }


def paper_style_plot(succ_agg, fail_agg, out_png: str, *,
                     ema_alpha: float, min_n: int,
                     ylim: tuple[float, float], xlim: tuple[float, float] | None,
                     fail_color: str = "#c109c1", succ_color: str = "#02bfbf"):
    fig, ax = plt.subplots(figsize=(10, 5))
    xmax = 0
    for agg, color, alpha_fill, label, zorder in [
        (succ_agg, succ_color, 0.20, "success trajectory", 4),
        (fail_agg, fail_color, 0.15, "failure trajectory", 3),
    ]:
        if agg is None:
            continue
        ts = np.array(agg["frames"]); m = np.array(agg["mean"])
        s  = np.array(agg["std"]);    n = np.array(agg["n"])
        mask = n >= min_n
        if not mask.any():
            continue
        xs = ts[mask]; ms = ema(m[mask], ema_alpha); ss = ema(s[mask], ema_alpha)
        ax.plot(xs, ms, "-", color=color, lw=3.0, zorder=zorder, label=label)
        ax.fill_between(xs, ms - ss, ms + ss, alpha=alpha_fill, color=color, zorder=1)
        xmax = max(xmax, int(xs[-1]))
    if xlim is None:
        xlim = (0, xmax if xmax > 0 else 1)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("Frame Index",      fontsize=18, fontweight="bold")
    ax.set_ylabel("predicted reward", fontsize=18, fontweight="bold")
    ax.minorticks_on()
    ax.grid(which="major", alpha=0.4); ax.grid(which="minor", alpha=0.2)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(prop={"weight": "bold", "size": 16}, loc="upper left",
              handleheight=1.2, handlelength=2.5, handletextpad=0.5)
    plt.tight_layout()
    tmp = f"/tmp/__pos_gen_{os.getpid()}.png"
    plt.savefig(tmp, dpi=300); plt.close()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    shutil.move(tmp, out_png)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True,
                    help="One or more directories of .npy episodes")
    ap.add_argument("--ckpt", required=True, help="Tactile reward ckpt (.pth)")
    ap.add_argument("--task_text", required=True,
                    help="Instruction text used at inference")
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_json", default=None,
                    help="Optional raw curve JSON output path")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap episode count (random sample, seed=42). 0=no cap")
    ap.add_argument("--max_length", type=int, default=16)
    ap.add_argument("--infer_batch", type=int, default=128)
    ap.add_argument("--ema_alpha", type=float, default=0.15)
    ap.add_argument("--min_n", type=int, default=5,
                    help="Per-frame minimum sample count to draw")
    ap.add_argument("--ymax", type=float, default=1.0,
                    help="Y-axis upper limit (e.g. 0.4 if rewards are small)")
    ap.add_argument("--xmax", type=int, default=0,
                    help="X-axis upper limit; 0 = auto from data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    print(f"\nloading model: {args.ckpt}", flush=True)
    model, norm = load_model(args.ckpt, device)
    text_emb = embed_text(args.task_text, device)
    print(f"  text: {args.task_text!r}  normalize_mode={norm}", flush=True)

    print(f"\nloading data from {len(args.data_dirs)} dir(s)", flush=True)
    eps = load_episodes(args.data_dirs, args.limit or None, args.seed)
    n_s = sum(e["success"] for e in eps)
    n_f = len(eps) - n_s
    print(f"  loaded {len(eps)} eps ({n_s} succ / {n_f} fail)", flush=True)
    if not eps:
        print("no episodes loaded; abort", flush=True); sys.exit(1)

    print(f"\nrunning rolling inference", flush=True)
    succ_pool: dict[int, list[float]] = defaultdict(list)
    fail_pool: dict[int, list[float]] = defaultdict(list)
    t0 = time.time()
    for i, ep in enumerate(eps):
        r = rolling(ep["tactile"], model, text_emb, norm, device,
                    args.max_length, args.infer_batch)
        pool = succ_pool if ep["success"] else fail_pool
        for ti, v in enumerate(r, start=1):
            pool[ti].append(float(v))
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(eps) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(eps)}] {rate:.1f} ep/s  ETA {eta:.0f}s", flush=True)

    succ_agg = aggregate(succ_pool)
    fail_agg = aggregate(fail_pool)
    res = {
        "ckpt": args.ckpt,
        "task_text": args.task_text,
        "data_dirs": args.data_dirs,
        "n_success": n_s, "n_fail": n_f,
        "ema_alpha": args.ema_alpha, "min_n": args.min_n,
        "success": succ_agg, "fail": fail_agg,
    }
    if args.out_json:
        tmp_json = f"/tmp/__pos_gen_{os.getpid()}.json"
        with open(tmp_json, "w") as f:
            json.dump(res, f, indent=2)
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        shutil.move(tmp_json, args.out_json)
        print(f"raw json: {args.out_json}", flush=True)

    xlim = (0, args.xmax) if args.xmax > 0 else None
    paper_style_plot(succ_agg, fail_agg, args.out_png,
                     ema_alpha=args.ema_alpha, min_n=args.min_n,
                     ylim=(0.0, args.ymax), xlim=xlim)
    print(f"saved: {args.out_png}", flush=True)


if __name__ == "__main__":
    main()
