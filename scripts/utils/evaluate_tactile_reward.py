"""Run a tactile force-field trajectory `.npy` through a Tactile-ReWiND
reward model and dump per-frame predicted progress as a CSV + PNG.

Mirrors `_compute_tactile_reward()` in forge_env.py:
  shear+normal (T, 40, 25, 3) → select channels → normalize → 16-frame
  sliding window → TactileReWiNDTransformer + text emb → progress.

Usage:
    python evaluate_tactile_reward.py \\
        --traj /mnt/tank/.../ep_2700/ep8_env008.npy \\
        --ckpt /mnt/tank/uber/Tactile-Reward/peg_curriculum_retrain/peg_curr_epoch17.pth \\
        --text "grasp peg and insert to another hole" \\
        --out  /tmp/peg_ep8_env008_tactile_reward
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


def load_traj(path):
    raw = np.load(path, allow_pickle=True)
    if raw.dtype == object or raw.ndim == 0:
        d = raw.item()
        return d["Tactile"], int(d.get("Success", -1))
    return raw, -1


def load_tactile_model(ckpt_path, tactile_root, device):
    if tactile_root not in sys.path:
        sys.path.insert(0, tactile_root)
    from tools.tactile_model import TactileReWiNDTransformer

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("args", {})
    if hasattr(cfg, "__dict__"):
        cfg = vars(cfg)

    num_strided = cfg.get("num_strided_layers", None) or 3
    bimanual_axis = cfg.get("bimanual_axis", None) or "height"

    cfg_shear = cfg.get("shear_channels", None)
    if cfg_shear:
        shear_channels = tuple(cfg_shear)
    else:
        ic = int(cfg.get("in_channels", 2))
        shear_channels = (0, 1, 2) if ic == 3 else (1, 2)
    in_channels = len(shear_channels)

    max_length = cfg.get("max_length", 16)
    norm_mode = cfg.get("normalize_mode", None)
    if norm_mode is None:
        norm_mode = "per_channel" if cfg.get("normalize_per_channel") else "off"

    model = TactileReWiNDTransformer(
        max_length=max_length,
        text_dim=384,
        hidden_dim=cfg.get("hidden_dim", 512),
        num_heads=cfg.get("num_heads", 8),
        num_layers=cfg.get("num_layers", 4),
        per_hand_dim=cfg.get("per_hand_dim", 384),
        num_strided_layers=num_strided,
        bimanual_axis=bimanual_axis,
        in_channels=in_channels,
    ).to(device).eval()
    model.load_state_dict(state["model_state_dict"])
    for p in model.parameters():
        p.requires_grad = False
    return model, max_length, shear_channels, norm_mode, in_channels


def encode_text(text, device):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    minilm = AutoModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L12-v2"
    ).to(device).eval()
    with torch.no_grad():
        enc = tok([text], padding=True, return_tensors="pt").to(device)
        out = minilm(**enc)
        tok_emb = out[0]
        mask = enc["attention_mask"].unsqueeze(-1).expand(tok_emb.size()).float()
        text_emb = (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return text_emb.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="path to tactile .npy (no _camera suffix)")
    ap.add_argument("--ckpt", default="/mnt/tank/uber/Tactile-Reward/peg_curriculum_retrain/peg_curr_epoch17.pth")
    ap.add_argument("--text", default="grasp peg and insert to another hole")
    ap.add_argument("--tactile-root",
                    default="/mnt/home/tactile/tactile_isaaclab/external/third-party/Tactile-ReWiND")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    tactile, success = load_traj(args.traj)
    print(f"loaded {args.traj}")
    print(f"  shape={tactile.shape} dtype={tactile.dtype} success={success}")
    T = tactile.shape[0]

    print(f"loading tactile model: {args.ckpt}")
    model, max_length, shear_channels, norm_mode, in_channels = load_tactile_model(
        args.ckpt, args.tactile_root, device)
    print(f"  max_length={max_length}  shear_channels={shear_channels}  "
          f"normalize={norm_mode}  in_channels={in_channels}")
    print(f"encoding text instruction: {args.text!r}")
    text_emb = encode_text(args.text, device)

    # (T, 40, 25, 3) -> select channels per ckpt config -> (T, 40, 25, C)
    full = torch.from_numpy(tactile).to(device).float()
    if full.shape[-1] != 3:
        raise SystemExit(f"expected last-dim=3 tactile tensor, got shape {tuple(full.shape)}")
    current = full[..., list(shear_channels)]                       # (T, 40, 25, C)

    L = max_length
    progress = np.zeros(T, dtype=np.float32)
    text_b = text_emb.expand(1, -1)
    with torch.no_grad():
        for t in range(T):
            if t < L - 1:
                pad = current[0:1].expand(L - 1 - t, -1, -1, -1)
                tail = current[:t + 1]
                window = torch.cat([pad, tail], dim=0)              # (L, 40, 25, C)
            else:
                window = current[t - L + 1:t + 1]                   # (L, 40, 25, C)
            window = window.unsqueeze(0)                            # (1, L, 40, 25, C)

            if norm_mode == "global":
                denom = window.abs().amax(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
                window = window / denom
            elif norm_mode == "per_channel":
                denom = window.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
                window = window / denom

            x = window.permute(0, 1, 4, 2, 3).contiguous()          # (1, L, C, 40, 25)
            pred = model(x, text_b)
            if pred.ndim == 3:
                pred = pred.squeeze(-1)
            progress[t] = float(pred[0, -1].item())

    if args.out is None:
        args.out = os.path.splitext(args.traj)[0] + "_tactile_reward"
    csv_path = args.out + ".csv"
    png_path = args.out + ".png"
    with open(csv_path, "w") as f:
        f.write("frame,predicted_reward\n")
        for i, p in enumerate(progress):
            f.write(f"{i},{p:.6f}\n")
    print(f"wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(progress, color="C2", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axhline(1, color="gray", linewidth=0.5, linestyle=":")
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("frame")
        ax.set_ylabel("predicted reward")
        ax.set_title(f"{os.path.basename(args.traj)} | T={T} | success={success} | TACTILE")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"wrote {png_path}")
    except Exception as e:
        print(f"[warn] matplotlib failed: {e}")

    print(f"\nfirst 5  : {progress[:5]}")
    print(f"last 5   : {progress[-5:]}")
    print(f"min={progress.min():.4f}  max={progress.max():.4f}  mean={progress.mean():.4f}")


if __name__ == "__main__":
    main()
