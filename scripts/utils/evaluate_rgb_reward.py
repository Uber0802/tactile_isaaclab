"""Run an RGB trajectory `.npy` through the peg visual ReWiND reward model
and dump per-frame predicted progress as a CSV + PNG plot.

The pipeline matches `_compute_visual_reward()` in forge_env.py:
  RGB → ImageNet normalize → DINOv2 ViT-B/14 → ReWiNDTransformer + text emb
  → progress[:, -1] per frame (sliding 16-frame window).

Usage:
    python evaluate_rgb_reward.py \\
        --traj /mnt/tank/.../pegpickplace_paired/multipos/ep_2700/ep5_env005_camera.npy \\
        --ckpt /mnt/tank/uber/Tactile-Reward/ckpt_visual/peg_rgb_multipos.pth \\
        --text "grasp peg and insert to another hole" \\
        --out  /tmp/peg_ep5_env005_reward
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


def load_traj(path):
    raw = np.load(path, allow_pickle=True)
    if raw.dtype == object or raw.ndim == 0:
        d = raw.item()
        return d["Camera"], int(d.get("Success", -1))
    return raw, -1


def load_rewind_model(ckpt_path, rewind_root, max_length, device):
    """Import ReWiNDTransformer from `rewind_root/model.py` via importlib so
    we don't collide with Tactile-ReWiND's same-named class."""
    model_path = os.path.join(rewind_root, "model.py")
    spec = importlib.util.spec_from_file_location("rewind_visual_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    if rewind_root not in sys.path:
        sys.path.insert(0, rewind_root)
    spec.loader.exec_module(mod)
    ReWiNDTransformer = mod.ReWiNDTransformer

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("args", None)
    cfg_max = getattr(cfg, "max_length", max_length) if cfg is not None else max_length
    if cfg_max != max_length:
        print(f"[warn] ckpt max_length={cfg_max}, overriding to {max_length}")
    model_args = cfg if cfg is not None else argparse.Namespace()
    model_args.max_length = max_length

    model = ReWiNDTransformer(
        args=model_args, video_dim=768, text_dim=384, hidden_dim=512,
    ).to(device).eval()
    model.load_state_dict(state["model_state_dict"])
    for p in model.parameters():
        p.requires_grad = False
    return model


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
        text_emb = F.normalize(text_emb, p=2, dim=1)
    return text_emb.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="path to *_camera.npy")
    ap.add_argument("--ckpt", default="/mnt/tank/uber/Tactile-Reward/ckpt_visual/peg_rgb_multipos.pth")
    ap.add_argument("--text", default="grasp peg and insert to another hole")
    ap.add_argument("--rewind-root",
                    default="/mnt/home/tactile/tactile_isaaclab/external/third-party/ReWiND")
    ap.add_argument("--max-length", type=int, default=16)
    ap.add_argument("--out", default=None,
                    help="prefix for output files (.csv + .png). default = traj path stem")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ema", type=float, default=1.0,
                    help="EMA alpha for smoothing (1.0 = no smoothing). "
                         "smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]")
    args = ap.parse_args()

    device = torch.device(args.device)
    rgb, success = load_traj(args.traj)
    print(f"loaded {args.traj}")
    print(f"  shape={rgb.shape} dtype={rgb.dtype} success={success}")
    T = rgb.shape[0]

    print(f"loading reward model: {args.ckpt}")
    model = load_rewind_model(args.ckpt, args.rewind_root, args.max_length, device)
    print(f"loading DINOv2 backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print(f"encoding text instruction: {args.text!r}")
    text_emb = encode_text(args.text, device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Encode ALL frames once through DINOv2 (B=T images, cheap on RTX 4090).
    rgb_t = torch.from_numpy(rgb).to(device).float() / 255.0      # (T, H, W, 3)
    rgb_t = rgb_t.permute(0, 3, 1, 2)                             # (T, 3, H, W)
    if rgb_t.shape[-1] != 224:
        rgb_t = F.interpolate(rgb_t, size=(224, 224), mode="bilinear", align_corners=False)
    rgb_t = (rgb_t - mean) / std

    with torch.no_grad():
        # batch large frame count to avoid OOM
        batch = 64
        feats = []
        for i in range(0, T, batch):
            feats.append(backbone(rgb_t[i:i + batch]))
        feats = torch.cat(feats, dim=0)                           # (T, 768)

    # Sliding 16-frame window — match what forge_env / RewindRewardShaper do:
    # at step t, take frames [t-15 ... t]; pad with frame[0] if t < 16.
    L = args.max_length
    progress = np.zeros(T, dtype=np.float32)
    text_b = text_emb.expand(1, -1)
    with torch.no_grad():
        for t in range(T):
            if t < L - 1:
                # Pad head with feat[0] to fill window.
                pad = feats[0:1].expand(L - 1 - t, -1)            # (L-1-t, 768)
                tail = feats[:t + 1]                              # (t+1, 768)
                window = torch.cat([pad, tail], dim=0)            # (L, 768)
            else:
                window = feats[t - L + 1:t + 1]                   # (L, 768)
            inp = window.unsqueeze(0)                             # (1, L, 768)
            pred = model(inp, text_b)
            if pred.ndim == 3:
                pred = pred.squeeze(-1)
            progress[t] = float(pred[0, -1].item())

    # EMA smoothing.
    alpha = float(args.ema)
    if alpha < 1.0:
        smoothed = np.zeros_like(progress)
        smoothed[0] = progress[0]
        for t in range(1, T):
            smoothed[t] = alpha * progress[t] + (1.0 - alpha) * smoothed[t - 1]
    else:
        smoothed = progress.copy()

    # Write CSV + PNG.
    if args.out is None:
        args.out = os.path.splitext(args.traj)[0] + "_reward"
    csv_path = args.out + ".csv"
    png_path = args.out + ".png"
    with open(csv_path, "w") as f:
        if alpha < 1.0:
            f.write("frame,predicted_reward,ema_smoothed\n")
            for i, (p, s) in enumerate(zip(progress, smoothed)):
                f.write(f"{i},{p:.6f},{s:.6f}\n")
        else:
            f.write("frame,predicted_reward\n")
            for i, p in enumerate(progress):
                f.write(f"{i},{p:.6f}\n")
    print(f"wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(smoothed, color="C0", linewidth=2.5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axhline(1, color="gray", linewidth=0.5, linestyle=":")
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("frame")
        ax.set_ylabel("predicted reward")
        ax.set_title(f"{os.path.basename(args.traj)} | T={T} | success={success}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"wrote {png_path}")
    except Exception as e:
        print(f"[warn] matplotlib failed: {e}")

    series = smoothed if alpha < 1.0 else progress
    print(f"\nfirst 5 ({'smoothed' if alpha < 1.0 else 'raw'}): {series[:5]}")
    print(f"last 5  : {series[-5:]}")
    print(f"min={series.min():.4f}  max={series.max():.4f}  mean={series.mean():.4f}")


if __name__ == "__main__":
    main()
