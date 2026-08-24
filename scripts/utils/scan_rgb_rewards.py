"""Batch-evaluate predicted reward over many RGB trajectories. Prints a
summary table sorted by mean — useful for finding low-mean / failure
trajectories.
"""
import argparse
import glob
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
    model_path = os.path.join(rewind_root, "model.py")
    spec = importlib.util.spec_from_file_location("rewind_visual_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    if rewind_root not in sys.path:
        sys.path.insert(0, rewind_root)
    spec.loader.exec_module(mod)
    ReWiNDTransformer = mod.ReWiNDTransformer

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("args", None)
    if cfg is None:
        cfg = argparse.Namespace()
    cfg.max_length = max_length

    model = ReWiNDTransformer(
        args=cfg, video_dim=768, text_dim=384, hidden_dim=512,
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


def predict_trajectory(rgb, backbone, model, text_emb, max_length, device):
    """Return per-frame predicted reward (np.ndarray length T)."""
    rgb_t = torch.from_numpy(rgb).to(device).float() / 255.0
    rgb_t = rgb_t.permute(0, 3, 1, 2)
    if rgb_t.shape[-1] != 224:
        rgb_t = F.interpolate(rgb_t, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    rgb_t = (rgb_t - mean) / std

    T = rgb_t.shape[0]
    with torch.no_grad():
        batch = 128
        feats = []
        for i in range(0, T, batch):
            feats.append(backbone(rgb_t[i:i + batch]))
        feats = torch.cat(feats, dim=0)

    L = max_length
    progress = np.zeros(T, dtype=np.float32)
    with torch.no_grad():
        for t in range(T):
            if t < L - 1:
                pad = feats[0:1].expand(L - 1 - t, -1)
                tail = feats[:t + 1]
                window = torch.cat([pad, tail], dim=0)
            else:
                window = feats[t - L + 1:t + 1]
            inp = window.unsqueeze(0)
            pred = model(inp, text_emb)
            if pred.ndim == 3:
                pred = pred.squeeze(-1)
            progress[t] = float(pred[0, -1].item())
    return progress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="directory containing *_camera.npy trajectories")
    ap.add_argument("--ckpt", default="/mnt/tank/uber/Tactile-Reward/ckpt_visual/peg_rgb_multipos.pth")
    ap.add_argument("--text", default="grasp peg and insert to another hole")
    ap.add_argument("--rewind-root",
                    default="/mnt/home/tactile/tactile_isaaclab/external/third-party/ReWiND")
    ap.add_argument("--max-length", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of trajectories scanned (debug)")
    args = ap.parse_args()

    device = torch.device(args.device)
    paths = sorted(glob.glob(os.path.join(args.dir, "*_camera.npy")))
    if args.limit is not None:
        paths = paths[:args.limit]
    print(f"found {len(paths)} *_camera.npy in {args.dir}")

    print(f"loading model {args.ckpt}")
    model = load_rewind_model(args.ckpt, args.rewind_root, args.max_length, device)
    print(f"loading DINOv2 backbone")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print(f"encoding text {args.text!r}")
    text_emb = encode_text(args.text, device).expand(1, -1)

    results = []  # (name, success, T, min, max, mean, last)
    for i, p in enumerate(paths):
        rgb, success = load_traj(p)
        prog = predict_trajectory(rgb, backbone, model, text_emb,
                                  args.max_length, device)
        results.append((
            os.path.basename(p),
            success,
            rgb.shape[0],
            float(prog.min()),
            float(prog.max()),
            float(prog.mean()),
            float(prog[-1]),
        ))
        if (i + 1) % 5 == 0 or i + 1 == len(paths):
            print(f"  scanned {i + 1}/{len(paths)}")

    # Sort by mean ascending — low mean = "model thinks this episode is bad".
    results.sort(key=lambda r: r[5])

    header = f"{'name':<32} {'succ':>4} {'T':>4}  {'min':>6} {'max':>6} {'mean':>6} {'last':>6}"
    print()
    print(header)
    print("-" * len(header))
    for name, success, T, mn, mx, mean, last in results:
        print(f"{name:<32} {success:>4} {T:>4}  {mn:>6.3f} {mx:>6.3f} {mean:>6.3f} {last:>6.3f}")


if __name__ == "__main__":
    main()
