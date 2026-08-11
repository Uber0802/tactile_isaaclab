"""Cache frozen DINOv2 ViT-B/14 features for `*_camera.npy` rollout episodes.

The visual reward model (ReWiND) never fine-tunes the backbone, so every epoch
of training would otherwise re-encode the same frames. Encoding once to disk
turns a ~10 min/epoch job into a ~10 s/epoch one, and the cache is reusable
across hyperparameter sweeps and tasks.

Preprocessing is byte-for-byte what `ForgeEnv._compute_visual_reward` does at
RL time (forge_env.py:545) and what `evaluate_rgb_reward.py` does offline:
    uint8 RGB -> /255 -> (T,3,H,W) -> bilinear resize to 224 if needed
    -> ImageNet mean/std -> dinov2_vitb14 -> (T, 768) CLS features.

Output mirrors the input tree:
    <cache_dir>/<relative dir>/<ep name>.npz   {"feat": (T,768) float16,
                                                "success": int,
                                                "src": original path}

Usage:
    python scripts/precompute_dino_features.py \\
        --data_dirs /mnt/scratch/tactile/gearpickplace_curriculum_seed2_paired_multipos \\
        --cache_dir /mnt/scratch/tactile/dino_cache/gear_seed2_multipos
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True,
                    help="Dataset roots. Globbed recursively for `*_camera.npy`.")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--backbone", default="dinov2_vitb14",
                    help="Must match FORGE_VISUAL_REWARD_BACKBONE at RL time.")
    ap.add_argument("--frame_batch", type=int, default=64,
                    help="Frames per backbone forward. 64 fits comfortably in 8 GB.")
    ap.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-encode episodes whose .npz already exists.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if args.amp != "off" and device.type == "cuda":
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    files: list[tuple[str, str]] = []       # (abs path, key relative to its root)
    for d in args.data_dirs:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"data_dir does not exist: {d}")
        root_name = os.path.basename(d.rstrip("/"))
        for p in glob.glob(os.path.join(d, "**", "*_camera.npy"), recursive=True):
            files.append((p, os.path.join(root_name, os.path.relpath(p, d))))
    files.sort()
    print(f"[dino] found {len(files)} camera episodes across {len(args.data_dirs)} dir(s)")
    if not files:
        sys.exit("[dino] nothing to do")

    todo = []
    for p, key in files:
        out = os.path.join(args.cache_dir, os.path.splitext(key)[0] + ".npz")
        if not args.overwrite and os.path.exists(out):
            continue
        todo.append((p, out))
    print(f"[dino] {len(todo)} to encode, {len(files) - len(todo)} already cached")
    if not todo:
        return

    print(f"[dino] loading backbone {args.backbone}")
    backbone = torch.hub.load("facebookresearch/dinov2", args.backbone)
    backbone = backbone.to(device).eval()
    for prm in backbone.parameters():
        prm.requires_grad = False

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    n_frames = 0
    for src, out in tqdm(todo, desc="encoding"):
        try:
            d = np.load(src, allow_pickle=True).item()
        except Exception as e:
            print(f"  skip {src}: {e}", file=sys.stderr)
            continue
        if "Camera" not in d:
            continue
        rgb = d["Camera"]                                   # (T, H, W, 3) uint8
        T = rgb.shape[0]

        feats = []
        with torch.no_grad(), torch.amp.autocast(
                device_type="cuda", dtype=amp_dtype,
                enabled=amp_dtype is not None and device.type == "cuda"):
            for i in range(0, T, args.frame_batch):
                chunk = torch.from_numpy(rgb[i:i + args.frame_batch]).to(device)
                chunk = chunk.float().div_(255.0).permute(0, 3, 1, 2)
                if chunk.shape[-1] != 224 or chunk.shape[-2] != 224:
                    chunk = F.interpolate(chunk, size=(224, 224), mode="bilinear",
                                          align_corners=False)
                chunk = (chunk - mean) / std
                feats.append(backbone(chunk).float())
        feat = torch.cat(feats, dim=0).cpu().numpy().astype(np.float16)   # (T, 768)

        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, feat=feat, success=np.int64(d.get("Success", 0)), src=src)
        n_frames += T

    print(f"[dino] done. encoded {n_frames} frames -> {args.cache_dir}")


if __name__ == "__main__":
    main()
