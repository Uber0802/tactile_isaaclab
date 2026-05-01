"""Pre-compute IsaacLab-aligned AnyTouch2 npy files (one-shot).

For each raw (T, 320, 480, 2) npy under --src, runs the same alignment
transform `TactileReWiNDDataset(align_to_isaaclab=True)` applies on the fly
and saves the (T, 40, 25, 2) result under --dst as float16. Also writes
sibling metadata H5 files pointing at --dst with `data_already_aligned=True`
so the dataloader can skip the transform.

After this:
  * `data_dir` of new metadata    -> --dst
  * each saved npy is ~50 KB (vs ~1.2 GB raw)
  * dataloader reads small files, no transform work, GPU stops starving

Usage:
    python scripts/precompute_aligned.py
    # then train with the *_aligned.h5 metadata files
"""
from __future__ import annotations

import os
import sys
import shutil
import argparse
from pathlib import Path

import numpy as np
import torch as th
import h5py
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from tools.tactile_dataset import TactileReWiNDDataset


def align_traj_np(traj_thwc: np.ndarray) -> np.ndarray:
    """(T, 320, 480, 2) float16/float32  ->  (T, 40, 25, 2) float16."""
    x = th.from_numpy(np.ascontiguousarray(traj_thwc, dtype=np.float32))
    x = x.permute(0, 3, 1, 2).contiguous()                  # (T, C, H, W)
    x = TactileReWiNDDataset._align_to_isaaclab(x)          # (T, C, 40, 25)
    return x.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float16)  # (T, 40, 25, 2)


def main(args):
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in os.listdir(src) if f.endswith(".npy"))
    if args.limit:
        files = files[: args.limit]
    print(f"aligning {len(files)} files: {src} -> {dst}")

    bytes_in = bytes_out = 0
    for f in tqdm(files):
        in_path = src / f
        out_path = dst / f
        if out_path.exists() and not args.overwrite:
            bytes_in += in_path.stat().st_size
            bytes_out += out_path.stat().st_size
            continue
        traj = np.load(in_path)
        if traj.ndim != 4 or traj.shape[1:] != (320, 480, 2):
            print(f"  skipping {f}: shape {traj.shape}")
            continue
        aligned = align_traj_np(traj)
        np.save(out_path, aligned)
        bytes_in += in_path.stat().st_size
        bytes_out += out_path.stat().st_size

    ratio = bytes_in / max(bytes_out, 1)
    print(f"\ntotal raw  : {bytes_in / 1e9:.2f} GB")
    print(f"total align: {bytes_out / 1e6:.2f} MB ({ratio:.0f}x reduction)")

    # Build new metadata H5s pointing at --dst with data_already_aligned=True.
    md_dir = Path(args.metadata_dir)
    for split in ("train", "eval"):
        src_md = md_dir / f"tactile_metadata_{split}.h5"
        dst_md = md_dir / f"tactile_metadata_{split}_aligned.h5"
        if not src_md.exists():
            print(f"  no source metadata: {src_md} (skipping)")
            continue
        shutil.copy2(src_md, dst_md)
        with h5py.File(dst_md, "a") as h5:
            h5.attrs["data_dir"] = str(dst)
            h5.attrs["data_already_aligned"] = True
        print(f"wrote {dst_md}  (data_dir={dst}, data_already_aligned=True)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/tank/uber/AnyTouch2/tactile_dataset",
                    help="Raw AnyTouch2 npy directory.")
    ap.add_argument("--dst", default="/mnt/tank/uber/Tactile-Reward/aligned_npy",
                    help="Output directory for pre-aligned npy.")
    ap.add_argument("--metadata_dir", default="/mnt/tank/uber/Tactile-Reward",
                    help="Where existing metadata H5 files live; will write *_aligned.h5 here.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap files processed (for smoke testing).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    main(args)
