"""Convert an RGB trajectory .npy into an mp4.

Usage:
    python npy_to_mp4.py <input.npy> [output.mp4] [--fps 15]
"""
import argparse
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output", nargs="?", default=None)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    loaded = np.load(args.input, allow_pickle=True)
    # Two formats supported:
    #   - raw ndarray (T, H, W, 3) uint8   ← legacy
    #   - dict with "Camera" key            ← new (includes "Success" too)
    if isinstance(loaded, np.ndarray) and loaded.ndim == 4:
        arr = loaded
        success = None
    else:
        payload = loaded.item() if loaded.ndim == 0 else loaded
        arr = payload["Camera"]
        success = payload.get("Success")
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise SystemExit(f"Expected (T, H, W, 3) uint8, got {arr.shape} {arr.dtype}")
    if success is not None:
        print(f"  Success: {success}")

    out_path = args.output or os.path.splitext(args.input)[0] + ".mp4"

    try:
        import imageio.v2 as imageio
        imageio.mimsave(out_path, arr, fps=args.fps, codec="libx264", quality=8)
        print(f"wrote {out_path}  ({arr.shape[0]} frames @ {args.fps} fps)")
    except ImportError:
        import cv2
        H, W = arr.shape[1:3]
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        for frame in arr:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"wrote {out_path}  ({arr.shape[0]} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
