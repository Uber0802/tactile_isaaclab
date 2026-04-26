"""Convert tactile force field PNGs to per-episode MP4 videos.

Left and right images are concatenated side-by-side.

Usage:
    python scripts/tactile_record_to_mp4.py
    python scripts/tactile_record_to_mp4.py --record-dir /path/to/tactile_record --fps 15 --episodes 0 1 2
"""

import argparse
import os
import re

import cv2
import numpy as np


def get_episode_dirs(base_dir: str) -> list[tuple[int, str]]:
    """Return sorted (episode_index, path) pairs under base_dir."""
    entries = []
    for name in os.listdir(base_dir):
        m = re.fullmatch(r"episode(\d+)", name)
        if m:
            entries.append((int(m.group(1)), os.path.join(base_dir, name)))
    entries.sort(key=lambda x: x[0])
    return entries


def get_sorted_pngs(episode_dir: str) -> list[str]:
    """Return PNG paths in a directory sorted by filename."""
    files = [f for f in os.listdir(episode_dir) if f.endswith(".png")]
    files.sort()
    return [os.path.join(episode_dir, f) for f in files]


def make_episode_video(
    left_pngs: list[str],
    right_pngs: list[str],
    out_path: str,
    fps: int,
) -> None:
    """Write an MP4 with left and right frames concatenated horizontally."""
    if not left_pngs and not right_pngs:
        print(f"  [skip] no frames found")
        return

    # Pair frames; use black placeholder if one side is missing.
    n = max(len(left_pngs), len(right_pngs))

    writer = None
    for i in range(n):
        left_img = cv2.imread(left_pngs[i]) if i < len(left_pngs) else None
        right_img = cv2.imread(right_pngs[i]) if i < len(right_pngs) else None

        if left_img is not None:
            left_img = cv2.rotate(left_img, cv2.ROTATE_90_CLOCKWISE)
        if right_img is not None:
            right_img = cv2.rotate(right_img, cv2.ROTATE_90_CLOCKWISE)

        # Determine frame size from whichever side is available.
        if left_img is None and right_img is None:
            continue
        h = (left_img if left_img is not None else right_img).shape[0]
        w = (left_img if left_img is not None else right_img).shape[1]

        if left_img is None:
            left_img = np.zeros((h, w, 3), dtype=np.uint8)
        if right_img is None:
            right_img = np.zeros((h, w, 3), dtype=np.uint8)

        # Resize right to match left if shapes differ.
        if left_img.shape != right_img.shape:
            right_img = cv2.resize(right_img, (left_img.shape[1], left_img.shape[0]))

        frame = np.concatenate([left_img, right_img], axis=1)

        if writer is None:
            fh, fw = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))

        writer.write(frame)

    if writer is not None:
        writer.release()
        print(f"  saved → {out_path}  ({n} frames @ {fps} fps)")
    else:
        print(f"  [skip] could not write video (no valid frames)")


def main():
    parser = argparse.ArgumentParser(description="Tactile record → per-episode MP4")
    parser.add_argument(
        "--record-dir",
        default="/mnt/home/uber/IsaacLab/tactile_record",
        help="Root tactile_record directory",
    )
    parser.add_argument("--fps", type=int, default=15, help="Output video frame rate")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for MP4s (default: <record-dir>/videos)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to process (default: all)",
    )
    args = parser.parse_args()

    left_base = os.path.join(args.record_dir, "tactile_force_field_left")
    right_base = os.path.join(args.record_dir, "tactile_force_field_right")

    if not os.path.isdir(left_base) and not os.path.isdir(right_base):
        raise FileNotFoundError(f"Neither left nor right tactile dir found under {args.record_dir}")

    out_dir = args.out_dir or os.path.join(args.record_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)

    # Collect episode dirs from whichever side exists.
    base_for_listing = left_base if os.path.isdir(left_base) else right_base
    all_episodes = get_episode_dirs(base_for_listing)

    if not all_episodes:
        print("No episode directories found.")
        return

    if args.episodes is not None:
        filter_set = set(args.episodes)
        all_episodes = [(idx, path) for idx, path in all_episodes if idx in filter_set]

    print(f"Processing {len(all_episodes)} episode(s) → {out_dir}")

    for ep_idx, _ in all_episodes:
        ep_name = f"episode{ep_idx}"
        left_dir = os.path.join(left_base, ep_name)
        right_dir = os.path.join(right_base, ep_name)

        left_pngs = get_sorted_pngs(left_dir) if os.path.isdir(left_dir) else []
        right_pngs = get_sorted_pngs(right_dir) if os.path.isdir(right_dir) else []

        out_path = os.path.join(out_dir, f"{ep_name}.mp4")
        print(f"episode {ep_idx:4d}  left={len(left_pngs)} right={len(right_pngs)} frames")
        make_episode_video(left_pngs, right_pngs, out_path, fps=args.fps)


if __name__ == "__main__":
    main()
