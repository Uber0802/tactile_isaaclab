"""Compare full-resolution AnyTouch2 quiver to the IsaacLab-aligned (20x25) sub-sample.

Per frame layout (1920x1080):
  +--------------+----------+----------+
  |              | left raw | right raw|
  |     RGB      |  STEP=8  |  STEP=8  |
  |              |  (40x30) |  (40x30) |
  |              +----------+----------+
  |              | left 20x25 | right 20x25 |
  |              |  ALIGNED   |   ALIGNED   |
  +--------------+------------+-------------+

Visual style mirrors `AnyTouch2/scripts/extract_force_field_mirrored.py`:
white background, black rest-position dots, red arrows (right) / blue arrows (left),
arrow length scaled by magnitude with a 5% global-max suppression threshold.

Usage:
    python scripts/visualize_aligned_sample.py \\
        --eval_metadata /mnt/tank/uber/Tactile-Reward/tactile_metadata_eval.h5 \\
        --datasets_root /mnt/tank/uber/AnyTouch2/datasets \\
        --output_dir /mnt/tank/uber/Tactile-Reward/aligned_compare \\
        --tasks broom_sweep_fruit pour_water --limit 2
"""
from __future__ import annotations

import os
import re
import sys
import argparse

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm
import torch as th

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.tactile_dataset import TactileReWiNDDataset


def find_rgb_dir(task: str, traj_idx: str, datasets_root: str):
    for d in ("data1", "data2"):
        p = os.path.join(datasets_root, d, task, task, str(traj_idx), "color")
        if os.path.isdir(p):
            return p
    return None


def list_rgb_frames(rgb_dir: str):
    files = [(int(m.group(1)), f) for f, m in
             ((f, re.match(r"^(\d+)\.png$", f)) for f in os.listdir(rgb_dir))
             if m is not None]
    files.sort()
    return [os.path.join(rgb_dir, f) for _, f in files]


def parse_traj_idx(npy_filename: str) -> str:
    m = re.match(r"^(.+)__(\d+)\.npy$", npy_filename)
    return m.group(2) if m else "0"


def sample_grid_step(field: np.ndarray, step: int):
    """field: (T, H, W, 2) -> (T, h, w, 2) plus grid coords (xg, yg)."""
    H, W = field.shape[1], field.shape[2]
    ys = np.arange(step // 2, H, step)
    xs = np.arange(step // 2, W, step)
    xg, yg = np.meshgrid(xs, ys)
    sampled = field[:, yg, xg, :]
    return sampled, xg, yg


def aligned_full_grid_xy(target_h: int, target_w: int, h_full: int, w_full: int):
    """The exact (linspace) sample positions used by `_align_to_isaaclab`.

    Returns xg, yg in the FULL-resolution (h_full, w_full) coordinate space
    so the aligned panel can render at the same scale as the raw panel.
    """
    h_idx = np.round(np.linspace(0, h_full - 1, target_h)).astype(int)
    w_idx = np.round(np.linspace(0, w_full - 1, target_w)).astype(int)
    xg, yg = np.meshgrid(w_idx, h_idx)
    return xg, yg


def to_quiver(u: np.ndarray, v: np.ndarray, mag_global_max: float, step: float,
              mag_threshold_frac: float = 0.05):
    eps = 1e-8
    mag = np.sqrt(u * u + v * v)
    mask = mag >= mag_global_max * mag_threshold_frac
    strength = mag / (mag_global_max + eps)
    arrow_len = step * (0.5 + 2.0 * strength)
    vx = (u / (mag + eps)) * arrow_len * mask
    vy = (v / (mag + eps)) * arrow_len * mask
    return vx, vy


def render_one(
    task: str,
    traj_file: str,
    data_dir: str,
    datasets_root: str,
    out_path: str,
    video_frames: int | None,
    fps: int,
    raw_step: int,
    frame_stride: int | None = None,
):
    npy_path = os.path.join(data_dir, traj_file)
    traj = np.load(npy_path, mmap_mode="r")  # (T_tac, 320, 480, 2)
    T_tac = len(traj)
    if T_tac < 3:
        print(f"  skipping {traj_file}: too short ({T_tac} frames)")
        return

    traj_idx_str = parse_traj_idx(traj_file)
    rgb_dir = find_rgb_dir(task, traj_idx_str, datasets_root)
    if rgb_dir is None:
        print(f"  no RGB dir for {task}/{traj_idx_str} — skipping")
        return
    rgb_frames = list_rgb_frames(rgb_dir)
    M = len(rgb_frames)
    if M == 0:
        print(f"  no PNGs in {rgb_dir} — skipping")
        return

    if frame_stride is not None and frame_stride > 1:
        tac_idx = np.arange(0, T_tac, frame_stride)
        n_video = len(tac_idx)
        rgb_idx = np.round(np.linspace(0, M - 1, n_video)).astype(int)
    else:
        n_video = T_tac if video_frames is None else min(video_frames, T_tac)
        proportions = np.linspace(0, 1, n_video)
        tac_idx = np.round(proportions * (T_tac - 1)).astype(int)
        rgb_idx = np.round(proportions * (M - 1)).astype(int)
    tac_video = np.ascontiguousarray(traj[tac_idx], dtype=np.float32)  # (n_video, 320, 480, 2)

    # Split into hands at full resolution (320, 240) per hand.
    H_full, W_full = tac_video.shape[1], tac_video.shape[2]
    half_w = W_full // 2
    left_full = tac_video[:, :, :half_w, :]
    right_full = tac_video[:, :, half_w:, :]
    Hh, Wh = left_full.shape[1], left_full.shape[2]   # 320, 240

    # Raw quiver grid (STEP=8 sub-sampling, just for rendering — not what the model sees).
    left_step, xg_step, yg_step = sample_grid_step(left_full, raw_step)
    right_step, _, _ = sample_grid_step(right_full, raw_step)

    # Aligned 20x25 grid: transpose first (un-rotate AnyTouch2 build's 90° CCW),
    # then linspace sample. Aligned indices live in the TRANSPOSED (240, 320)
    # frame; we map them back to the original (320, 240) plotting coords so
    # rest dots show *where* the aligned samples come from.
    target_h, target_w = 20, 25
    h_idx_t = np.round(np.linspace(0, 239, target_h)).astype(int)   # 240 axis -> 20
    w_idx_t = np.round(np.linspace(0, 319, target_w)).astype(int)   # 320 axis -> 25

    left_t = left_full.transpose(0, 2, 1, 3)     # (T, 240, 320, 2)
    right_t = right_full.transpose(0, 2, 1, 3)
    left_aligned = left_t[:, h_idx_t[:, None], w_idx_t[None, :], :]      # (T, 20, 25, 2)
    right_aligned = right_t[:, h_idx_t[:, None], w_idx_t[None, :], :]

    # In the original (320, 240) frame: transposed_H = orig_W, transposed_W = orig_H.
    # So aligned[t, i, j] sits at (orig_col = h_idx_t[i], orig_row = w_idx_t[j]).
    xg_aligned = np.tile(h_idx_t[:, None], (1, target_w))   # (20, 25), col coord
    yg_aligned = np.tile(w_idx_t[None, :], (target_h, 1))   # (20, 25), row coord

    # Cross-check against the actual dataset transform.
    aligned_check = TactileReWiNDDataset._align_to_isaaclab(
        th.from_numpy(tac_video).permute(0, 3, 1, 2)
    ).permute(0, 2, 3, 1).numpy()  # (T, 40, 25, 2)
    np.testing.assert_allclose(aligned_check[:, :target_h, :, :], left_aligned, atol=1e-5)

    # Single global threshold so colors / arrow lengths are comparable across panels and frames.
    mag_global_max = max(
        float(np.linalg.norm(left_step, axis=-1).max()),
        float(np.linalg.norm(right_step, axis=-1).max()),
        1e-6,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = plt.figure(figsize=(16, 9), dpi=120)  # 1920x1080
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 3, width_ratios=[1.5, 1, 1], height_ratios=[1, 1],
                  figure=fig,
                  left=0.04, right=0.98, top=0.93, bottom=0.04,
                  wspace=0.10, hspace=0.18)
    ax_rgb = fig.add_subplot(gs[:, 0])
    ax_raw_L = fig.add_subplot(gs[0, 1])
    ax_raw_R = fig.add_subplot(gs[0, 2])
    ax_aligned_L = fig.add_subplot(gs[1, 1])
    ax_aligned_R = fig.add_subplot(gs[1, 2])

    rgb0 = np.array(Image.open(rgb_frames[rgb_idx[0]]).convert("RGB"))
    im_rgb = ax_rgb.imshow(rgb0)
    ax_rgb.set_title(f"RGB — {task}", fontsize=13)
    ax_rgb.set_xticks([]); ax_rgb.set_yticks([])

    def init_panel(ax, xg, yg, color, title):
        ax.set_facecolor("white")
        ax.set_xlim(-1, Wh); ax.set_ylim(Hh, -1)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=11)
        ax.scatter(xg, yg, s=8, c="black", alpha=0.3)
        return ax.quiver(
            xg, yg,
            np.zeros_like(xg, dtype=float), np.zeros_like(xg, dtype=float),
            color=color, angles="xy", scale_units="xy", scale=1,
            width=0.006, headwidth=4, headlength=5, headaxislength=4,
        )

    qL_raw = init_panel(ax_raw_L, xg_step, yg_step, "blue",
                        f"left raw — STEP={raw_step} ({xg_step.shape[0]}x{xg_step.shape[1]})")
    qR_raw = init_panel(ax_raw_R, xg_step, yg_step, "red",
                        f"right raw — STEP={raw_step} ({xg_step.shape[0]}x{xg_step.shape[1]})")
    qL_align = init_panel(ax_aligned_L, xg_aligned, yg_aligned, "blue",
                          f"left aligned — sampled to {target_h}x{target_w}")
    qR_align = init_panel(ax_aligned_R, xg_aligned, yg_aligned, "red",
                          f"right aligned — sampled to {target_h}x{target_w}")

    fig.suptitle(f"{task} / traj {traj_idx_str}   |   "
                 f"320x240 markers per hand → 20x25 (long↔long: H 320→20 stride~16, "
                 f"W 240→25 stride~9-10)",
                 fontsize=11)

    fig.canvas.draw()
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                quality=8, macro_block_size=1)
    try:
        for t in range(n_video):
            rgb = np.array(Image.open(rgb_frames[rgb_idx[t]]).convert("RGB"))
            im_rgb.set_data(rgb)
            vxL, vyL = to_quiver(left_step[t, :, :, 0], left_step[t, :, :, 1],
                                 mag_global_max, raw_step)
            vxR, vyR = to_quiver(right_step[t, :, :, 0], right_step[t, :, :, 1],
                                 mag_global_max, raw_step)
            qL_raw.set_UVC(vxL.ravel(), (-vyL).ravel())
            qR_raw.set_UVC(vxR.ravel(), (-vyR).ravel())

            # Aligned panel: scale by an effective step that makes arrow lengths
            # visually comparable (10 px between samples in W ≈ raw step).
            aligned_step_eff = float(np.mean([
                Hh / target_h, Wh / target_w,
            ]))  # ≈ (16 + 9.6) / 2 ≈ 12.8
            vxLa, vyLa = to_quiver(left_aligned[t, :, :, 0], left_aligned[t, :, :, 1],
                                   mag_global_max, aligned_step_eff)
            vxRa, vyRa = to_quiver(right_aligned[t, :, :, 0], right_aligned[t, :, :, 1],
                                   mag_global_max, aligned_step_eff)
            qL_align.set_UVC(vxLa.ravel(), (-vyLa).ravel())
            qR_align.set_UVC(vxRa.ravel(), (-vyRa).ravel())

            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(buf.copy())
    finally:
        writer.close()
        plt.close(fig)


def main(args):
    with h5py.File(args.eval_metadata, "r") as h5:
        data_dir = args.data_dir_override or h5.attrs["data_dir"]
        all_tasks = sorted(h5.keys())
        meta = {
            t: [
                s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                for s in np.asarray(h5[t]["trajectory_files"])
            ]
            for t in all_tasks
        }

    if args.tasks:
        all_tasks = [t for t in all_tasks if t in set(args.tasks)]

    pairs = []
    for task in all_tasks:
        for traj_file in meta.get(task, []):
            pairs.append((task, traj_file))
    if args.limit:
        pairs = pairs[: args.limit]

    print(f"rendering {len(pairs)} comparison videos to {args.output_dir}")
    for task, traj_file in tqdm(pairs):
        out_path = os.path.join(
            args.output_dir,
            f"{task}__{parse_traj_idx(traj_file)}__compare.mp4",
        )
        if os.path.exists(out_path) and not args.overwrite:
            continue
        try:
            render_one(
                task=task,
                traj_file=traj_file,
                data_dir=data_dir,
                datasets_root=args.datasets_root,
                out_path=out_path,
                video_frames=args.video_frames,
                fps=args.fps,
                raw_step=args.raw_step,
                frame_stride=args.frame_stride,
            )
        except Exception as e:
            print(f"  ERROR rendering {task}/{traj_file}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_metadata",
                    default="/mnt/tank/uber/Tactile-Reward/tactile_metadata_eval.h5")
    ap.add_argument("--data_dir_override", default=None)
    ap.add_argument("--datasets_root",
                    default="/mnt/tank/uber/AnyTouch2/datasets")
    ap.add_argument("--output_dir",
                    default="/mnt/tank/uber/Tactile-Reward/aligned_compare")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Filter to these task names.")
    ap.add_argument("--limit", type=int, default=3,
                    help="Cap number of (task, traj) pairs rendered.")
    ap.add_argument("--video_frames", type=int, default=240,
                    help="Frames per output video (subsampled evenly from full traj). "
                         "Ignored if --frame_stride is set.")
    ap.add_argument("--frame_stride", type=int, default=None,
                    help="Render every Nth tactile frame (5x speedup at stride=5). "
                         "Overrides --video_frames.")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--raw_step", type=int, default=8,
                    help="Stride used for the raw-panel quiver (visualization only).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    main(args)
