"""Render per-trajectory eval MP4s.

Each MP4 layout (per frame):
  +-------------------+-----------+-----------+
  |       RGB         | tactile R | tactile L |
  +-------------------+-----------+-----------+
  |        predicted progress curve           |
  +-------------------------------------------+

Inputs:
  * a tactile-reward checkpoint
  * the eval metadata H5
  * the AnyTouch2 raw `datasets/` root, where RGB lives at
    `data{1,2}/{task}/{task}/{traj_idx}/color/{frame}.png`

Tactile and RGB have different frame counts; we sample both at
`max_length` evenly-spaced positions (proportional alignment).

Usage:
    python scripts/visualize_eval.py \\
        --ckpt /mnt/tank/uber/Tactile-Reward/checkpoints/tactile_rewind_epoch19.pth \\
        --datasets_root /mnt/tank/uber/AnyTouch2/datasets \\
        --output_dir /mnt/tank/uber/Tactile-Reward/eval_videos \\
        --limit 10
"""
from __future__ import annotations

import os
import re
import sys
import argparse

import numpy as np
import torch
import h5py
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.tactile_model import TactileReWiNDTransformer


def load_model(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("args", {})
    model = TactileReWiNDTransformer(
        max_length=cfg.get("max_length", 16),
        text_dim=384,
        hidden_dim=cfg.get("hidden_dim", 512),
        num_heads=cfg.get("num_heads", 8),
        num_layers=cfg.get("num_layers", 4),
        per_hand_dim=cfg.get("per_hand_dim", 384),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, cfg


def find_rgb_dir(task: str, traj_idx: str, datasets_root: str):
    for d in ("data1", "data2"):
        p = os.path.join(datasets_root, d, task, task, str(traj_idx), "color")
        if os.path.isdir(p):
            return p
    return None


def list_rgb_frames(rgb_dir: str):
    files = [f for f in os.listdir(rgb_dir) if f.endswith(".png")]
    keyed = []
    for f in files:
        m = re.match(r"^(\d+)\.png$", f)
        if m:
            keyed.append((int(m.group(1)), f))
    keyed.sort()
    return [os.path.join(rgb_dir, f) for _, f in keyed]


def parse_traj_idx(npy_filename: str) -> str:
    m = re.match(r"^(.+)__(\d+)\.npy$", npy_filename)
    return m.group(2) if m else "0"


def to_chw_tensor(frames_thwc: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(frames_thwc, dtype=np.float32))
    x = x.permute(0, 3, 1, 2).contiguous()
    return x.unsqueeze(0).to(device)


def sample_grid(field: np.ndarray, step: int):
    """field: (T, H, W, 2). Returns (T, h_grid, w_grid, 2) plus the grid coords.

    Mirrors AnyTouch2's `extract_force_field_data2.py` STEP-stride sampling.
    """
    H, W = field.shape[1], field.shape[2]
    ys = np.arange(step // 2, H, step)
    xs = np.arange(step // 2, W, step)
    xg, yg = np.meshgrid(xs, ys)
    sampled = field[:, yg, xg, :]
    return sampled, xg, yg


def to_quiver_uv(uv: np.ndarray, mag_global_max: float, step: int,
                 mag_threshold_frac: float = 0.05):
    """Match AnyTouch2 marker style: arrow length = STEP * (0.5 + 2.0 * strength),
    suppressed below `mag_threshold_frac` of the global max."""
    u = uv[..., 0]
    v = uv[..., 1]
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
    text_emb: np.ndarray,
    text_str: str,
    data_dir: str,
    datasets_root: str,
    model: TactileReWiNDTransformer,
    device: torch.device,
    max_length: int,
    fps: int,
    out_path: str,
    video_frames: int | None = None,
    frame_stride: int | None = None,
    middle_frames: int | None = None,
    step: int = 8,
    mag_threshold_frac: float = 0.05,
):
    npy_path = os.path.join(data_dir, traj_file)
    traj = np.load(npy_path, mmap_mode="r")  # (T_tac, 320, 480, 2) float16
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

    # Default: render every tactile frame. --middle_frames takes N consecutive frames
    # centered on the trajectory midpoint; --frame_stride takes every Nth tactile frame;
    # --video_frames caps the total to N evenly-spaced frames.
    if middle_frames is not None:
        n_video = min(middle_frames, T_tac)
        start = (T_tac - n_video) // 2
        tac_idx = np.arange(start, start + n_video)
        rgb_idx = np.round(np.linspace(
            start / max(T_tac - 1, 1) * (M - 1),
            (start + n_video - 1) / max(T_tac - 1, 1) * (M - 1),
            n_video,
        )).astype(int)
    elif frame_stride is not None:
        tac_idx = np.arange(0, T_tac, frame_stride)
        n_video = len(tac_idx)
        rgb_idx = np.round(np.linspace(0, M - 1, n_video)).astype(int)
    else:
        n_video = T_tac if video_frames is None else min(video_frames, T_tac)
        proportions = np.linspace(0, 1, n_video)
        tac_idx = np.round(proportions * (T_tac - 1)).astype(int)
        rgb_idx = np.round(proportions * (M - 1)).astype(int)
    tac_video = np.ascontiguousarray(traj[tac_idx], dtype=np.float32)  # (n_video, 320, 480, 2)

    # Model only sees max_length frames; sample evenly from the rendered video.
    model_idx_in_video = np.linspace(0, n_video - 1, max_length).astype(int)
    tac_model = tac_video[model_idx_in_video]
    x = to_chw_tensor(tac_model, device)
    text = torch.from_numpy(text_emb).float().unsqueeze(0).to(device)
    with torch.no_grad():
        progress_sparse = model(x, text).squeeze(-1).squeeze(0).cpu().numpy()
    progress_full = np.interp(np.arange(n_video), model_idx_in_video, progress_sparse)

    # AnyTouch2 stores each frame as [right | left] concatenated along width.
    # Keep that ordering here so the rendered panels match the reference scripts.
    H, W = tac_video.shape[1], tac_video.shape[2]
    right_full = tac_video[:, :, : W // 2, :]
    left_full = tac_video[:, :, W // 2 :, :]
    right_grid, xg_r, yg_r = sample_grid(right_full, step)
    left_grid, xg_l, yg_l = sample_grid(left_full, step)
    Hh, Wh = left_full.shape[1], left_full.shape[2]

    mag_global_max = max(
        float(np.linalg.norm(left_grid, axis=-1).max()),
        float(np.linalg.norm(right_grid, axis=-1).max()),
        1e-6,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = plt.figure(figsize=(16, 9), dpi=120)  # 1920x1080
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 3, height_ratios=[3.2, 1], width_ratios=[1.6, 1, 1],
                  figure=fig,
                  left=0.04, right=0.98, top=0.93, bottom=0.10,
                  wspace=0.10, hspace=0.30)
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[0, 2])
    ax_curve = fig.add_subplot(gs[1, :])

    rgb0 = np.array(Image.open(rgb_frames[rgb_idx[0]]).convert("RGB"))
    im_rgb = ax_rgb.imshow(rgb0)
    ax_rgb.set_title(f"RGB — {task}", fontsize=12)
    ax_rgb.set_xticks([]); ax_rgb.set_yticks([])

    def init_tactile_panel(ax, xg, yg, color, title):
        ax.set_facecolor("white")
        ax.set_xlim(-1, Wh)
        ax.set_ylim(Hh, -1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12)
        ax.scatter(xg, yg, s=10, c="black", alpha=0.3)
        return ax.quiver(
            xg, yg,
            np.zeros_like(xg, dtype=float), np.zeros_like(xg, dtype=float),
            color=color, angles="xy", scale_units="xy", scale=1,
            width=0.006, headwidth=4, headlength=5, headaxislength=4,
        )

    qR = init_tactile_panel(ax_right, xg_r, yg_r, "red",
                            f"right hand ({Hh}×{Wh}, step={step})")
    qL = init_tactile_panel(ax_left,  xg_l, yg_l, "blue",
                            f"left hand  ({Hh}×{Wh}, step={step})")

    t_axis = np.arange(n_video)
    ax_curve.plot(t_axis, progress_full, "-", color="tab:blue", lw=2.0, alpha=0.5,
                  label="predicted progress")
    ax_curve.plot(model_idx_in_video, progress_sparse, "o", color="tab:blue",
                  markersize=4, alpha=0.6, label="model samples")
    cur_dot, = ax_curve.plot([t_axis[0]], [progress_full[0]], "o",
                             color="tab:red", markersize=10, label="current frame")
    ax_curve.set_xlim(-0.5, n_video - 0.5)
    ax_curve.set_ylim(-0.05, 1.05)
    ax_curve.set_xlabel(f"video frame (trajectory: {T_tac} → {n_video} rendered)")
    ax_curve.set_ylabel("progress")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc="upper left", fontsize=10)
    ax_curve.set_title(f"instruction: {text_str!r}   |   "
                       f"mean(progress[1:])={progress_sparse[1:].mean():.3f}", fontsize=11)

    fig.canvas.draw()

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                quality=8, macro_block_size=1)
    try:
        for t in range(n_video):
            rgb = np.array(Image.open(rgb_frames[rgb_idx[t]]).convert("RGB"))
            im_rgb.set_data(rgb)
            vxL, vyL = to_quiver_uv(left_grid[t],  mag_global_max, step, mag_threshold_frac)
            vxR, vyR = to_quiver_uv(right_grid[t], mag_global_max, step, mag_threshold_frac)
            qL.set_UVC(vxL.ravel(), (-vyL).ravel())
            qR.set_UVC(vxR.ravel(), (-vyR).ravel())
            cur_dot.set_data([t_axis[t]], [progress_full[t]])
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(buf.copy())
    finally:
        writer.close()
        plt.close(fig)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)
    max_length = cfg.get("max_length", 16)
    print(f"loaded {args.ckpt} (max_length={max_length})")

    with h5py.File(args.eval_metadata, "r") as h5:
        data_dir = args.data_dir_override or h5.attrs["data_dir"]
        tasks = sorted(h5.keys())
        meta = {}
        for task in tasks:
            grp = h5[task]
            instructions = [
                s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                for s in np.asarray(grp["instructions"])
            ]
            traj_files = [
                s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                for s in np.asarray(grp["trajectory_files"])
            ]
            meta[task] = {
                "lang": np.asarray(grp["minilm_lang_embedding"], dtype=np.float32),
                "instructions": instructions,
                "trajs": traj_files,
            }

    if args.tasks:
        tasks = [t for t in tasks if t in set(args.tasks)]

    pairs = []
    for task in tasks:
        for traj_file in meta[task]["trajs"]:
            pairs.append((task, traj_file))
    if args.limit:
        pairs = pairs[: args.limit]

    print(f"rendering {len(pairs)} videos to {args.output_dir}")

    for task, traj_file in tqdm(pairs):
        out_path = os.path.join(
            args.output_dir,
            f"{task}__{parse_traj_idx(traj_file)}.mp4",
        )
        if os.path.exists(out_path) and not args.overwrite:
            continue
        text_emb = meta[task]["lang"][0]
        text_str = (meta[task]["instructions"][0]
                    if meta[task]["instructions"] else task.replace("_", " "))
        try:
            render_one(
                task=task,
                traj_file=traj_file,
                text_emb=text_emb,
                text_str=text_str,
                data_dir=data_dir,
                datasets_root=args.datasets_root,
                model=model,
                device=device,
                max_length=max_length,
                fps=args.fps,
                out_path=out_path,
                video_frames=args.video_frames,
                frame_stride=args.frame_stride,
                middle_frames=args.middle_frames,
                step=args.step,
                mag_threshold_frac=args.mag_threshold_frac,
            )
        except Exception as e:
            print(f"  ERROR rendering {task}/{traj_file}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval_metadata",
                    default="/mnt/tank/uber/Tactile-Reward/tactile_metadata_eval.h5")
    ap.add_argument("--data_dir_override", default=None)
    ap.add_argument("--datasets_root",
                    default="/mnt/tank/uber/AnyTouch2/datasets",
                    help="Root containing data1/ and data2/ with RGB PNG sequences.")
    ap.add_argument("--output_dir",
                    default="/mnt/tank/uber/Tactile-Reward/eval_videos")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Only render these task names.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap total videos rendered (for fast iteration).")
    ap.add_argument("--fps", type=int, default=20,
                    help="Output video frame rate.")
    ap.add_argument("--video_frames", type=int, default=None,
                    help="If set, render this many evenly-spaced frames. Default: every "
                         "frame of the trajectory.")
    ap.add_argument("--frame_stride", type=int, default=None,
                    help="If set, render every Nth tactile frame. Overrides --video_frames.")
    ap.add_argument("--middle_frames", type=int, default=None,
                    help="If set, render N consecutive frames centered on the trajectory "
                         "midpoint. Overrides --frame_stride and --video_frames.")
    ap.add_argument("--step", type=int, default=8,
                    help="Marker grid stride (matches AnyTouch2 extract_force_field_data2.py).")
    ap.add_argument("--mag_threshold_frac", type=float, default=0.05,
                    help="Suppress arrows below this fraction of the global max magnitude.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    main(args)
