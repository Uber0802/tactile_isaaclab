"""Raw-tactile ReWiND dataset (no precomputed embeddings).

Reads a small metadata H5 produced by `scripts/anytouch2_to_h5.py`,
then memory-maps the actual `(N, 320, 480, 2)` npy trajectories on the
fly so we never materialise the full ~178 GB dataset in RAM.

Each `__getitem__` returns one sample with rewind / negative-text augmentation
matching the original ReWiND dataset.py.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List

import h5py
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset


class TactileReWiNDDataset(Dataset):
    """ReWiND-style sampler over AnyTouch2 tactile trajectories.

    Output schema per item:
      video_array : float32  (max_length, 2, 320, 480)
      text_array  : float32  (text_dim,)
      progress    : float32  (max_length,)
      class_label : float32  (max_length,)   # 1 = positive sample, 0 = negative
    """

    DEFAULT_MULTISCALE_RESOLUTIONS = (
        # 4:5 aspect ratio
        (8, 10), (20, 25), (40, 50), (60, 75), (80, 100),
        (120, 150), (160, 200), (200, 250),
        # 3:4 aspect ratio
        (9, 12), (15, 20), (30, 40), (60, 80),
        (90, 120), (120, 160), (180, 240), (240, 320),
        # extra: matches IsaacLab-shape height with wider aspect
        (20, 35),
    )

    def __init__(
        self,
        metadata_h5_path: str,
        max_length: int = 16,
        rewind: bool = True,
        rewind_ratio: float = 0.8,
        sample_neg: bool = True,
        neg_ratio: float = 0.2,
        epoch_steps: int = 200,
        batch_size: int = 16,
        data_dir_override: str | None = None,
        align_to_isaaclab: bool = False,
        multiscale_align: bool = False,
        multiscale_resolutions=None,
    ):
        self.metadata_path = metadata_h5_path
        self.max_length = max_length
        self.rewind = rewind
        self.rewind_ratio = rewind_ratio
        self.sample_neg = sample_neg
        self.neg_ratio = neg_ratio
        self.epoch_steps = epoch_steps
        self.batch_size = batch_size
        self.align_to_isaaclab = align_to_isaaclab
        self.multiscale_align = multiscale_align
        self.multiscale_resolutions = list(
            multiscale_resolutions or self.DEFAULT_MULTISCALE_RESOLUTIONS
        )
        # Set per-call by __getitem__ so workers don't need shared state.
        self._current_target = (20, 25)

        # Load all metadata up front into plain Python objects so DataLoader
        # workers do not need to share the H5 handle.
        self._tasks: List[str] = []
        self._lang_emb: Dict[str, np.ndarray] = {}
        self._traj_files: Dict[str, List[str]] = {}
        with h5py.File(metadata_h5_path, "r") as h5:
            self._data_dir = data_dir_override or h5.attrs["data_dir"]
            data_already_aligned = bool(h5.attrs.get("data_already_aligned", False))
            for task in sorted(h5.keys()):
                grp = h5[task]
                self._tasks.append(task)
                self._lang_emb[task] = np.asarray(grp["minilm_lang_embedding"], dtype=np.float32)
                self._traj_files[task] = [
                    s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in np.asarray(grp["trajectory_files"])
                ]
        if not self._tasks:
            raise RuntimeError(f"no tasks found in {metadata_h5_path}")

        # If the metadata declares the npy files are already aligned, skip the
        # on-the-fly transform regardless of how the caller set the flag.
        if data_already_aligned and self.align_to_isaaclab:
            print(f"[TactileReWiNDDataset] {metadata_h5_path} marks "
                  f"data_already_aligned=True; disabling on-the-fly alignment.")
            self.align_to_isaaclab = False

    @property
    def tasks(self) -> List[str]:
        return list(self._tasks)

    def __len__(self) -> int:
        return self.batch_size * self.epoch_steps

    def _sample_text(self, task: str) -> th.Tensor:
        emb = self._lang_emb[task]
        idx = random.randint(0, emb.shape[0] - 1) if emb.shape[0] > 1 else 0
        return th.from_numpy(emb[idx]).float()

    def _sample_negative_text(self, task: str) -> th.Tensor:
        if len(self._tasks) == 1:
            return self._sample_text(task)
        other = task
        while other == task:
            other = random.choice(self._tasks)
        return self._sample_text(other)

    def _load_traj(self, task: str) -> np.ndarray:
        traj_file = random.choice(self._traj_files[task])
        path = os.path.join(self._data_dir, traj_file)
        return np.load(path, mmap_mode="r")  # (N, 320, 480, 2) float16

    def _resize_indices(self, traj_idx: np.ndarray, progress: np.ndarray):
        # Pad/subsample in index space so the float32 buffer is only ever
        # materialised at max_length frames, not the full traj length.
        T = traj_idx.shape[0]
        if T < self.max_length:
            pad = self.max_length - T
            traj_idx = np.concatenate([traj_idx, np.full(pad, traj_idx[-1], dtype=traj_idx.dtype)])
            progress = np.concatenate([progress, np.full(pad, progress[-1], dtype=progress.dtype)])
        elif T > self.max_length:
            local = np.linspace(0, T - 1, self.max_length).astype(int)
            traj_idx = traj_idx[local]
            progress = progress[local]
        return traj_idx, progress

    def _to_torch_frames(self, traj: np.ndarray, traj_idx: np.ndarray) -> th.Tensor:
        frames_thwc = traj[traj_idx].astype(np.float32, copy=True)
        x = th.from_numpy(frames_thwc).permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
        if self.align_to_isaaclab:
            target_h, target_w = self._current_target
            x = self._align_to_isaaclab(x, target_h=target_h, target_w=target_w)
        return x

    @staticmethod
    def _align_to_isaaclab(x: th.Tensor,
                            target_h: int = 20, target_w: int = 25) -> th.Tensor:
        """AnyTouch2 (T, C, 320, 480) -> bimanual-H-stacked (T, C, 2*target_h, target_w).

        Pipeline:
          1. Split width into two hands -> each (T, C, 320, 240).
          2. Transpose H<->W so the long axis (raw H=320) lands on W,
             matching IsaacLab's (20, 25) convention.
          3. Linspace-sample each hand to (target_h, target_w) covering the
             full input range.
          4. Concatenate along H (top=left, bottom=right) -> (2*target_h, target_w).

        With defaults (20, 25), output is (T, C, 40, 25). Multi-scale training
        passes other (target_h, target_w) on different batches.

        NOTE: vector channels are NOT swapped. The 90° CCW rotation in
        build_tactile_dataset.py also rotated the vectors via (Fx, Fy) ->
        (-Fy, Fx); we leave that as-is and let the network learn the mapping.
        """
        _, _, _, W = x.shape
        if W % 2 != 0:
            raise ValueError(f"expected even width to split bimanual, got W={W}")
        half_w = W // 2
        left = x[..., :half_w].transpose(2, 3)    # (T, C, 240, 320)
        right = x[..., half_w:].transpose(2, 3)
        Hh, Wh = left.shape[-2], left.shape[-1]   # 240, 320

        if Hh < target_h or Wh < target_w:
            raise ValueError(f"per-hand input ({Hh}, {Wh}) smaller than target "
                             f"({target_h}, {target_w}); cannot sub-sample.")

        h_idx = th.linspace(0, Hh - 1, target_h, device=x.device).round().long()
        w_idx = th.linspace(0, Wh - 1, target_w, device=x.device).round().long()
        left = left[..., h_idx[:, None], w_idx[None, :]]    # (T, C, target_h, target_w)
        right = right[..., h_idx[:, None], w_idx[None, :]]
        return th.cat([left, right], dim=2).contiguous()    # (T, C, 2*target_h, target_w)

    def _sample_forward(self, traj: np.ndarray):
        N = len(traj)
        start = random.randint(0, N - 3)
        end = random.randint(start + 3, N)
        full_len = N - start
        traj_idx = np.arange(start, end, dtype=np.int64)
        progress = ((np.arange(end - start) + 1) / full_len).astype(np.float32)
        traj_idx, progress = self._resize_indices(traj_idx, progress)
        frames = self._to_torch_frames(traj, traj_idx)
        return frames, progress

    def _sample_rewind(self, traj: np.ndarray):
        N = len(traj)
        # Pick a forward window straddling the midpoint, long enough to rewind.
        for _ in range(8):
            start = random.randint(0, N // 2)
            end = random.randint(N // 2, N)
            if end - start >= 3:
                break
        else:
            return self._sample_forward(traj)

        full_len = N - start
        fwd_len = end - start
        fwd_idx = np.arange(start, end, dtype=np.int64)
        fwd_progress = ((np.arange(fwd_len) + 1) / full_len).astype(np.float32)

        # Random truncation of the reverse playback (matches original dataset.py:
        # clip[::-1][1:rev_end] → traj indices end-2, end-3, ..., end-rev_end).
        rev_end = random.randint(2, fwd_len)
        rev_idx = np.arange(end - 2, end - 1 - rev_end, -1, dtype=np.int64)
        rev_progress = fwd_progress[::-1][1:rev_end].copy()

        traj_idx = np.concatenate([fwd_idx, rev_idx])
        progress = np.concatenate([fwd_progress, rev_progress]).astype(np.float32)

        traj_idx, progress = self._resize_indices(traj_idx, progress)
        frames = self._to_torch_frames(traj, traj_idx)
        return frames, progress

    def __getitem__(self, idx):
        # Decide the alignment target shape for this batch. All samples in the
        # same batch share a target so the dataloader can stack them. Workers
        # don't need shared state because the resolution is purely a function
        # of `idx // batch_size`.
        if self.align_to_isaaclab and self.multiscale_align:
            group = idx // self.batch_size
            self._current_target = self.multiscale_resolutions[
                group % len(self.multiscale_resolutions)
            ]
        else:
            self._current_target = (20, 25)

        task = random.choice(self._tasks)
        traj = self._load_traj(task)
        if len(traj) < 3:
            # Defensive fallback: try another task.
            return self.__getitem__((idx + 1) % len(self))

        if self.rewind and random.random() < self.rewind_ratio:
            frames, progress = self._sample_rewind(traj)
        else:
            frames, progress = self._sample_forward(traj)

        label = np.ones_like(progress, dtype=np.float32)
        if self.sample_neg and random.random() < self.neg_ratio:
            text = self._sample_negative_text(task)
            progress = np.zeros_like(progress)
            label = np.zeros_like(label)
        else:
            text = self._sample_text(task)

        return {
            "video_array": frames,
            "text_array": text,
            "progress": th.from_numpy(progress),
            "class_label": th.from_numpy(label),
        }
