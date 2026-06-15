"""Single-task overfit on the IsaacLab Forge tactile dataset.

Each `data_2/ep*.npy` is a dict produced by the patched forge_env._flush_tactile_episode:
    {"Task": str, "Tactile": (T, 40, 25, 3) float16, "Success": int 0/1}

Behaviour:
  * Successful trajectories: forward sample with linear progress 0 -> 1,
    plus rewind augmentation (matching ReWiND).
  * Failed trajectories: progress is identically 0 across all frames.
  * Single task — instructions sampled from a fixed paraphrase pool.
  * Eval happens on the SAME data (overfit check); we just want to confirm
    the model can fit this distribution before bothering with held-out splits.

Usage:
    python scripts/train_isaaclab_overfit.py \\
        --data_dir /mnt/home/uber/tactile_isaaclab/tactile_dataset/data_2 \\
        --epochs 20 --batch_size 16 --steps_per_epoch 100 \\
        --ckpt_dir /mnt/tank/uber/Tactile-Reward/checkpoints_isaaclab_overfit
"""
from __future__ import annotations

import os
import re
import sys
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import mse_loss
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.tactile_model import TactileReWiNDTransformer


# ---------- IsaacLab single-task dataset ----------

DEFAULT_INSTRUCTIONS = [
    "grasp peg and insert to another hole",
    "pick up the peg and insert it into the second hole",
    "use the gripper to grab the peg and place it in the other hole",
    "grasp the cylindrical peg and slot it into the adjacent hole",
    "transfer the peg from one hole to another",
]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode_instructions_minilm(instructions: List[str], device) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(device)
    model.eval()
    with torch.no_grad():
        enc = tok(instructions, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model(**enc)
        emb = mean_pooling(out, enc["attention_mask"]).cpu().numpy().astype(np.float32)
    return emb


class IsaacLabSingleTaskDataset(Dataset):
    """Reads `data_dir/ep*.npy` (each a dict with Task/Tactile/Success).

    Output schema per item:
      video_array : float32  (max_length, 2, 40, 25)
      text_array  : float32  (text_dim,)
      progress    : float32  (max_length,)
      class_label : float32  (max_length,)   # 1 = positive sample, 0 = negative
    """

    def __init__(
        self,
        data_dir: str,
        instructions_emb: np.ndarray,   # (N_inst, 384)
        max_length: int = 16,
        rewind: bool = True,
        rewind_ratio: float = 0.8,
        rewind_peak_min: float = 0.5,
        rewind_peak_max: float = 0.8,
        sample_neg: bool = True,
        neg_ratio: float = 0.0,         # no other tasks → defaults off
        epoch_steps: int = 200,
        batch_size: int = 16,
        shear_channels: tuple = (1, 2),
        success_only: bool = False,
        synthetic_success_threshold: Optional[int] = None,
        balance_success_fail: bool = False,
    ):
        self.data_dir = data_dir
        self.max_length = max_length
        self.rewind = rewind
        self.rewind_ratio = rewind_ratio
        self.sample_neg = sample_neg
        self.neg_ratio = neg_ratio
        self.epoch_steps = epoch_steps
        self.batch_size = batch_size
        self.shear_channels = tuple(shear_channels)
        self.instructions_emb = instructions_emb.astype(np.float32)
        self.synthetic_success_threshold = synthetic_success_threshold
        self.rewind_peak_min = rewind_peak_min
        self.rewind_peak_max = rewind_peak_max
        self.balance_success_fail = balance_success_fail

        # Walk dir; for each npy try dict format first, fall back to raw ndarray
        # plus a synthetic Success rule if a threshold is provided.
        self.entries: List[Dict] = []
        n_succ = n_fail = n_skip = 0
        n_synth = 0
        ep_re = re.compile(r"^ep(\d+)\.npy$")
        for fn in sorted(os.listdir(data_dir)):
            if not fn.endswith(".npy"):
                continue
            path = os.path.join(data_dir, fn)
            try:
                arr = np.load(path, allow_pickle=True)
            except Exception:
                n_skip += 1
                continue

            success: Optional[int] = None
            if arr.dtype == object:
                d = arr.item() if arr.ndim == 0 else None
                if isinstance(d, dict) and "Success" in d and "Tactile" in d:
                    success = int(d["Success"])
            elif arr.ndim == 4:
                # raw ndarray (T, H, W, C) — derive synthetic label from filename
                m = ep_re.match(fn)
                if m and self.synthetic_success_threshold is not None:
                    ep_idx = int(m.group(1))
                    success = 1 if ep_idx >= self.synthetic_success_threshold else 0
                    n_synth += 1

            if success is None:
                n_skip += 1
                continue
            self.entries.append({"file": fn, "success": success})
            if success:
                n_succ += 1
            else:
                n_fail += 1
        msg = (f"[IsaacLabSingleTaskDataset] {data_dir}: "
               f"success={n_succ}, fail={n_fail}, skipped={n_skip}")
        if n_synth:
            msg += f"  (synthetic-labeled raw ndarray: {n_synth}, threshold={self.synthetic_success_threshold})"
        print(msg)
        if success_only:
            self.entries = [e for e in self.entries if e["success"]]
            print(f"  filtered to success_only={len(self.entries)}")
        if not self.entries:
            raise RuntimeError(f"no usable trajectories under {data_dir}")

        # For balanced sampling: pre-bucketed by success.
        self._success_entries = [e for e in self.entries if e["success"]]
        self._fail_entries = [e for e in self.entries if not e["success"]]
        if self.balance_success_fail:
            if not self._success_entries or not self._fail_entries:
                print(f"  WARNING: balance_success_fail requested but only one "
                      f"class is present (succ={len(self._success_entries)}, "
                      f"fail={len(self._fail_entries)}); falling back to uniform.")
                self.balance_success_fail = False
            else:
                print(f"  balanced sampling: even idx -> success "
                      f"({len(self._success_entries)}), "
                      f"odd idx -> fail ({len(self._fail_entries)})")

    def __len__(self) -> int:
        return self.batch_size * self.epoch_steps

    def _sample_text(self) -> torch.Tensor:
        idx = random.randint(0, self.instructions_emb.shape[0] - 1)
        return torch.from_numpy(self.instructions_emb[idx]).float()

    def _load_traj(self, entry: Dict) -> np.ndarray:
        arr = np.load(os.path.join(self.data_dir, entry["file"]), allow_pickle=True)
        if arr.dtype == object:
            d = arr.item()
            traj = np.asarray(d["Tactile"])      # dict format
        else:
            traj = np.asarray(arr)               # raw ndarray fallback
        # drop unwanted channels (default keep (Fx, Fy) shear)
        traj = traj[..., list(self.shear_channels)]   # (T, 40, 25, 2)
        return traj

    def _to_torch_frames(self, traj: np.ndarray, idx_arr: np.ndarray) -> torch.Tensor:
        f = traj[idx_arr].astype(np.float32, copy=True)            # (T, 40, 25, 2)
        x = torch.from_numpy(f).permute(0, 3, 1, 2).contiguous()   # (T, 2, 40, 25)
        return x

    def _resize_indices(self, idx_arr, progress):
        T = idx_arr.shape[0]
        if T < self.max_length:
            pad = self.max_length - T
            idx_arr = np.concatenate([idx_arr, np.full(pad, idx_arr[-1], dtype=idx_arr.dtype)])
            progress = np.concatenate([progress, np.full(pad, progress[-1], dtype=progress.dtype)])
        elif T > self.max_length:
            local = np.linspace(0, T - 1, self.max_length).astype(int)
            idx_arr = idx_arr[local]
            progress = progress[local]
        return idx_arr, progress

    def _sample_forward(self, traj):
        N = len(traj)
        start = random.randint(0, N - 3)
        end = random.randint(start + 3, N)
        full_len = N - start
        idx_arr = np.arange(start, end, dtype=np.int64)
        progress = ((np.arange(end - start) + 1) / full_len).astype(np.float32)
        idx_arr, progress = self._resize_indices(idx_arr, progress)
        return self._to_torch_frames(traj, idx_arr), progress

    def _sample_rewind(self, traj):
        """Forward to a peak frame at proportion peak_ratio of T, then reverse to 0.

        peak_ratio is drawn uniformly from [rewind_peak_min, rewind_peak_max].
        max_length is split half-half between forward and reverse legs so the
        output is exactly max_length frames (no padding).
        """
        N = len(traj)
        if N < 4:
            return self._sample_forward(traj)
        peak_ratio = random.uniform(self.rewind_peak_min, self.rewind_peak_max)
        peak_frame = max(2, min(N - 1, int(round(peak_ratio * (N - 1)))))

        n_fwd = (self.max_length + 1) // 2          # 9 if max_length=16
        n_rev = self.max_length - n_fwd              # 7
        fwd_idx = np.round(np.linspace(0, peak_frame, n_fwd)).astype(np.int64)
        fwd_progress = np.linspace(0, peak_ratio, n_fwd).astype(np.float32)
        if n_rev > 0:
            rev_idx = np.round(np.linspace(peak_frame - 1, 0, n_rev)).astype(np.int64)
            rev_progress = np.linspace(
                peak_ratio * (n_fwd - 1) / max(n_fwd, 1), 0, n_rev
            ).astype(np.float32)
        else:
            rev_idx = np.empty(0, dtype=np.int64)
            rev_progress = np.empty(0, dtype=np.float32)
        idx_arr = np.concatenate([fwd_idx, rev_idx])
        progress = np.concatenate([fwd_progress, rev_progress])
        return self._to_torch_frames(traj, idx_arr), progress

    def __getitem__(self, idx):
        if self.balance_success_fail:
            # Even idx -> success, odd -> fail; gives 50/50 per batch.
            pool = self._success_entries if idx % 2 == 0 else self._fail_entries
        else:
            pool = self.entries
        entry = random.choice(pool)
        traj = self._load_traj(entry)
        if len(traj) < 3:
            return self.__getitem__((idx + 1) % len(self))

        if entry["success"] == 1:
            # Successful traj: forward sample + rewind augmentation
            if self.rewind and random.random() < self.rewind_ratio:
                frames, progress = self._sample_rewind(traj)
            else:
                frames, progress = self._sample_forward(traj)
            label = np.ones_like(progress, dtype=np.float32)
        else:
            # Failure: pick a forward window but progress is forced to 0 throughout
            frames, _ = self._sample_forward(traj)
            progress = np.zeros(self.max_length, dtype=np.float32)
            label = np.zeros(self.max_length, dtype=np.float32)

        text = self._sample_text()
        return {
            "video_array": frames,
            "text_array":  text,
            "progress":    torch.from_numpy(progress),
            "class_label": torch.from_numpy(label),
        }


# ---------- training ----------

class CosineWithMinLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, opt, max_steps, max_lr, min_lr, last_epoch=-1):
        self.max_steps, self.max_lr, self.min_lr = max_steps, max_lr, min_lr
        super().__init__(opt, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.max_steps:
            cos = 0.5 * (1 + math.cos(math.pi * self.last_epoch / self.max_steps))
            return [self.min_lr + (self.max_lr - self.min_lr) * cos for _ in self.base_lrs]
        return [self.min_lr for _ in self.base_lrs]


def train_step(model, batch, optimizer, scheduler, device, clip_grad, log_to_wandb):
    model.train()
    optimizer.zero_grad()

    video = batch["video_array"].to(device, non_blocking=True).float()
    text = batch["text_array"].to(device, non_blocking=True).float()
    progress = batch["progress"].to(device, non_blocking=True).float()

    pred = model(video, text).squeeze(-1)
    loss = mse_loss(pred[:, 1:], progress[:, 1:])

    loss.backward()
    if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    metrics = {"loss": float(loss.item()), "lr": optimizer.param_groups[0]["lr"]}
    if log_to_wandb:
        wandb.log({f"train/{k}": v for k, v in metrics.items()})
    return metrics


def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_to_wandb = bool(args.wandb_entity)
    if log_to_wandb:
        wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                   name=args.run_name, config=vars(args))

    instructions = DEFAULT_INSTRUCTIONS
    if args.instructions:
        instructions = list(args.instructions)
    print(f"instructions ({len(instructions)}):")
    for s in instructions:
        print(f"  - {s!r}")
    instructions_emb = encode_instructions_minilm(instructions, device)
    print(f"encoded {instructions_emb.shape[0]} paraphrases -> dim {instructions_emb.shape[1]}")

    train_ds = IsaacLabSingleTaskDataset(
        data_dir=args.data_dir,
        instructions_emb=instructions_emb,
        max_length=args.max_length,
        rewind=args.rewind,
        rewind_ratio=args.rewind_ratio,
        sample_neg=False,
        neg_ratio=0.0,
        epoch_steps=args.steps_per_epoch,
        batch_size=args.batch_size,
        shear_channels=tuple(args.shear_channels),
        success_only=args.success_only,
        synthetic_success_threshold=args.synthetic_success_threshold,
        rewind_peak_min=args.rewind_peak_min,
        rewind_peak_max=args.rewind_peak_max,
        balance_success_fail=args.balance_success_fail,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # Stash the encoder config on `args` so the saved ckpt is self-describing
    # (eval scripts read these from cfg).
    args.num_strided_layers = 3
    args.bimanual_axis = "height"

    model = TactileReWiNDTransformer(
        max_length=args.max_length, text_dim=384, hidden_dim=args.hidden_dim,
        num_heads=args.num_heads, num_layers=args.num_layers,
        per_hand_dim=args.per_hand_dim,
        num_strided_layers=args.num_strided_layers,   # IsaacLab (40, 25) is small
        bimanual_axis=args.bimanual_axis,             # bimanual concat along H
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f} M")

    if args.init_from:
        state = torch.load(args.init_from, map_location=device, weights_only=False)
        msg = model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"init from {args.init_from}  (missing={len(msg.missing_keys)}, "
              f"unexpected={len(msg.unexpected_keys)})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = CosineWithMinLR(optimizer, max_steps=total_steps,
                                max_lr=args.lr, min_lr=args.min_lr)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            m = train_step(model, batch, optimizer, scheduler, device,
                           args.clip_grad, log_to_wandb)
            global_step += 1
            pbar.set_postfix(loss=f"{m['loss']:.4f}", lr=f"{m['lr']:.2e}")

        ckpt_path = os.path.join(args.ckpt_dir, f"isaaclab_overfit_epoch{epoch}.pth")
        torch.save({
            "args": vars(args),
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }, ckpt_path)
        print(f"saved {ckpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Directory with ep*.npy (dict format from forge_env).")
    ap.add_argument("--ckpt_dir",
                    default="/mnt/tank/uber/Tactile-Reward/checkpoints_isaaclab_overfit")
    ap.add_argument("--instructions", nargs="*", default=None,
                    help="Override the default 5 paraphrases.")
    ap.add_argument("--init_from", default=None,
                    help="Optional: warm-start from an existing aligned ckpt "
                         "(load with strict=False).")
    ap.add_argument("--shear_channels", type=int, nargs=2, default=[1, 2],
                    help="Channels of (Fx, Fy) inside the (40, 25, 3) tensor.")
    ap.add_argument("--success_only", action="store_true",
                    help="Discard failure trajectories (debug use).")
    ap.add_argument("--synthetic_success_threshold", type=int, default=None,
                    help="For raw-ndarray ep*.npy files (no Success in dict): "
                         "ep_idx >= threshold -> Success=1, else 0. Use this when "
                         "you don't have labelled data yet but want to fake them "
                         "from the assumption that later RL episodes succeed more.")

    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_project", default="tactile-rewind")
    ap.add_argument("--run_name", default="isaaclab_overfit")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--steps_per_epoch", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--clip_grad", action="store_true")
    ap.add_argument("--rewind", action="store_true")
    ap.add_argument("--rewind_ratio", type=float, default=0.8)
    ap.add_argument("--rewind_peak_min", type=float, default=0.5,
                    help="Min peak progress for rewind sampling (uniform).")
    ap.add_argument("--rewind_peak_max", type=float, default=0.8,
                    help="Max peak progress for rewind sampling.")
    ap.add_argument("--balance_success_fail", action="store_true",
                    help="50/50 success/fail per batch (even/odd idx). Strongly "
                         "recommended when success ratio is < 30%%.")

    ap.add_argument("--max_length", type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--per_hand_dim", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
