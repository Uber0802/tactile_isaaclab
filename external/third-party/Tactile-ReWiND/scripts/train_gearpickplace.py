"""Train a tactile ReWiND reward model on the Forge GearMesh PickPlace task.

Combines multiple `ep*.npy` data dirs, holds out N success + N fail trajectories
for evaluation, and trains on the remainder. Each ep file is a dict produced by
forge_env._flush_tactile_episode:
    {"Task": str, "Tactile": (T, 40, 25, 3) float16, "Success": int 0/1}

Note: the Task field on disk is hardcoded as the peg string for ALL forge dumps
(peg/gear/nut). We override via DEFAULT_GEAR_INSTRUCTIONS.

Heads-up: gear data is heavily success-imbalanced (~4% success). Combine both
gear dirs and keep --balance_success_fail to avoid drowning the success signal.

Usage:
    python scripts/train_gearpickplace.py \\
        --data_dirs /mnt/tank/tactile/tactile_dataset/gearpickplace \\
                    /mnt/tank/tactile/tactile_dataset/gearpickplace_baselineA \\
        --ckpt_dir /mnt/tank/tactile/Tactile-Reward/checkpoints_gearpickplace_balanced \\
        --epochs 100 --balance_success_fail --rewind --clip_grad
"""
from __future__ import annotations

import os
import sys
import math
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


DEFAULT_GEAR_INSTRUCTIONS = [
    "pick up the gear and mesh it onto the shaft",
    "grasp the gear and slide it down onto the gear shaft",
    "use the gripper to pick the gear and engage it on the shaft",
    "grab the gear and place it onto the meshing shaft",
    "lift the gear and mesh it onto the rotating shaft",
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


def scan_entries(data_dirs: List[str]) -> List[Dict]:
    entries: List[Dict] = []
    for d in data_dirs:
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".npy"):
                continue
            path = os.path.join(d, fn)
            try:
                arr = np.load(path, allow_pickle=True)
            except Exception:
                continue
            if arr.dtype != object:
                continue
            dct = arr.item() if arr.ndim == 0 else None
            if not (isinstance(dct, dict) and "Success" in dct and "Tactile" in dct):
                continue
            entries.append({"dir": d, "file": fn, "success": int(dct["Success"])})
    return entries


def pick_holdout(entries: List[Dict], n_succ: int, n_fail: int, seed: int
                 ) -> Tuple[List[Dict], List[Dict]]:
    """Deterministic split: shuffle once with `seed`, peel off N_succ + N_fail.

    Returns (train_entries, eval_entries).
    """
    rng = random.Random(seed)
    succ = [e for e in entries if e["success"]]
    fail = [e for e in entries if not e["success"]]
    rng.shuffle(succ)
    rng.shuffle(fail)
    if len(succ) < n_succ:
        raise RuntimeError(f"not enough success episodes: have {len(succ)}, need {n_succ}")
    if len(fail) < n_fail:
        raise RuntimeError(f"not enough fail episodes: have {len(fail)}, need {n_fail}")
    eval_set = succ[:n_succ] + fail[:n_fail]
    eval_keys = {(e["dir"], e["file"]) for e in eval_set}
    train_set = [e for e in entries if (e["dir"], e["file"]) not in eval_keys]
    return train_set, eval_set


class GearPickPlaceDataset(Dataset):
    """Per-item schema:
        video_array : float32  (max_length, 2, 40, 25)
        text_array  : float32  (text_dim,)
        progress    : float32  (max_length,)
        class_label : float32  (max_length,)
    """

    def __init__(
        self,
        entries: List[Dict],
        instructions_emb: np.ndarray,
        max_length: int = 16,
        rewind: bool = True,
        rewind_ratio: float = 0.8,
        rewind_peak_min: float = 0.5,
        rewind_peak_max: float = 0.8,
        epoch_steps: int = 200,
        batch_size: int = 16,
        shear_channels: tuple = (1, 2),
        balance_success_fail: bool = False,
        forward_full_traj_ratio: float = 0.0,
    ):
        self.entries = entries
        self.max_length = max_length
        self.rewind = rewind
        self.rewind_ratio = rewind_ratio
        self.rewind_peak_min = rewind_peak_min
        self.rewind_peak_max = rewind_peak_max
        self.epoch_steps = epoch_steps
        self.batch_size = batch_size
        self.shear_channels = tuple(shear_channels)
        self.instructions_emb = instructions_emb.astype(np.float32)
        self.balance_success_fail = balance_success_fail
        self.forward_full_traj_ratio = forward_full_traj_ratio

        self._success = [e for e in entries if e["success"]]
        self._fail = [e for e in entries if not e["success"]]
        print(f"[GearPickPlaceDataset] train entries={len(entries)} "
              f"(success={len(self._success)}, fail={len(self._fail)})")
        if self.balance_success_fail:
            if not self._success or not self._fail:
                print("  WARN: balance_success_fail requested but one class missing; uniform.")
                self.balance_success_fail = False
            else:
                print(f"  balanced sampling: even idx -> success, odd idx -> fail")

    def __len__(self) -> int:
        return self.batch_size * self.epoch_steps

    def _sample_text(self) -> torch.Tensor:
        idx = random.randint(0, self.instructions_emb.shape[0] - 1)
        return torch.from_numpy(self.instructions_emb[idx]).float()

    def _load_traj(self, entry: Dict) -> np.ndarray:
        arr = np.load(os.path.join(entry["dir"], entry["file"]), allow_pickle=True)
        d = arr.item()
        traj = np.asarray(d["Tactile"])               # (T, 40, 25, 3)
        traj = traj[..., list(self.shear_channels)]   # (T, 40, 25, 2)
        return traj

    def _to_torch_frames(self, traj: np.ndarray, idx_arr: np.ndarray) -> torch.Tensor:
        f = traj[idx_arr].astype(np.float32, copy=True)
        return torch.from_numpy(f).permute(0, 3, 1, 2).contiguous()   # (T, 2, 40, 25)

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
        # With prob forward_full_traj_ratio, draw the *full* trajectory (start=0,
        # end=N) so progress runs 1/N -> 1.0. Matches the inference distribution
        # (eval always feeds start=0, end=T) and anchors the "progress reaches 1
        # at end" signal that random sub-window sampling rarely provides.
        if random.random() < self.forward_full_traj_ratio:
            start = 0
            end = N
        else:
            start = random.randint(0, N - 3)
            end = random.randint(start + 3, N)
        full_len = N - start
        idx_arr = np.arange(start, end, dtype=np.int64)
        progress = ((np.arange(end - start) + 1) / full_len).astype(np.float32)
        idx_arr, progress = self._resize_indices(idx_arr, progress)
        return self._to_torch_frames(traj, idx_arr), progress

    def _sample_rewind(self, traj):
        N = len(traj)
        if N < 4:
            return self._sample_forward(traj)
        peak_ratio = random.uniform(self.rewind_peak_min, self.rewind_peak_max)
        peak_frame = max(2, min(N - 1, int(round(peak_ratio * (N - 1)))))
        n_fwd = (self.max_length + 1) // 2
        n_rev = self.max_length - n_fwd
        fwd_idx = np.round(np.linspace(0, peak_frame, n_fwd)).astype(np.int64)
        fwd_progress = np.linspace(0, peak_ratio, n_fwd).astype(np.float32)
        if n_rev > 0:
            rev_idx = np.round(np.linspace(peak_frame - 1, 0, n_rev)).astype(np.int64)
            rev_progress = np.linspace(peak_ratio * (n_fwd - 1) / max(n_fwd, 1), 0, n_rev
                                       ).astype(np.float32)
        else:
            rev_idx = np.empty(0, dtype=np.int64)
            rev_progress = np.empty(0, dtype=np.float32)
        idx_arr = np.concatenate([fwd_idx, rev_idx])
        progress = np.concatenate([fwd_progress, rev_progress])
        return self._to_torch_frames(traj, idx_arr), progress

    def __getitem__(self, idx):
        if self.balance_success_fail:
            pool = self._success if idx % 2 == 0 else self._fail
        else:
            pool = self.entries
        entry = random.choice(pool)
        traj = self._load_traj(entry)
        if len(traj) < 3:
            return self.__getitem__((idx + 1) % len(self))

        if entry["success"] == 1:
            if self.rewind and random.random() < self.rewind_ratio:
                frames, progress = self._sample_rewind(traj)
            else:
                frames, progress = self._sample_forward(traj)
            label = np.ones_like(progress, dtype=np.float32)
        else:
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

    # ---- scan + holdout ----
    all_entries = scan_entries(args.data_dirs)
    if not all_entries:
        raise RuntimeError(f"no usable trajectories under {args.data_dirs}")
    n_succ_total = sum(1 for e in all_entries if e["success"])
    n_fail_total = len(all_entries) - n_succ_total
    print(f"total entries: {len(all_entries)} "
          f"(success={n_succ_total}, fail={n_fail_total})")

    train_entries, eval_entries = pick_holdout(
        all_entries, n_succ=args.n_eval_succ, n_fail=args.n_eval_fail, seed=args.seed
    )
    print(f"holdout: {len(eval_entries)} (success={args.n_eval_succ}, fail={args.n_eval_fail})")
    print(f"train:   {len(train_entries)} "
          f"(success={sum(1 for e in train_entries if e['success'])}, "
          f"fail={sum(1 for e in train_entries if not e['success'])})")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    holdout_path = os.path.join(args.ckpt_dir, "eval_holdout.json")
    with open(holdout_path, "w") as fh:
        json.dump({
            "seed": args.seed,
            "n_eval_succ": args.n_eval_succ,
            "n_eval_fail": args.n_eval_fail,
            "data_dirs": args.data_dirs,
            "eval_entries": eval_entries,
            "train_entry_count": len(train_entries),
        }, fh, indent=2)
    print(f"wrote holdout split -> {holdout_path}")

    # ---- instructions ----
    instructions = DEFAULT_GEAR_INSTRUCTIONS
    if args.instructions:
        instructions = list(args.instructions)
    print(f"instructions ({len(instructions)}):")
    for s in instructions:
        print(f"  - {s!r}")
    instructions_emb = encode_instructions_minilm(instructions, device)
    print(f"encoded {instructions_emb.shape[0]} paraphrases -> dim {instructions_emb.shape[1]}")

    # ---- dataset / loader ----
    train_ds = GearPickPlaceDataset(
        entries=train_entries,
        instructions_emb=instructions_emb,
        max_length=args.max_length,
        rewind=args.rewind,
        rewind_ratio=args.rewind_ratio,
        rewind_peak_min=args.rewind_peak_min,
        rewind_peak_max=args.rewind_peak_max,
        epoch_steps=args.steps_per_epoch,
        batch_size=args.batch_size,
        shear_channels=tuple(args.shear_channels),
        balance_success_fail=args.balance_success_fail,
        forward_full_traj_ratio=args.forward_full_traj_ratio,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    args.num_strided_layers = 3
    args.bimanual_axis = "height"

    model = TactileReWiNDTransformer(
        max_length=args.max_length, text_dim=384, hidden_dim=args.hidden_dim,
        num_heads=args.num_heads, num_layers=args.num_layers,
        per_hand_dim=args.per_hand_dim,
        num_strided_layers=args.num_strided_layers,
        bimanual_axis=args.bimanual_axis,
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

    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            m = train_step(model, batch, optimizer, scheduler, device,
                           args.clip_grad, log_to_wandb)
            global_step += 1
            pbar.set_postfix(loss=f"{m['loss']:.4f}", lr=f"{m['lr']:.2e}")

        ckpt_path = os.path.join(args.ckpt_dir, f"gearpickplace_epoch{epoch}.pth")
        torch.save({
            "args": vars(args),
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }, ckpt_path)
        print(f"saved {ckpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True,
                    help="One or more directories containing ep*.npy dicts "
                         "(Task/Tactile/Success).")
    ap.add_argument("--ckpt_dir",
                    default="/mnt/tank/tactile/Tactile-Reward/checkpoints_gearpickplace_balanced")
    ap.add_argument("--instructions", nargs="*", default=None,
                    help="Override the default 5 gear-pickplace paraphrases.")
    ap.add_argument("--init_from", default=None,
                    help="Optional warm-start ckpt path.")
    ap.add_argument("--shear_channels", type=int, nargs=2, default=[1, 2],
                    help="Channels of (Fx, Fy) inside the (40, 25, 3) tensor.")

    ap.add_argument("--n_eval_succ", type=int, default=10,
                    help="Number of success episodes held out for eval.")
    ap.add_argument("--n_eval_fail", type=int, default=10,
                    help="Number of fail episodes held out for eval.")

    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_project", default="tactile-rewind")
    ap.add_argument("--run_name", default="gearpickplace_balanced")

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--steps_per_epoch", type=int, default=400)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--clip_grad", action="store_true")
    ap.add_argument("--rewind", action="store_true")
    ap.add_argument("--rewind_ratio", type=float, default=0.2)
    ap.add_argument("--forward_full_traj_ratio", type=float, default=0.5,
                    help="Fraction of forward (non-rewind) success samples that "
                         "use the full trajectory (start=0, end=N) instead of a "
                         "random sub-window. Matches inference distribution and "
                         "anchors progress->1.0 supervision. Set to 0 for the "
                         "original ReWiND behavior.")
    ap.add_argument("--rewind_peak_min", type=float, default=0.5)
    ap.add_argument("--rewind_peak_max", type=float, default=0.8)
    ap.add_argument("--balance_success_fail", action="store_true",
                    help="50/50 success/fail per batch (even/odd idx).")

    ap.add_argument("--max_length", type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--per_hand_dim", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
