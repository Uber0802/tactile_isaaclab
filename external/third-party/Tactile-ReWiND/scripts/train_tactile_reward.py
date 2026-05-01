"""Train a tactile ReWiND reward model on AnyTouch2 (single-source, no OpenX).

Mirrors the original `train_reward.py` loss recipe but:
  * removes the OpenX co-training branch (we only have AnyTouch2),
  * trains the CNN encoder + transformer end-to-end (no precomputed embeddings),
  * keeps rewind augmentation + dataset-level negative sampling +
    in-batch roll-based negative pairing.
"""
from __future__ import annotations

import os
import sys
import math
import random
import argparse

import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from torch.optim import Optimizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.tactile_model import TactileReWiNDTransformer
from tools.tactile_dataset import TactileReWiNDDataset


class CosineWithMinLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, max_steps: int, max_lr: float, min_lr: float,
                 last_epoch: int = -1):
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.max_steps:
            cos = 0.5 * (1 + math.cos(math.pi * self.last_epoch / self.max_steps))
            return [self.min_lr + (self.max_lr - self.min_lr) * cos for _ in self.base_lrs]
        return [self.min_lr for _ in self.base_lrs]


def train_step(model, batch, optimizer, scheduler, device, clip_grad: bool, log_to_wandb: bool):
    model.train()
    optimizer.zero_grad()

    video = batch["video_array"].to(device, non_blocking=True).float()    # (B, T, 2, H, W)
    text = batch["text_array"].to(device, non_blocking=True).float()      # (B, text_dim)
    progress = batch["progress"].to(device, non_blocking=True).float()    # (B, T)
    label = batch["class_label"].to(device, non_blocking=True).float()    # (B, T)

    # Roll-based in-batch negatives: keep videos, shift text by one position.
    neg_text = torch.roll(text, shifts=1, dims=0)
    neg_progress = torch.zeros_like(progress)
    neg_label = torch.zeros_like(label)

    all_video = torch.cat([video, video], dim=0)
    all_text = torch.cat([text, neg_text], dim=0)
    all_progress = torch.cat([progress, neg_progress], dim=0)
    all_label = torch.cat([label, neg_label], dim=0)

    pred = model(all_video, all_text).squeeze(-1)        # (2B, T)

    # Skip the first frame's prediction (no useful causal history).
    pred_t = pred[:, 1:]
    target_t = all_progress[:, 1:]

    # Per-sample mask: positive sample if any frame's class_label is 1.
    is_pos = all_label[:, 0].bool()                      # (2B,)
    pos_loss = mse_loss(pred_t[is_pos], target_t[is_pos]) if is_pos.any() else pred.sum() * 0
    neg_loss = mse_loss(pred_t[~is_pos], target_t[~is_pos]) if (~is_pos).any() else pred.sum() * 0

    n_pos = int(is_pos.sum().item())
    n_neg = int((~is_pos).sum().item())
    total = max(n_pos + n_neg, 1)
    loss = pos_loss * n_pos / total + neg_loss * n_neg / total

    loss.backward()
    if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    metrics = {
        "loss": float(loss.item()),
        "pos_loss": float(pos_loss.item()) if n_pos > 0 else 0.0,
        "neg_loss": float(neg_loss.item()) if n_neg > 0 else 0.0,
        "lr": optimizer.param_groups[0]["lr"],
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
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
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    train_ds = TactileReWiNDDataset(
        metadata_h5_path=args.train_metadata,
        max_length=args.max_length,
        rewind=args.rewind,
        rewind_ratio=args.rewind_ratio,
        sample_neg=True,
        neg_ratio=args.neg_ratio,
        epoch_steps=args.steps_per_epoch,
        batch_size=args.batch_size,
        data_dir_override=args.data_dir_override,
        align_to_isaaclab=args.isaaclab_aligned,
        multiscale_align=args.multiscale_align,
    )
    print(f"train: {len(train_ds.tasks)} tasks, {len(train_ds)} samples per epoch")
    if args.isaaclab_aligned:
        print("alignment: AnyTouch2 (320, 480, 2) → IsaacLab (40, 25, 2) "
              "[long↔long, bimanual concat along H]")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    if args.isaaclab_aligned:
        num_strided_layers = args.num_strided_layers if args.num_strided_layers > 0 else 3
        bimanual_axis = "height"
    else:
        num_strided_layers = args.num_strided_layers if args.num_strided_layers > 0 else 5
        bimanual_axis = "width"

    model = TactileReWiNDTransformer(
        max_length=args.max_length,
        text_dim=384,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        per_hand_dim=args.per_hand_dim,
        num_strided_layers=num_strided_layers,
        bimanual_axis=bimanual_axis,
    ).to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {n_params / 1e6:.2f} M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = CosineWithMinLR(optimizer, max_steps=total_steps, max_lr=args.lr, min_lr=args.min_lr)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            metrics = train_step(model, batch, optimizer, scheduler, device,
                                 args.clip_grad, log_to_wandb)
            global_step += 1
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}",
                             pos=f"{metrics['pos_loss']:.4f}",
                             neg=f"{metrics['neg_loss']:.4f}")

        ckpt_path = os.path.join(args.ckpt_dir, f"tactile_rewind_epoch{epoch}.pth")
        torch.save({
            "args": vars(args),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }, ckpt_path)
        print(f"saved {ckpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train_metadata",
                    default="/mnt/tank/uber/Tactile-Reward/tactile_metadata_train.h5")
    ap.add_argument("--data_dir_override", default=None,
                    help="Override the data_dir attribute stored in the metadata H5.")

    # Logging
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_project", default="tactile-rewind")
    ap.add_argument("--run_name", default="tactile_rewind_anytouch2")
    ap.add_argument("--ckpt_dir", default="/mnt/tank/uber/Tactile-Reward/checkpoints")

    # Optim / schedule
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--clip_grad", action="store_true")

    # Augmentation
    ap.add_argument("--rewind", action="store_true")
    ap.add_argument("--rewind_ratio", type=float, default=0.8)
    ap.add_argument("--neg_ratio", type=float, default=0.2)

    # Model
    ap.add_argument("--max_length", type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--per_hand_dim", type=int, default=384)
    ap.add_argument("--num_strided_layers", type=int, default=0,
                    help="0 = auto (5 if !isaaclab_aligned else 3).")

    # Cross-dataset alignment
    ap.add_argument("--isaaclab_aligned", action="store_true",
                    help="Pool AnyTouch2 input to (40, 25) bimanual-H so it matches "
                         "IsaacLab Forge tactile (long↔long).")
    ap.add_argument("--multiscale_align", action="store_true",
                    help="When --isaaclab_aligned: each batch picks a random "
                         "(target_h, target_w) from a preset 17-resolution grid "
                         "(4:5 / 3:4 ratios + IsaacLab-like 20x35), so the encoder "
                         "sees scale-varied inputs during training. Eval still uses (20,25).")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
