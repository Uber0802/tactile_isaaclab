"""Train a ReWiND visual reward model on cached DINOv2 features.

Visual twin of `train_taRL.py`. Same training signal and sampling; the only
differences are the input modality and the model:

    train_taRL.py        (T,40,25,3) tactile -> TactileReWiNDTransformer
    train_visual_taRL.py (T,768) DINOv2 CLS  -> ReWiNDTransformer  (ReWiND/model.py)

Run `precompute_dino_features.py` first — this script reads the .npz cache, not
the raw `*_camera.npy`, because the DINOv2 backbone is frozen at both train and
RL time so re-encoding every epoch is pure waste.

Everything that touches inference compatibility is pinned to what
`ForgeEnv._init_visual_reward` / `_compute_visual_reward` (forge_env.py:399-620)
expect, so the produced .pth is a drop-in FORGE_VISUAL_REWARD_CKPT:
  * ReWiNDTransformer(args=<Namespace with max_length>, video_dim=768,
    text_dim=384, hidden_dim=512)
  * MiniLM mean-pooled text embedding, L2-NORMALIZED (forge_env.py:499). The
    tactile trainer does NOT normalize — do not copy that half of the recipe.
  * checkpoint dict {"args": argparse.Namespace, "model_state_dict",
    "optimizer_state_dict", "epoch"}. `args` MUST stay a Namespace: forge_env
    reads max_length with getattr(), which silently falls back to 16 on a dict.

Usage:
    python scripts/train_visual_taRL.py \\
        --feat_dirs /mnt/scratch/tactile/dino_cache/gear_seed2_multipos \\
        --ckpt_dir  /mnt/lab-tank/tactile/Tactile-Reward/ckpt_visual/gear_seed2_multipos \\
        --run_name  gear_rgb_seed2_multipos \\
        --task_texts "pick up the gear and mesh it onto the shaft" ...
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import random
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

DEFAULT_REWIND_ROOT = "/mnt/home/tactile/tactile_isaaclab/external/third-party/ReWiND"


class CosineWithMinLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, max_steps: int, max_lr: float,
                 min_lr: float, last_epoch: int = -1):
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.max_steps:
            cos = 0.5 * (1 + math.cos(math.pi * self.last_epoch / self.max_steps))
            return [self.min_lr + (self.max_lr - self.min_lr) * cos for _ in self.base_lrs]
        return [self.min_lr for _ in self.base_lrs]


def load_rewind_class(rewind_root: str):
    """Import ReWiNDTransformer by explicit path. Tactile-ReWiND ships its own
    top-level `model.py`, so a plain `from model import ...` would resolve to
    whichever landed in sys.modules first — see the same guard in
    forge_env.py:433-456."""
    model_path = os.path.join(rewind_root, "model.py")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ReWiND model.py not found at {model_path}")
    spec = importlib.util.spec_from_file_location("rewind_visual_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    if rewind_root not in sys.path:
        sys.path.insert(0, rewind_root)
    spec.loader.exec_module(mod)
    return mod.ReWiNDTransformer


def embed_texts(texts: List[str], device: torch.device) -> np.ndarray:
    """MiniLM mean-pool + L2 normalize, matching forge_env.py:487-501."""
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    minilm = AutoModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L12-v2").to(device).eval()
    with torch.no_grad():
        enc = tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        out = minilm(**enc)
        tok_emb = out[0]
        mask = enc["attention_mask"].unsqueeze(-1).expand(tok_emb.size()).float()
        emb = (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = F.normalize(emb, p=2, dim=1)
    del minilm, tok
    return emb.float().cpu().numpy()


def stratified_split(eps: List[dict], test_ratio: float, seed: int):
    succ = [i for i, e in enumerate(eps) if e["success"] == 1]
    fail = [i for i, e in enumerate(eps) if e["success"] == 0]
    rng = random.Random(seed)
    rng.shuffle(succ)
    rng.shuffle(fail)
    n_succ_test = max(1, int(round(len(succ) * test_ratio)))
    n_fail_test = max(1, int(round(len(fail) * test_ratio)))
    test = set(succ[:n_succ_test] + fail[:n_fail_test])
    train_idx = [i for i in range(len(eps)) if i not in test]
    return train_idx, sorted(test)


class VisualDataset(Dataset):
    """Same sampler as train_taRL.py's Data3Dataset, over (T,768) features."""

    def __init__(self, eps: List[dict], indices: List[int], text_embs: np.ndarray,
                 max_length: int = 16, batch_size: int = 128, steps_per_epoch: int = 100,
                 rewind_ratio: float = 0.8, success_prob: float = 0.5, seed: int = 0):
        self.eps = eps
        if text_embs.ndim == 1:
            text_embs = text_embs[None, :]
        self.text_embs = text_embs.astype(np.float32)
        self.max_length = max_length
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.rewind_ratio = rewind_ratio
        self.success_prob = success_prob
        self._succ_idx = [i for i in indices if eps[i]["success"] == 1]
        self._fail_idx = [i for i in indices if eps[i]["success"] == 0]
        if not self._succ_idx or not self._fail_idx:
            raise ValueError(f"need at least one success and one fail episode in train "
                             f"(got {len(self._succ_idx)}/{len(self._fail_idx)})")
        self._rng = random.Random(seed)

    def __len__(self):
        return self.batch_size * self.steps_per_epoch

    def _sample_forward(self, N: int):
        """Random K in [1, N], linspace max_length over 0..K-1 (pad with frame
        K-1 when K < max_length), label progress (idx+1)/N. Mirrors the sliding
        window `_compute_visual_reward` builds at step K."""
        K = self._rng.randint(1, N) if N >= 1 else 1
        if K >= self.max_length:
            idx = np.round(np.linspace(0, K - 1, self.max_length)).astype(np.int64)
        else:
            idx = np.concatenate([np.arange(K, dtype=np.int64),
                                  np.full(self.max_length - K, K - 1, dtype=np.int64)])
        prog = ((idx + 1) / N).astype(np.float32)
        return idx, prog

    def _sample_rewind(self, N: int):
        """Forward to random K then reverse back to 0; progress rises then falls."""
        if N < 2:
            return self._sample_forward(N)
        K = self._rng.randint(2, N)
        fwd_idx = np.arange(K, dtype=np.int64)
        rev_idx = np.arange(K - 2, -1, -1, dtype=np.int64)
        fwd_prog = ((np.arange(K) + 1) / N).astype(np.float32)
        rev_prog = fwd_prog[::-1][1:K].copy()
        full_idx = np.concatenate([fwd_idx, rev_idx])
        full_prog = np.concatenate([fwd_prog, rev_prog]).astype(np.float32)
        T = full_idx.shape[0]
        if T >= self.max_length:
            local = np.round(np.linspace(0, T - 1, self.max_length)).astype(int)
            return full_idx[local], full_prog[local]
        pad = self.max_length - T
        idx = np.concatenate([full_idx, np.full(pad, full_idx[-1], dtype=full_idx.dtype)])
        prog = np.concatenate([full_prog, np.full(pad, full_prog[-1], dtype=full_prog.dtype)])
        return idx, prog

    def __getitem__(self, _):
        is_success = self._rng.random() < self.success_prob
        pool = self._succ_idx if is_success else self._fail_idx
        ep = self.eps[self._rng.choice(pool)]
        N = ep["feat"].shape[0]
        if is_success and self._rng.random() < self.rewind_ratio:
            idx, prog = self._sample_rewind(N)
        else:
            idx, prog = self._sample_forward(N)
        if not is_success:
            prog = np.zeros_like(prog)              # failure -> ground-truth 0
        frames = ep["feat"][idx].astype(np.float32, copy=True)     # (L, 768)
        t_idx = self._rng.randrange(self.text_embs.shape[0])
        return {
            "video_array": torch.from_numpy(frames),
            "text_array": torch.from_numpy(self.text_embs[t_idx]),
            "progress": torch.from_numpy(prog),
            "class_label": torch.full((self.max_length,), 1.0 if is_success else 0.0),
        }


def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = random.Random((info.seed + worker_id) % (2 ** 31))


def train_step(model, batch, optimizer, scheduler, device, clip_grad, amp_dtype):
    model.train()
    optimizer.zero_grad()
    video = batch["video_array"].to(device, non_blocking=True).float()
    text = batch["text_array"].to(device, non_blocking=True).float()
    progress = batch["progress"].to(device, non_blocking=True).float()
    label = batch["class_label"].to(device, non_blocking=True).float()

    use_amp = amp_dtype is not None and device.type == "cuda"
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        pred = model(video, text).squeeze(-1)            # (B, L)
        pred_t = pred[:, 1:]
        target_t = progress[:, 1:]
        is_pos = label[:, 0].bool()
        pos_loss = mse_loss(pred_t[is_pos], target_t[is_pos]) if is_pos.any() else pred.sum() * 0
        neg_loss = mse_loss(pred_t[~is_pos], target_t[~is_pos]) if (~is_pos).any() else pred.sum() * 0
        n_pos, n_neg = int(is_pos.sum()), int((~is_pos).sum())
        total = max(n_pos + n_neg, 1)
        loss = pos_loss * n_pos / total + neg_loss * n_neg / total
    loss.backward()
    if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {
        "loss": float(loss.item()),
        "pos_loss": float(pos_loss.item()) if n_pos > 0 else 0.0,
        "neg_loss": float(neg_loss.item()) if n_neg > 0 else 0.0,
        "lr": optimizer.param_groups[0]["lr"],
    }


def evaluate_test(model, eps, test_idx, text_embs, max_length, device, amp_dtype):
    """Held-out final-frame reward gap: success mean - fail mean. This is the
    ckpt-selection criterion — NOT train loss."""
    model.eval()
    text = torch.from_numpy(text_embs[0]).float().unsqueeze(0).to(device)
    succ_finals, fail_finals = [], []
    use_amp = amp_dtype is not None and device.type == "cuda"
    for i in test_idx:
        ep = eps[i]
        N = ep["feat"].shape[0]
        if N >= max_length:
            idx = np.round(np.linspace(0, N - 1, max_length)).astype(np.int64)
        else:
            idx = np.concatenate([np.arange(N),
                                  np.full(max_length - N, N - 1, dtype=np.int64)])
        x = torch.from_numpy(ep["feat"][idx].astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype,
                                                 enabled=use_amp):
            pred = model(x, text).squeeze(-1).squeeze(0).float().cpu().numpy()
        (succ_finals if ep["success"] else fail_finals).append(float(pred[-1]))
    s_mean = float(np.mean(succ_finals)) if succ_finals else float("nan")
    f_mean = float(np.mean(fail_finals)) if fail_finals else float("nan")
    return {"test_n_succ": len(succ_finals), "test_n_fail": len(fail_finals),
            "test_succ_final_mean": s_mean, "test_fail_final_mean": f_mean,
            "test_gap": s_mean - f_mean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dirs", nargs="+", required=True,
                    help="DINO cache dirs from precompute_dino_features.py "
                         "(globbed recursively for *.npz).")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--run_name", default="rgb_rewind")
    ap.add_argument("--task_texts", nargs="+",
                    default=["grasp peg and insert to another hole"],
                    help="Instruction paraphrases; one drawn per sample, "
                         "the FIRST is used for eval and should be the string "
                         "you pass as FORGE_VISUAL_REWARD_INSTRUCTION.")
    ap.add_argument("--rewind_root", default=DEFAULT_REWIND_ROOT)
    ap.add_argument("--init_from", default=None,
                    help="Optional .pth to warm-start from (e.g. peg_rgb_multipos.pth).")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--steps_per_epoch", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--min_lr", type=float, default=1e-7)
    ap.add_argument("--clip_grad", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--max_length", type=int, default=16)
    ap.add_argument("--rewind_ratio", type=float, default=0.8)
    ap.add_argument("--success_prob", type=float, default=0.5)
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--test_eval_every", type=int, default=1)

    ap.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(os.path.join(args.ckpt_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- load cached features ------------------------------------------------
    files = []
    for d in args.feat_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"feat_dir does not exist: {d} "
                                    f"(run precompute_dino_features.py first)")
        files.extend(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True))
    files = sorted(files)
    print(f"[visual] scanning {len(files)} cached episodes across {len(args.feat_dirs)} dir(s)")
    if args.max_episodes and len(files) > args.max_episodes:
        rng = random.Random(args.seed)
        files = sorted(rng.sample(files, args.max_episodes))
        print(f"[visual] capped to {args.max_episodes} (seed={args.seed})")

    eps = []
    for p in files:
        z = np.load(p, allow_pickle=True)
        feat = z["feat"]
        if feat.ndim != 2 or feat.shape[1] != 768:
            print(f"  skip {p}: unexpected feat shape {feat.shape}", file=sys.stderr)
            continue
        eps.append({"file": p, "feat": feat, "success": int(z["success"])})
    n_succ = sum(e["success"] for e in eps)
    print(f"[visual] loaded {len(eps)} episodes: {n_succ} success, {len(eps) - n_succ} fail")

    train_idx, test_idx = stratified_split(eps, args.test_ratio, args.seed)
    test_succ = sum(eps[i]["success"] for i in test_idx)
    print(f"[visual] split: {len(train_idx)} train, {len(test_idx)} test "
          f"(test {test_succ} success / {len(test_idx) - test_succ} fail)")
    with open(os.path.join(args.ckpt_dir, "test_files.json"), "w") as f:
        json.dump({"test_files": [eps[i]["file"] for i in test_idx],
                   "n_success": test_succ, "n_fail": len(test_idx) - test_succ,
                   "seed": args.seed}, f, indent=2)

    # --- text embeddings -----------------------------------------------------
    text_embs = embed_texts(list(args.task_texts), device)
    print(f"[visual] {len(args.task_texts)} paraphrases -> emb shape={text_embs.shape}")
    for t in args.task_texts:
        print(f"  - {t!r}")

    # --- model ---------------------------------------------------------------
    ReWiNDTransformer = load_rewind_class(args.rewind_root)
    # The checkpoint's `args` doubles as the model config at load time, so build
    # the model from the same Namespace we will later save.
    model_args = argparse.Namespace(**vars(args))
    model = ReWiNDTransformer(args=model_args, video_dim=768, text_dim=384,
                              hidden_dim=512).to(device)
    if args.init_from:
        state = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"[visual] warm-started from {args.init_from}")

    amp_dtype = None
    if args.amp != "off" and device.type == "cuda":
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        print(f"[visual] amp dtype={amp_dtype}")

    train_ds = VisualDataset(
        eps=eps, indices=train_idx, text_embs=text_embs,
        max_length=args.max_length, batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch, rewind_ratio=args.rewind_ratio,
        success_prob=args.success_prob, seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineWithMinLR(optimizer, max_steps=args.epochs * args.steps_per_epoch,
                                max_lr=args.lr, min_lr=args.min_lr)

    history = []
    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        last = None
        for batch in pbar:
            last = train_step(model, batch, optimizer, scheduler, device,
                              args.clip_grad, amp_dtype)
            global_step += 1
            pbar.set_postfix(loss=f"{last['loss']:.4f}", pos=f"{last['pos_loss']:.4f}",
                             neg=f"{last['neg_loss']:.4f}")
        rec = {"epoch": epoch, "step": global_step, "train_loss": last["loss"]}
        if args.test_eval_every and (epoch + 1) % args.test_eval_every == 0:
            test = evaluate_test(model, eps, test_idx, text_embs, args.max_length,
                                 device, amp_dtype)
            rec.update(test)
            print(f"  test: succ_final={test['test_succ_final_mean']:.3f}  "
                  f"fail_final={test['test_fail_final_mean']:.3f}  "
                  f"gap={test['test_gap']:+.3f}")
        history.append(rec)

        ckpt = os.path.join(args.ckpt_dir, f"{args.run_name}_epoch{epoch}.pth")
        # Format pinned to forge_env._init_visual_reward: `args` stays a
        # Namespace so getattr(cfg, "max_length") resolves.
        torch.save({"args": model_args,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch}, ckpt)
        print(f"  saved {ckpt}")

    with open(os.path.join(args.ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[visual] DONE. history -> {os.path.join(args.ckpt_dir, 'history.json')}")


if __name__ == "__main__":
    main()
