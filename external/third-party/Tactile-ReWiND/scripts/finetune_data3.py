"""Fine-tune a Tactile-ReWiND ckpt on the IsaacLab Forge `data_3_baseline` rollouts.

Inputs:
  * .npy episodes ({"Task", "Tactile" (T, 40, 25, 3) fp16, "Success" 0/1})
  * pretrained Tactile-ReWiND ckpt

Training signal:
  * Success episode → forward (or rewind) playback with ground-truth ramp 0→1
  * Failure episode → forward playback with ground-truth progress = 0
  * Single task → no mismatched-language negative (just use real failures)

A 90/10 stratified split is written to `<ckpt_dir>/test_files.json` so the
user can hold out the 10% test set when running `eval_reward_curves_data3.py
--include_files`.

Usage:
    python scripts/finetune_data3.py \\
        --pretrained /mnt/lab-tank/uber/Tactile-Reward/checkpoints_3ch_norm/tactile_rewind_epoch10.pth \\
        --data_dir /mnt/tank/tactile/tactile_dataset/data_3_baseline \\
        --ckpt_dir /mnt/lab-tank/uber/Tactile-Reward/checkpoints_data3_finetune \\
        --run_name data3_finetune \\
        --epochs 30 --batch_size 64 --steps_per_epoch 100 --lr 1e-5 \\
        --amp bf16 --compile
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from typing import List

import h5py  # noqa: F401  (kept for parity with sibling scripts)
import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.tactile_model import TactileReWiNDTransformer


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


def mean_pool(out, mask):
    tok = out[0]
    m = mask.unsqueeze(-1).expand(tok.size()).float()
    return (tok * m).sum(1) / m.sum(1).clamp(min=1e-9)


def embed_text(text: str, device: torch.device) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    enc = tok([text], padding=True, truncation=True, return_tensors="pt").to(device)
    m = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(device).eval()
    with torch.no_grad():
        out = m(**enc)
        e = mean_pool(out, enc["attention_mask"]).float().squeeze(0)
    del m, tok
    return e.cpu().numpy()


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
    test_idx = sorted(test)
    return train_idx, test_idx


class Data3Dataset(Dataset):
    """Wraps an in-memory list of {tactile, success} episodes."""

    def __init__(self, eps: List[dict], indices: List[int], text_embs: np.ndarray,
                 max_length: int = 16, batch_size: int = 64, steps_per_epoch: int = 100,
                 rewind_ratio: float = 0.5, success_prob: float = 0.5,
                 normalize_mode: str = "off", seed: int = 0):
        if normalize_mode not in ("off", "global", "per_channel"):
            raise ValueError(f"normalize_mode must be off|global|per_channel, "
                             f"got {normalize_mode!r}")
        self.eps = eps
        # text_embs: (N_paraphrases, 384) — one row per task instruction paraphrase.
        # Each sample picks one at random for language robustness.
        if text_embs.ndim == 1:
            text_embs = text_embs[None, :]
        self.text_embs = text_embs.astype(np.float32)
        self.max_length = max_length
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.rewind_ratio = rewind_ratio
        self.success_prob = success_prob
        self.normalize_mode = normalize_mode
        self._succ_idx = [i for i in indices if eps[i]["success"] == 1]
        self._fail_idx = [i for i in indices if eps[i]["success"] == 0]
        if not self._succ_idx or not self._fail_idx:
            raise ValueError(f"need at least one success and one fail episode in train "
                             f"(got {len(self._succ_idx)}/{len(self._fail_idx)})")
        # Per-worker RNG seeded with worker_id in worker_init_fn.
        self._rng = random.Random(seed)

    def __len__(self):
        return self.batch_size * self.steps_per_epoch

    def _sample_forward(self, N: int):
        """Full alignment with inference patch: random K∈[1, N], linspace 16 over
        0..K-1 (pad with frame K-1 when K<max_length), label progress (idx+1)/N.
        Matches inference where step K (even small K<16) sees buffer [0..K-1]
        padded to 16 slots."""
        K = self._rng.randint(1, N) if N >= 1 else 1
        if K >= self.max_length:
            idx = np.round(np.linspace(0, K - 1, self.max_length)).astype(np.int64)
        else:
            idx = np.concatenate([np.arange(K, dtype=np.int64),
                                  np.full(self.max_length - K, K - 1, dtype=np.int64)])
        prog = ((idx + 1) / N).astype(np.float32)
        return idx, prog

    def _sample_rewind(self, N: int):
        """Forward to random K, then reverse back to 0. Total length 2K-1, then
        linspace 16. Progress goes up to ≈K/N then back down to ≈1/N."""
        if N < 2:
            return self._sample_forward(N)
        # Rewind needs K ≥ 2 for the reverse to exist; lower bound at 2 (instead
        # of max_length) so rewind augmentation also sees short trajectories.
        K = self._rng.randint(2, N) if N >= 2 else 2
        fwd_idx = np.arange(K, dtype=np.int64)
        rev_idx = np.arange(K - 2, -1, -1, dtype=np.int64)            # K-2 .. 0
        fwd_prog = ((np.arange(K) + 1) / N).astype(np.float32)         # 1/N .. K/N
        rev_prog = fwd_prog[::-1][1:K].copy()                          # (K-1)/N .. 1/N
        full_idx = np.concatenate([fwd_idx, rev_idx])
        full_prog = np.concatenate([fwd_prog, rev_prog]).astype(np.float32)
        T = full_idx.shape[0]                                          # 2K-1
        if T >= self.max_length:
            local = np.round(np.linspace(0, T - 1, self.max_length)).astype(int)
            idx = full_idx[local]
            prog = full_prog[local]
        else:
            pad = self.max_length - T
            idx = np.concatenate([full_idx, np.full(pad, full_idx[-1], dtype=full_idx.dtype)])
            prog = np.concatenate([full_prog, np.full(pad, full_prog[-1], dtype=full_prog.dtype)])
        return idx, prog

    def _resize(self, idx: np.ndarray, prog: np.ndarray):
        T = idx.shape[0]
        if T < self.max_length:
            pad = self.max_length - T
            idx = np.concatenate([idx, np.full(pad, idx[-1], dtype=idx.dtype)])
            prog = np.concatenate([prog, np.full(pad, prog[-1], dtype=prog.dtype)])
        elif T > self.max_length:
            local = np.linspace(0, T - 1, self.max_length).astype(int)
            idx = idx[local]
            prog = prog[local]
        return idx, prog

    def __getitem__(self, _):
        is_success = self._rng.random() < self.success_prob
        pool = self._succ_idx if is_success else self._fail_idx
        ep = self.eps[self._rng.choice(pool)]
        N = ep["tactile"].shape[0]
        if is_success and self._rng.random() < self.rewind_ratio:
            idx, prog = self._sample_rewind(N)
        else:
            idx, prog = self._sample_forward(N)
        if not is_success:
            prog = np.zeros_like(prog)            # failure → ground-truth reward 0
        frames = ep["tactile"][idx].astype(np.float32, copy=True)  # (T, 40, 25, 3)
        if self.normalize_mode == "global":
            denom = np.abs(frames).max(axis=None, keepdims=True)
            np.maximum(denom, 1e-6, out=denom)
            frames /= denom
        elif self.normalize_mode == "per_channel":
            denom = np.abs(frames).max(axis=(0, 1, 2), keepdims=True)
            np.maximum(denom, 1e-6, out=denom)
            frames /= denom
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
        # Random paraphrase per call.
        t_idx = self._rng.randrange(self.text_embs.shape[0])
        return {
            "video_array": x,
            "text_array": torch.from_numpy(self.text_embs[t_idx]),
            "progress": torch.from_numpy(prog),
            "class_label": torch.full_like(torch.from_numpy(prog), 1.0 if is_success else 0.0),
        }


def worker_init_fn(worker_id):
    # Each worker gets its own deterministic RNG so different workers don't
    # produce identical sample sequences.
    info = torch.utils.data.get_worker_info()
    seed = (info.seed + worker_id) % (2 ** 31)
    info.dataset._rng = random.Random(seed)


def load_pretrained(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("args", {})
    isaaclab_aligned = bool(cfg.get("isaaclab_aligned", True))
    num_strided = cfg.get("num_strided_layers", 0) or 3
    bimanual_axis = "height" if isaaclab_aligned else "width"
    model = TactileReWiNDTransformer(
        max_length=cfg.get("max_length", 16),
        text_dim=384,
        hidden_dim=cfg.get("hidden_dim", 512),
        num_heads=cfg.get("num_heads", 8),
        num_layers=cfg.get("num_layers", 4),
        per_hand_dim=cfg.get("per_hand_dim", 384),
        num_strided_layers=num_strided,
        bimanual_axis=bimanual_axis,
        in_channels=cfg.get("in_channels", 3),
    ).to(device)
    sd = state["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model, cfg


def build_scratch_model(args, device: torch.device):
    """Fresh model + cfg dict for `--from_scratch`. Defaults mirror the
    AnyTouch2 + isaaclab_aligned 3ch training so the same eval scripts work."""
    model = TactileReWiNDTransformer(
        max_length=args.max_length, text_dim=384,
        hidden_dim=args.hidden_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, per_hand_dim=args.per_hand_dim,
        num_strided_layers=args.num_strided_layers,
        bimanual_axis="height", in_channels=args.in_channels,
    ).to(device)
    cfg = {"max_length": args.max_length, "in_channels": args.in_channels,
           "num_strided_layers": args.num_strided_layers,
           "hidden_dim": args.hidden_dim, "num_heads": args.num_heads,
           "num_layers": args.num_layers, "per_hand_dim": args.per_hand_dim,
           "isaaclab_aligned": True, "normalize_mode": "off"}
    return model, cfg


def train_step(model, batch, optimizer, scheduler, device, clip_grad, amp_dtype):
    model.train()
    optimizer.zero_grad()
    video = batch["video_array"].to(device, non_blocking=True).float()
    text = batch["text_array"].to(device, non_blocking=True).float()
    progress = batch["progress"].to(device, non_blocking=True).float()
    label = batch["class_label"].to(device, non_blocking=True).float()

    use_amp = amp_dtype is not None and device.type == "cuda"
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        pred = model(video, text).squeeze(-1)            # (B, T)
        pred_t = pred[:, 1:]
        target_t = progress[:, 1:]
        is_pos = label[:, 0].bool()                      # (B,)
        pos_loss = mse_loss(pred_t[is_pos], target_t[is_pos]) if is_pos.any() else pred.sum() * 0
        neg_loss = mse_loss(pred_t[~is_pos], target_t[~is_pos]) if (~is_pos).any() else pred.sum() * 0
        n_pos = int(is_pos.sum())
        n_neg = int((~is_pos).sum())
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
        "n_pos": n_pos, "n_neg": n_neg,
    }


def evaluate_test(model, eps, test_idx, text_embs, max_length, normalize_mode,
                  device, amp_dtype):
    """Quick held-out fwd reward: success_final_mean - fail_final_mean.
    Uses the first paraphrase as the deterministic eval text."""
    model.eval()
    text = torch.from_numpy(text_embs[0].astype(np.float32)).float().unsqueeze(0).to(device)
    succ_finals, fail_finals = [], []
    use_amp = amp_dtype is not None and device.type == "cuda"
    for i in test_idx:
        ep = eps[i]
        N = ep["tactile"].shape[0]
        if N >= max_length:
            idx = np.round(np.linspace(0, N - 1, max_length)).astype(np.int64)
        else:
            idx = np.concatenate([np.arange(N), np.full(max_length - N, N - 1, dtype=np.int64)])
        frames = ep["tactile"][idx].astype(np.float32, copy=True)
        if normalize_mode == "global":
            denom = np.abs(frames).max(axis=None, keepdims=True)
            np.maximum(denom, 1e-6, out=denom)
            frames /= denom
        elif normalize_mode == "per_channel":
            denom = np.abs(frames).max(axis=(0, 1, 2), keepdims=True)
            np.maximum(denom, 1e-6, out=denom)
            frames /= denom
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            pred = model(x, text).squeeze(-1).squeeze(0).float().cpu().numpy()
        (succ_finals if ep["success"] else fail_finals).append(float(pred[-1]))
    s_mean = float(np.mean(succ_finals)) if succ_finals else float("nan")
    f_mean = float(np.mean(fail_finals)) if fail_finals else float("nan")
    return {"test_n_succ": len(succ_finals), "test_n_fail": len(fail_finals),
            "test_succ_final_mean": s_mean, "test_fail_final_mean": f_mean,
            "test_gap": s_mean - f_mean}


def main():
    ap = argparse.ArgumentParser()
    init_grp = ap.add_mutually_exclusive_group(required=True)
    init_grp.add_argument("--pretrained", default=None,
                          help="Init weights from this ckpt (fine-tune mode).")
    init_grp.add_argument("--from_scratch", action="store_true",
                          help="Train a fresh model with the architecture defaults.")
    ap.add_argument("--data_dirs", nargs="+", required=True,
                    help="One or more dataset directories. Each is globbed "
                         "recursively (`**/*.npy`) so curriculum-style trees with "
                         "`ep_XXX/*.npy` subdirs work without extra flags.")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--run_name", default="data3_finetune")
    ap.add_argument("--task_texts", nargs="+",
                    default=["grasp peg and insert to another hole"],
                    help="One or more instruction paraphrases. A random one is "
                         "drawn per sample during training; eval uses the first.")

    # Model arch — only consulted when --from_scratch (otherwise inherited).
    ap.add_argument("--in_channels", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--per_hand_dim", type=int, default=384)
    ap.add_argument("--num_strided_layers", type=int, default=3)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps_per_epoch", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min_lr", type=float, default=1e-7)
    ap.add_argument("--clip_grad", action="store_true")

    ap.add_argument("--rewind_ratio", type=float, default=0.5)
    ap.add_argument("--success_prob", type=float, default=0.5,
                    help="Probability of drawing a success episode (vs fail) per sample. "
                         "0.5 = class-balanced regardless of base rate.")
    ap.add_argument("--max_length", type=int, default=16)
    ap.add_argument("--normalize", choices=["off", "global", "per_channel", "inherit"],
                    default="inherit",
                    help="'inherit' (default) reuses the pretrained ckpt's normalize mode "
                         "so train/eval preprocessing stays aligned. Specify 'global' or "
                         "'per_channel' to override (but then the ckpt's encoder may have "
                         "trained on a different input scale).")

    ap.add_argument("--max_episodes", type=int, default=0,
                    help="Cap total episodes loaded into RAM (random sample if "
                         "exceeded). Needed for curriculum dirs with 10K+ files. "
                         "0 = load all.")
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--test_eval_every", type=int, default=1,
                    help="Evaluate on the held-out test set every N epochs (0 = never).")

    ap.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    ap.add_argument("--compile", dest="compile_model",
                    action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--compile_mode", default="reduce-overhead",
                    choices=["default", "reduce-overhead", "max-autotune"])

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(os.path.join(args.ckpt_dir, "finetune_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ─── load model + cfg ───────────────────────────────────────────────────
    if args.pretrained:
        model, cfg = load_pretrained(args.pretrained, device)
        src = f"pretrained={args.pretrained}"
    else:
        model, cfg = build_scratch_model(args, device)
        src = "from_scratch"
    inherited_mode = cfg.get("normalize_mode")
    if inherited_mode is None:
        inherited_mode = "per_channel" if cfg.get("normalize_per_channel") else "off"
    normalize_mode = inherited_mode if args.normalize == "inherit" else args.normalize
    print(f"[finetune] init: {src}")
    print(f"[finetune] cfg: in_channels={cfg.get('in_channels')}, "
          f"max_length={cfg.get('max_length')}, "
          f"normalize_mode={inherited_mode} → using {normalize_mode}")

    # ─── load + split data ──────────────────────────────────────────────────
    files: list[str] = []
    for d in args.data_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"data_dir does not exist: {d}")
        # Recursive glob handles both flat (peg) and ep_XXX-nested (gear/nut curriculum).
        files.extend(glob.glob(os.path.join(d, "**", "*.npy"), recursive=True))
    files = sorted(files)
    print(f"[finetune] scanning {len(files)} files across {len(args.data_dirs)} dir(s)")
    if args.max_episodes and len(files) > args.max_episodes:
        rng = random.Random(args.seed)
        files = rng.sample(files, args.max_episodes)
        files.sort()
        print(f"[finetune] capped to {args.max_episodes} (seed={args.seed})")
    eps = []
    for p in files:
        try:
            d = np.load(p, allow_pickle=True).item()
        except Exception as e:
            print(f"  skip {p}: {e}", file=sys.stderr)
            continue
        tac = d["Tactile"]
        if tac.ndim != 4 or tac.shape[1:] != (40, 25, 3):
            continue
        # Use the full path as the identifier so the same basename across
        # different `ep_XXX/` subdirs doesn't collide.
        eps.append({"file": os.path.relpath(p), "tactile": tac,
                    "success": int(d["Success"])})
    n_succ = sum(e["success"] for e in eps)
    n_fail = len(eps) - n_succ
    print(f"[finetune] loaded {len(eps)} episodes: {n_succ} success, {n_fail} fail")

    train_idx, test_idx = stratified_split(eps, args.test_ratio, args.seed)
    test_succ = sum(eps[i]["success"] for i in test_idx)
    test_fail = len(test_idx) - test_succ
    print(f"[finetune] split: {len(train_idx)} train, {len(test_idx)} test "
          f"(test {test_succ} success / {test_fail} fail)")
    test_files = [eps[i]["file"] for i in test_idx]
    with open(os.path.join(args.ckpt_dir, "test_files.json"), "w") as f:
        json.dump({"test_files": test_files, "n_success": test_succ,
                   "n_fail": test_fail, "seed": args.seed}, f, indent=2)
    print(f"[finetune] wrote held-out filenames → "
          f"{os.path.join(args.ckpt_dir, 'test_files.json')}")

    # ─── embed task texts (all paraphrases, one MiniLM call) ────────────────
    text_embs = np.stack([embed_text(t, device) for t in args.task_texts], axis=0)
    print(f"[finetune] {len(args.task_texts)} paraphrases → emb shape={text_embs.shape}")
    for t in args.task_texts:
        print(f"  - {t!r}")

    # ─── dataset / loader ───────────────────────────────────────────────────
    train_ds = Data3Dataset(
        eps=eps, indices=train_idx, text_embs=text_embs,
        max_length=args.max_length, batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch, rewind_ratio=args.rewind_ratio,
        success_prob=args.success_prob, normalize_mode=normalize_mode,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

    # ─── AMP / compile ──────────────────────────────────────────────────────
    amp_dtype = None
    if args.amp != "off" and device.type == "cuda":
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        print(f"[finetune] amp dtype={amp_dtype}")
    if args.compile_model:
        print(f"[finetune] compile mode={args.compile_mode}")
        model = torch.compile(model, mode=args.compile_mode)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = CosineWithMinLR(optimizer, max_steps=total_steps,
                                max_lr=args.lr, min_lr=args.min_lr)

    # ─── train loop ─────────────────────────────────────────────────────────
    global_step = 0
    history = []
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        last_metrics = None
        for batch in pbar:
            metrics = train_step(model, batch, optimizer, scheduler, device,
                                 args.clip_grad, amp_dtype)
            global_step += 1
            last_metrics = metrics
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}",
                             pos=f"{metrics['pos_loss']:.4f}",
                             neg=f"{metrics['neg_loss']:.4f}")
        # held-out eval
        ep_record = {"epoch": epoch, "step": global_step, "train_loss": last_metrics["loss"]}
        if args.test_eval_every and (epoch + 1) % args.test_eval_every == 0:
            test = evaluate_test(model, eps, test_idx, text_embs, args.max_length,
                                 normalize_mode, device, amp_dtype)
            ep_record.update(test)
            print(f"  test: succ_final={test['test_succ_final_mean']:.3f}  "
                  f"fail_final={test['test_fail_final_mean']:.3f}  "
                  f"gap={test['test_gap']:+.3f}")
        history.append(ep_record)
        # save ckpt
        save_model = getattr(model, "_orig_mod", model)
        out_cfg = dict(vars(args))
        # Preserve fields the eval scripts rely on (they read from saved args).
        out_cfg.update({
            "in_channels": cfg.get("in_channels", 3),
            "max_length": args.max_length,
            "isaaclab_aligned": True,         # data_3 is already aligned
            "num_strided_layers": cfg.get("num_strided_layers", 3) or 3,
            "hidden_dim": cfg.get("hidden_dim", 512),
            "num_heads": cfg.get("num_heads", 8),
            "num_layers": cfg.get("num_layers", 4),
            "per_hand_dim": cfg.get("per_hand_dim", 384),
            "normalize_mode": normalize_mode,
            # Legacy alias so old eval scripts still recognise per_channel runs.
            "normalize_per_channel": normalize_mode == "per_channel",
        })
        ckpt = os.path.join(args.ckpt_dir, f"{args.run_name}_epoch{epoch}.pth")
        torch.save({"args": out_cfg, "model_state_dict": save_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch, "global_step": global_step,
                    "history": history}, ckpt)
        print(f"  saved {ckpt}")

    with open(os.path.join(args.ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[finetune] DONE. history → {os.path.join(args.ckpt_dir, 'history.json')}")


if __name__ == "__main__":
    main()
