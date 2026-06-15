"""Multi-task fine-tune / from-scratch trainer.

Drives a single model across multiple tasks. Each task is described by a JSON
spec (--task_specs_json):

    {
      "peg":  {"data_dirs": [...], "task_texts": [...], "max_episodes": 0},
      "gear": {"data_dirs": [...], "task_texts": [...], "max_episodes": 5000},
      "nut":  {"data_dirs": [...], "task_texts": [...], "max_episodes": 5000}
    }

Sampling: each __getitem__ picks a task uniformly, then within that task picks
success-vs-fail by `--success_prob`, then picks one of the task's paraphrases
at random. Per-task stratified 90/10 split; held-out filenames are written to
<ckpt_dir>/test_files_<task>.json for sibling eval runs.

Reuses helpers from finetune_data3 to avoid duplication.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for finetune_data3

from finetune_data3 import (
    CosineWithMinLR,
    embed_text,
    load_pretrained,
    build_scratch_model,
    stratified_split,
    train_step,
)


class MultiTaskDataset(Dataset):
    """Episodes from many tasks, sampled with per-batch task balance.

    `task_eps`: dict[task_name -> list of episode dicts each with keys
                {file, tactile (T,40,25,3 fp16), success (0/1)}].
    `task_train_idx`: dict[task_name -> list of integer indices into task_eps[task]].
    `task_text_embs`: dict[task_name -> ndarray (N_paraphrases, 384)].
    """

    def __init__(self, task_eps: Dict[str, List[dict]],
                 task_train_idx: Dict[str, List[int]],
                 task_text_embs: Dict[str, np.ndarray],
                 max_length: int = 16, batch_size: int = 64,
                 steps_per_epoch: int = 100, rewind_ratio: float = 0.5,
                 success_prob: float = 0.5,
                 normalize_mode: str = "off", seed: int = 0):
        if normalize_mode not in ("off", "global", "per_channel"):
            raise ValueError(f"normalize_mode must be off|global|per_channel, "
                             f"got {normalize_mode!r}")
        self.task_eps = task_eps
        self.task_text_embs = {t: e.astype(np.float32) for t, e in task_text_embs.items()}
        self.max_length = max_length
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.rewind_ratio = rewind_ratio
        self.success_prob = success_prob
        self.normalize_mode = normalize_mode
        # Pre-split each task's train indices into success / fail buckets.
        self._task_succ: Dict[str, List[int]] = {}
        self._task_fail: Dict[str, List[int]] = {}
        for t, idxs in task_train_idx.items():
            self._task_succ[t] = [i for i in idxs if task_eps[t][i]["success"] == 1]
            self._task_fail[t] = [i for i in idxs if task_eps[t][i]["success"] == 0]
            if not self._task_succ[t] or not self._task_fail[t]:
                raise ValueError(f"task {t!r} needs ≥1 success and ≥1 fail in train "
                                 f"(got {len(self._task_succ[t])}/{len(self._task_fail[t])})")
        self._task_names = sorted(task_train_idx.keys())
        self._rng = random.Random(seed)

    def __len__(self):
        return self.batch_size * self.steps_per_epoch

    def _sample_forward(self, N: int):
        """Hybrid: random K∈[max_length, N] forward extent, prog=(idx+1)/N."""
        K = self._rng.randint(self.max_length, N) if N > self.max_length else N
        if K >= self.max_length:
            idx = np.round(np.linspace(0, K - 1, self.max_length)).astype(np.int64)
        else:
            idx = np.concatenate([np.arange(K, dtype=np.int64),
                                  np.full(self.max_length - K, K - 1, dtype=np.int64)])
        prog = ((idx + 1) / N).astype(np.float32)
        return idx, prog

    def _sample_rewind(self, N: int):
        """Forward to random K then reverse back to 0. Total 2K-1, linspace 16."""
        if N < 2:
            return self._sample_forward(N)
        K = self._rng.randint(self.max_length, N) if N > self.max_length else N
        fwd_idx = np.arange(K, dtype=np.int64)
        rev_idx = np.arange(K - 2, -1, -1, dtype=np.int64)
        fwd_prog = ((np.arange(K) + 1) / N).astype(np.float32)
        rev_prog = fwd_prog[::-1][1:K].copy()
        full_idx = np.concatenate([fwd_idx, rev_idx])
        full_prog = np.concatenate([fwd_prog, rev_prog]).astype(np.float32)
        T = full_idx.shape[0]
        if T >= self.max_length:
            local = np.round(np.linspace(0, T - 1, self.max_length)).astype(int)
            idx = full_idx[local]
            prog = full_prog[local]
        else:
            pad = self.max_length - T
            idx = np.concatenate([full_idx, np.full(pad, full_idx[-1], dtype=full_idx.dtype)])
            prog = np.concatenate([full_prog, np.full(pad, full_prog[-1], dtype=full_prog.dtype)])
        return idx, prog

    def __getitem__(self, _):
        task = self._rng.choice(self._task_names)
        is_success = self._rng.random() < self.success_prob
        pool = self._task_succ[task] if is_success else self._task_fail[task]
        ep = self.task_eps[task][self._rng.choice(pool)]
        N = ep["tactile"].shape[0]
        if is_success and self._rng.random() < self.rewind_ratio:
            idx, prog = self._sample_rewind(N)
        else:
            idx, prog = self._sample_forward(N)
        if not is_success:
            prog = np.zeros_like(prog)
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
        embs = self.task_text_embs[task]
        t_idx = self._rng.randrange(embs.shape[0])
        return {
            "video_array": x,
            "text_array": torch.from_numpy(embs[t_idx]),
            "progress": torch.from_numpy(prog),
            "class_label": torch.full_like(torch.from_numpy(prog), 1.0 if is_success else 0.0),
        }


def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    seed = (info.seed + worker_id) % (2 ** 31)
    info.dataset._rng = random.Random(seed)


def evaluate_test_multitask(model, task_eps, task_test_idx, task_text_embs,
                            max_length, normalize_mode, device, amp_dtype):
    """Per-task held-out gap. Returns dict[task -> {succ_final, fail_final, gap}]
    plus an 'avg_gap' aggregate."""
    model.eval()
    use_amp = amp_dtype is not None and device.type == "cuda"
    out: dict = {}
    for task in sorted(task_eps.keys()):
        text = torch.from_numpy(task_text_embs[task][0].astype(np.float32))
        text = text.float().unsqueeze(0).to(device)
        succ_finals, fail_finals = [], []
        for i in task_test_idx[task]:
            ep = task_eps[task][i]
            N = ep["tactile"].shape[0]
            if N >= max_length:
                idx = np.round(np.linspace(0, N - 1, max_length)).astype(np.int64)
            else:
                idx = np.concatenate([np.arange(N),
                                      np.full(max_length - N, N - 1, dtype=np.int64)])
            frames = ep["tactile"][idx].astype(np.float32, copy=True)
            if normalize_mode == "global":
                denom = np.abs(frames).max(axis=None, keepdims=True)
                np.maximum(denom, 1e-6, out=denom)
                frames /= denom
            elif normalize_mode == "per_channel":
                denom = np.abs(frames).max(axis=(0, 1, 2), keepdims=True)
                np.maximum(denom, 1e-6, out=denom)
                frames /= denom
            x = (torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
                 .unsqueeze(0).to(device))
            with torch.no_grad(), torch.amp.autocast(device_type="cuda",
                                                     dtype=amp_dtype, enabled=use_amp):
                pred = model(x, text).squeeze(-1).squeeze(0).float().cpu().numpy()
            (succ_finals if ep["success"] else fail_finals).append(float(pred[-1]))
        s = float(np.mean(succ_finals)) if succ_finals else float("nan")
        f = float(np.mean(fail_finals)) if fail_finals else float("nan")
        out[task] = {"n_succ": len(succ_finals), "n_fail": len(fail_finals),
                     "succ_final": s, "fail_final": f, "gap": s - f}
    valid = [v["gap"] for v in out.values() if not math.isnan(v["gap"])]
    out["avg_gap"] = float(np.mean(valid)) if valid else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    init_grp = ap.add_mutually_exclusive_group(required=True)
    init_grp.add_argument("--pretrained", default=None)
    init_grp.add_argument("--from_scratch", action="store_true")
    ap.add_argument("--task_specs_json", required=True,
                    help="JSON: {task: {data_dirs, task_texts, max_episodes}}")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--run_name", default="multitask")

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
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--clip_grad", action="store_true")

    ap.add_argument("--rewind_ratio", type=float, default=0.5)
    ap.add_argument("--success_prob", type=float, default=0.5)
    ap.add_argument("--max_length", type=int, default=16)
    ap.add_argument("--normalize", choices=["off", "global", "per_channel", "inherit"],
                    default="inherit")
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--test_eval_every", type=int, default=1)

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
    print(f"[multitask] init: {src}")
    print(f"[multitask] normalize_mode: inherited={inherited_mode} → using {normalize_mode}")

    # ─── load task specs + episodes ─────────────────────────────────────────
    with open(args.task_specs_json) as f:
        specs = json.load(f)
    print(f"[multitask] {len(specs)} tasks: {list(specs.keys())}")

    task_eps: Dict[str, List[dict]] = {}
    task_text_embs: Dict[str, np.ndarray] = {}
    task_train_idx: Dict[str, List[int]] = {}
    task_test_idx: Dict[str, List[int]] = {}

    rng_split = random.Random(args.seed)
    import glob as _glob
    for task, spec in specs.items():
        files: List[str] = []
        for d in spec["data_dirs"]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"task={task!r}: missing dir {d}")
            files.extend(_glob.glob(os.path.join(d, "**", "*.npy"), recursive=True))
        files = sorted(files)
        cap = int(spec.get("max_episodes", 0))
        if cap and len(files) > cap:
            files = random.Random(args.seed + hash(task) % 1000).sample(files, cap)
            files.sort()
        print(f"[multitask] task={task}: scanning {len(files)} files "
              f"(cap={cap or 'none'})")
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
            eps.append({"file": os.path.relpath(p), "tactile": tac,
                        "success": int(d["Success"])})
        n_succ = sum(e["success"] for e in eps)
        n_fail = len(eps) - n_succ
        print(f"  loaded {len(eps)} episodes: {n_succ} success, {n_fail} fail")
        task_eps[task] = eps
        # Per-task stratified split
        tr, te = stratified_split(eps, args.test_ratio, args.seed + hash(task) % 1000)
        task_train_idx[task] = tr
        task_test_idx[task] = te
        # Embed paraphrases
        embs = np.stack([embed_text(t, device) for t in spec["task_texts"]], axis=0)
        task_text_embs[task] = embs
        print(f"  {len(spec['task_texts'])} paraphrases → emb {embs.shape}")
        for s in spec["task_texts"]:
            print(f"    - {s!r}")
        # Write held-out filenames (for sibling eval script).
        test_files = [eps[i]["file"] for i in te]
        with open(os.path.join(args.ckpt_dir, f"test_files_{task}.json"), "w") as f:
            json.dump({"test_files": test_files,
                       "n_success": sum(eps[i]["success"] for i in te),
                       "n_fail": sum(1 - eps[i]["success"] for i in te),
                       "task": task, "seed": args.seed}, f, indent=2)

    # ─── dataset / loader ───────────────────────────────────────────────────
    train_ds = MultiTaskDataset(
        task_eps=task_eps, task_train_idx=task_train_idx,
        task_text_embs=task_text_embs,
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

    amp_dtype = None
    if args.amp != "off" and device.type == "cuda":
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        print(f"[multitask] amp dtype={amp_dtype}")
    if args.compile_model:
        print(f"[multitask] compile mode={args.compile_mode}")
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
        ep_record = {"epoch": epoch, "step": global_step,
                     "train_loss": last_metrics["loss"]}
        if args.test_eval_every and (epoch + 1) % args.test_eval_every == 0:
            test = evaluate_test_multitask(
                model, task_eps, task_test_idx, task_text_embs,
                args.max_length, normalize_mode, device, amp_dtype)
            ep_record["test"] = test
            parts = [f"{t}={v['gap']:+.3f}" for t, v in test.items() if t != "avg_gap"]
            print(f"  test gaps: " + "  ".join(parts) + f"  | avg={test['avg_gap']:+.3f}")
        history.append(ep_record)
        save_model = getattr(model, "_orig_mod", model)
        out_cfg = dict(vars(args))
        out_cfg.update({
            "in_channels": cfg.get("in_channels", args.in_channels),
            "max_length": args.max_length,
            "isaaclab_aligned": True,
            "num_strided_layers": cfg.get("num_strided_layers", args.num_strided_layers) or 3,
            "hidden_dim": cfg.get("hidden_dim", args.hidden_dim),
            "num_heads": cfg.get("num_heads", args.num_heads),
            "num_layers": cfg.get("num_layers", args.num_layers),
            "per_hand_dim": cfg.get("per_hand_dim", args.per_hand_dim),
            "normalize_mode": normalize_mode,
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
    print(f"[multitask] DONE. history → {os.path.join(args.ckpt_dir, 'history.json')}")


if __name__ == "__main__":
    main()
