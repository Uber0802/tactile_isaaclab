"""Drive the 3-task × 2-init experiment: train + eval + summary.

Tasks: peg / gear / nut.
Inits: scratch / anytouch2-finetune.
Eval splits: single-pos held-out (per training) + multi-pos (per task, peg skipped).

Outputs go under <root>/<task>_<init>/ for ckpts and
<root>/eval/<task>_<init>_<split>/ for eval plots+JSON, plus a top-level
summary.json + summary.png comparing the best ckpt of every (task, init).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

PY = "/mnt/tank/uber/conda-envs/tactile_isaaclab/bin/python"
REPO = "/mnt/home/uber/tactile_isaaclab/external/third-party/Tactile-ReWiND"

# Per-task config — only single-pos dir is required; multi-pos is optional.
TASKS = {
    "peg": {
        "single_dir": "/mnt/tank/tactile/tactile_dataset/data_3_baseline",
        "multi_dir": None,
        "task_texts": [
            "grasp peg and insert to another hole",
            "pick up the peg and insert it into the second hole",
            "use the gripper to grab the peg and place it in the other hole",
            "grasp the cylindrical peg and slot it into the adjacent hole",
            "transfer the peg from one hole to another",
        ],
    },
    "gear": {
        "single_dir": "/mnt/tank/tactile/tactile_dataset/gearpickplace_curriculum",
        "multi_dir": "/mnt/tank/tactile/tactile_dataset/gearpickplace_baselineA copy",
        "task_texts": [
            "pick up the gear and mesh it onto the shaft",
            "grasp the gear and slide it down onto the gear shaft",
            "use the gripper to pick the gear and engage it on the shaft",
            "grab the gear and place it onto the meshing shaft",
            "lift the gear and mesh it onto the rotating shaft",
        ],
    },
    "nut": {
        "single_dir": "/mnt/tank/tactile/tactile_dataset/nutpickplace_curriculum",
        "multi_dir": "/mnt/tank/tactile/tactile_dataset/nutpickplace_baselineA copy",
        "task_texts": [
            "pick up the nut and thread it onto the bolt",
            "grasp the nut and screw it onto the threaded shaft",
            "use the gripper to pick the nut and thread it onto the rod",
            "grab the nut and thread it onto the bolt",
            "lift the nut and thread it onto the screw",
        ],
    },
}


def sh(cmd: list, log_path: str):
    """Run a subprocess, tee output to a log file; raise on non-zero exit."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(f"  → {' '.join(repr(c) if ' ' in c else c for c in cmd)}")
    print(f"    log: {log_path}")
    t0 = time.time()
    with open(log_path, "w") as logf:
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO)
        rc = p.wait()
    dur = time.time() - t0
    print(f"    exit={rc}  duration={dur:.1f}s")
    if rc != 0:
        # Show last 30 lines of the failing log so failures are debuggable.
        try:
            tail = open(log_path).read().splitlines()[-30:]
            print("  --- last 30 log lines ---")
            for line in tail:
                print(f"    {line}")
        except Exception:
            pass
        raise RuntimeError(f"command failed: {' '.join(cmd)} (rc={rc})")
    return dur


def best_epoch_from_history(history_path: str) -> tuple[int, float]:
    """Pick the epoch whose test gap (succ_final - fail_final) is highest."""
    with open(history_path) as f:
        hist = json.load(f)
    rows = [(h["epoch"], h.get("test_gap", float("-inf"))) for h in hist
            if "test_gap" in h]
    if not rows:
        raise RuntimeError(f"no test_gap entries in {history_path}")
    best_ep, best_gap = max(rows, key=lambda r: r[1])
    return best_ep, best_gap


def run_train(task: str, init: str, gpu: int, args, root: str) -> dict:
    """Train one cell; return {ckpt_dir, best_ckpt, best_gap, duration}."""
    cfg = TASKS[task]
    run_name = f"{task}_{init}"
    ckpt_dir = os.path.join(root, run_name)
    log_path = os.path.join(root, "logs", f"train_{run_name}.log")

    cmd = [
        PY, "scripts/finetune_data3.py",
        "--ckpt_dir", ckpt_dir, "--run_name", run_name,
        "--data_dirs", cfg["single_dir"],
        "--task_texts", *cfg["task_texts"],
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--steps_per_epoch", str(args.steps_per_epoch),
        "--num_workers", str(args.num_workers),
        "--prefetch_factor", str(args.prefetch_factor),
        "--max_episodes", str(args.cap_episodes),
        "--test_ratio", "0.1",
        "--amp", "bf16",
        "--rewind_ratio", "0.5",
        "--success_prob", "0.5",
        "--seed", str(args.seed),
    ]
    if init == "scratch":
        cmd += ["--from_scratch", "--normalize", "global",
                "--lr", str(args.lr_scratch)]
    elif init == "at2ft":
        cmd += ["--pretrained", args.at2_ckpt,
                "--normalize", "inherit",
                "--lr", str(args.lr_finetune)]
    else:
        raise ValueError(f"unknown init {init!r}")
    if args.compile:
        cmd += ["--compile"]
    else:
        cmd += ["--no-compile"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"\n[train] {task}/{init}  gpu={gpu}")
    t0 = time.time()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as logf:
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                             cwd=REPO, env=env)
        rc = p.wait()
    dur = time.time() - t0
    print(f"  exit={rc}  duration={dur:.1f}s  log={log_path}")
    if rc != 0:
        tail = open(log_path).read().splitlines()[-30:]
        for line in tail:
            print(f"    {line}")
        raise RuntimeError(f"train failed: {task}/{init}")

    best_ep, best_gap = best_epoch_from_history(os.path.join(ckpt_dir, "history.json"))
    best_ckpt = os.path.join(ckpt_dir, f"{run_name}_epoch{best_ep}.pth")
    print(f"  best: epoch{best_ep}  test_gap={best_gap:+.3f}  ckpt={best_ckpt}")
    return {"ckpt_dir": ckpt_dir, "best_ckpt": best_ckpt, "best_epoch": best_ep,
            "best_gap_during_training": best_gap, "duration_s": dur,
            "test_files_json": os.path.join(ckpt_dir, "test_files.json")}


def run_eval(task: str, init: str, split: str, ckpt: str, data_dir: str,
             test_files_json: str | None, gpu: int, root: str) -> dict:
    """Run eval_reward_curves_data3.py for one (task, init, split) cell."""
    out_dir = os.path.join(root, "eval", f"{task}_{init}_{split}")
    log_path = os.path.join(root, "logs", f"eval_{task}_{init}_{split}.log")
    cmd = [
        PY, "scripts/eval_reward_curves_data3.py",
        "--ckpts", ckpt,
        "--data_dirs", data_dir,
        "--output_dir", out_dir,
    ]
    if test_files_json:
        cmd += ["--test_files_json", test_files_json]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"\n[eval]  {task}/{init}/{split}  gpu={gpu}")
    t0 = time.time()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as logf:
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                             cwd=REPO, env=env)
        rc = p.wait()
    dur = time.time() - t0
    print(f"  exit={rc}  duration={dur:.1f}s  log={log_path}")
    if rc != 0:
        tail = open(log_path).read().splitlines()[-30:]
        for line in tail:
            print(f"    {line}")
        raise RuntimeError(f"eval failed: {task}/{init}/{split}")

    # Read the per-ckpt JSON the eval wrote.
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    json_path = os.path.join(out_dir, f"{stem}_curves.json")
    with open(json_path) as f:
        curves = json.load(f)
    succ_final = curves["success_mean"][-1] if curves.get("success_mean") else None
    fail_final = curves["fail_mean"][-1] if curves.get("fail_mean") else None
    gap = (succ_final - fail_final) if (succ_final is not None and fail_final is not None) else None
    return {"out_dir": out_dir, "json": json_path, "duration_s": dur,
            "n_success": curves["n_success"], "n_fail": curves["n_fail"],
            "success_final": succ_final, "fail_final": fail_final, "gap": gap}


def make_summary_plot(summary: dict, out_path: str):
    """One bar chart showing gap on held-out vs multi-pos for each cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cells = []
    for task in ("peg", "gear", "nut"):
        for init in ("scratch", "at2ft"):
            row = summary["results"].get(f"{task}_{init}")
            if not row:
                continue
            cells.append((task, init, row))

    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [f"{t}\n{i}" for t, i, _ in cells]
    xs = np.arange(len(cells))
    width = 0.35
    held = [c[2]["evals"].get("heldout", {}).get("gap", 0) or 0 for c in cells]
    multi = [c[2]["evals"].get("multipos", {}).get("gap", 0) if c[2]["evals"].get("multipos") else None
             for c in cells]
    multi_present = [v is not None for v in multi]
    multi_vals = [v if v is not None else 0 for v in multi]

    ax.bar(xs - width / 2, held, width, label="single-pos held-out", color="tab:blue")
    bars_m = ax.bar(xs + width / 2, multi_vals, width, label="multi-pos", color="tab:orange")
    for bar, present in zip(bars_m, multi_present):
        if not present:
            bar.set_color("lightgray")
            bar.set_hatch("//")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("success_final − fail_final")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best")
    ax.set_title("Task × init: success-vs-fail gap across eval splits")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--root",
                    default="/mnt/lab-tank/uber/Tactile-Reward/exp_taskcompare")
    ap.add_argument("--at2_ckpt",
                    default="/mnt/lab-tank/uber/Tactile-Reward/checkpoints_3ch_globalnorm/tactile_rewind_epoch10.pth")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps_per_epoch", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--cap_episodes", type=int, default=5000)
    ap.add_argument("--lr_scratch", type=float, default=1e-4)
    ap.add_argument("--lr_finetune", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tasks", nargs="+", default=["peg", "gear", "nut"])
    ap.add_argument("--inits", nargs="+", default=["scratch", "at2ft"])
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    summary: dict = {"args": vars(args), "results": {}}
    print(f"=== experiment root: {args.root} ===")
    print(f"=== gpu: {args.gpu}, at2_ckpt: {args.at2_ckpt} ===")
    grand_t0 = time.time()

    for task in args.tasks:
        for init in args.inits:
            key = f"{task}_{init}"
            print(f"\n========== {key} ==========")
            train_info = run_train(task, init, args.gpu, args, args.root)
            entry = {"train": train_info, "evals": {}}
            # Eval 1: single-pos held-out (always)
            entry["evals"]["heldout"] = run_eval(
                task, init, "heldout",
                ckpt=train_info["best_ckpt"],
                data_dir=TASKS[task]["single_dir"],
                test_files_json=train_info["test_files_json"],
                gpu=args.gpu, root=args.root,
            )
            # Eval 2: multi-pos (if available)
            if TASKS[task]["multi_dir"]:
                entry["evals"]["multipos"] = run_eval(
                    task, init, "multipos",
                    ckpt=train_info["best_ckpt"],
                    data_dir=TASKS[task]["multi_dir"],
                    test_files_json=None,
                    gpu=args.gpu, root=args.root,
                )
            summary["results"][key] = entry

    summary["total_duration_s"] = time.time() - grand_t0
    summary_path = os.path.join(args.root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    plot_path = os.path.join(args.root, "summary.png")
    make_summary_plot(summary, plot_path)

    # Pretty table
    print("\n=== SUMMARY ===")
    print(f"{'cell':<14} {'best_ep':>7} {'train_gap':>9} {'held_succ':>9} "
          f"{'held_fail':>9} {'held_gap':>9} {'multi_succ':>10} {'multi_fail':>10} {'multi_gap':>9}")
    for key, e in summary["results"].items():
        h = e["evals"]["heldout"]
        m = e["evals"].get("multipos")
        held_succ = f"{h['success_final']:.3f}" if h.get("success_final") is not None else "—"
        held_fail = f"{h['fail_final']:.3f}" if h.get("fail_final") is not None else "—"
        held_gap = f"{h['gap']:+.3f}" if h.get("gap") is not None else "—"
        if m:
            multi_succ = f"{m['success_final']:.3f}" if m.get("success_final") is not None else "—"
            multi_fail = f"{m['fail_final']:.3f}" if m.get("fail_final") is not None else "—"
            multi_gap = f"{m['gap']:+.3f}" if m.get("gap") is not None else "—"
        else:
            multi_succ = multi_fail = multi_gap = "—"
        print(f"{key:<14} ep{e['train']['best_epoch']:>4} "
              f"{e['train']['best_gap_during_training']:>+9.3f} "
              f"{held_succ:>9} {held_fail:>9} {held_gap:>9} "
              f"{multi_succ:>10} {multi_fail:>10} {multi_gap:>9}")
    print(f"\ntotal duration: {summary['total_duration_s']:.0f}s")
    print(f"summary JSON: {summary_path}")
    print(f"summary PNG:  {plot_path}")


if __name__ == "__main__":
    main()
