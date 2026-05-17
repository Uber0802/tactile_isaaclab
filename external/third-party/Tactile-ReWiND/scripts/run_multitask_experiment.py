"""Run the multi-task complement to run_taskcompare_experiment.py.

Two cells:
  1. multitask_scratch:  one fresh model trained on peg + gear + nut
  2. multitask_at2ft:    same data, init from AnyTouch2 globalnorm epoch10

Each cell is evaluated per-task on (single-pos held-out, multi-pos when avail)
with the correct task instruction. Summary table compares multi-task results
against the single-task baselines saved in <baseline_root>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

PY = "/mnt/tank/uber/conda-envs/tactile_isaaclab/bin/python"
REPO = "/mnt/home/uber/tactile_isaaclab/external/third-party/Tactile-ReWiND"

TASK_SPECS = {
    "peg": {
        "data_dirs": ["/mnt/tank/tactile/tactile_dataset/data_3_baseline"],
        "task_texts": [
            "grasp peg and insert to another hole",
            "pick up the peg and insert it into the second hole",
            "use the gripper to grab the peg and place it in the other hole",
            "grasp the cylindrical peg and slot it into the adjacent hole",
            "transfer the peg from one hole to another",
        ],
        "max_episodes": 0,
        "multi_dir": None,
    },
    "gear": {
        "data_dirs": ["/mnt/tank/tactile/tactile_dataset/gearpickplace_curriculum"],
        "task_texts": [
            "pick up the gear and mesh it onto the shaft",
            "grasp the gear and slide it down onto the gear shaft",
            "use the gripper to pick the gear and engage it on the shaft",
            "grab the gear and place it onto the meshing shaft",
            "lift the gear and mesh it onto the rotating shaft",
        ],
        "max_episodes": 5000,
        "multi_dir": "/mnt/tank/tactile/tactile_dataset/gearpickplace_baselineA copy",
    },
    "nut": {
        "data_dirs": ["/mnt/tank/tactile/tactile_dataset/nutpickplace_curriculum"],
        "task_texts": [
            "pick up the nut and thread it onto the bolt",
            "grasp the nut and screw it onto the threaded shaft",
            "use the gripper to pick the nut and thread it onto the rod",
            "grab the nut and thread it onto the bolt",
            "lift the nut and thread it onto the screw",
        ],
        "max_episodes": 5000,
        "multi_dir": "/mnt/tank/tactile/tactile_dataset/nutpickplace_baselineA copy",
    },
}


def sh(cmd: list, log_path: str, env: dict):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.time()
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
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return dur


def best_epoch_by_avg_gap(history_path: str) -> tuple[int, float, dict]:
    with open(history_path) as f:
        hist = json.load(f)
    rows = [(h["epoch"], h["test"]["avg_gap"], h["test"])
            for h in hist if "test" in h]
    if not rows:
        raise RuntimeError(f"no test entries in {history_path}")
    best_ep, best_avg, best_per_task = max(rows, key=lambda r: r[1])
    return best_ep, best_avg, best_per_task


def run_train(init: str, gpu: int, args, root: str, spec_path: str) -> dict:
    run_name = f"multitask_{init}"
    ckpt_dir = os.path.join(root, run_name)
    log_path = os.path.join(root, "logs", f"train_{run_name}.log")
    cmd = [
        PY, "scripts/finetune_multitask.py",
        "--task_specs_json", spec_path,
        "--ckpt_dir", ckpt_dir, "--run_name", run_name,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--steps_per_epoch", str(args.steps_per_epoch),
        "--num_workers", str(args.num_workers),
        "--prefetch_factor", str(args.prefetch_factor),
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
        raise ValueError(init)
    if args.compile:
        cmd += ["--compile"]
    else:
        cmd += ["--no-compile"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"\n[train] {run_name}  gpu={gpu}")
    sh(cmd, log_path, env)
    best_ep, best_avg, best_per_task = best_epoch_by_avg_gap(
        os.path.join(ckpt_dir, "history.json"))
    best_ckpt = os.path.join(ckpt_dir, f"{run_name}_epoch{best_ep}.pth")
    print(f"  best: epoch{best_ep}  avg_gap={best_avg:+.3f}  per-task={best_per_task}")
    return {"ckpt_dir": ckpt_dir, "best_ckpt": best_ckpt, "best_epoch": best_ep,
            "train_avg_gap": best_avg, "train_per_task": best_per_task,
            "test_files_json": {t: os.path.join(ckpt_dir, f"test_files_{t}.json")
                                for t in TASK_SPECS}}


def run_eval(init: str, task: str, split: str, ckpt: str, data_dir: str,
             test_files_json: str | None, task_text: str, gpu: int,
             root: str) -> dict:
    out_dir = os.path.join(root, "eval", f"multitask_{init}_{task}_{split}")
    log_path = os.path.join(root, "logs",
                            f"eval_multitask_{init}_{task}_{split}.log")
    cmd = [
        PY, "scripts/eval_reward_curves_data3.py",
        "--ckpts", ckpt,
        "--data_dirs", data_dir,
        "--task_text_override", task_text,
        "--output_dir", out_dir,
    ]
    if test_files_json:
        cmd += ["--test_files_json", test_files_json]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"\n[eval]  multitask_{init}/{task}/{split}  gpu={gpu}")
    sh(cmd, log_path, env)
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    json_path = os.path.join(out_dir, f"{stem}_curves.json")
    with open(json_path) as f:
        curves = json.load(f)
    succ = curves["success_mean"][-1] if curves.get("success_mean") else None
    fail = curves["fail_mean"][-1] if curves.get("fail_mean") else None
    gap = (succ - fail) if (succ is not None and fail is not None) else None
    return {"n_success": curves["n_success"], "n_fail": curves["n_fail"],
            "success_final": succ, "fail_final": fail, "gap": gap,
            "json": json_path, "out_dir": out_dir}


def make_compare_plot(summary: dict, baseline_summary: dict | None, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cell_labels: list[str] = []
    held_vals: list[float] = []
    multi_vals: list[float] = []
    cell_colors: list[str] = []

    # Baseline single-task cells (from previous experiment)
    if baseline_summary:
        for task in ("peg", "gear", "nut"):
            for init in ("scratch", "at2ft"):
                key = f"{task}_{init}"
                e = baseline_summary["results"].get(key)
                if not e:
                    continue
                cell_labels.append(f"{task}\nsingle/{init}")
                held_vals.append(e["evals"]["heldout"]["gap"] or 0)
                m = e["evals"].get("multipos")
                multi_vals.append((m["gap"] if m and m.get("gap") is not None else None))
                cell_colors.append("tab:gray")
    # Multi-task cells
    for init in ("scratch", "at2ft"):
        per_task = summary["results"].get(init, {}).get("per_task", {})
        for task in ("peg", "gear", "nut"):
            row = per_task.get(task, {})
            cell_labels.append(f"{task}\nmulti/{init}")
            held = row.get("heldout", {}).get("gap")
            mp = row.get("multipos", {}).get("gap") if row.get("multipos") else None
            held_vals.append(held if held is not None else 0)
            multi_vals.append(mp)
            cell_colors.append("tab:blue" if init == "scratch" else "tab:orange")

    n = len(cell_labels)
    xs = np.arange(n)
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(12, n * 0.7), 5))
    multi_present = [v is not None for v in multi_vals]
    multi_plot = [v if v is not None else 0 for v in multi_vals]
    ax.bar(xs - width / 2, held_vals, width, label="single-pos held-out", color="tab:blue")
    bars_m = ax.bar(xs + width / 2, multi_plot, width, label="multi-pos", color="tab:orange")
    for bar, present in zip(bars_m, multi_present):
        if not present:
            bar.set_color("lightgray")
            bar.set_hatch("//")
    ax.set_xticks(xs)
    ax.set_xticklabels(cell_labels, fontsize=8)
    ax.set_ylabel("success_final − fail_final")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best")
    ax.set_title("single-task vs multi-task: success-vs-fail gap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--root",
                    default="/mnt/lab-tank/uber/Tactile-Reward/exp_multitask")
    ap.add_argument("--at2_ckpt",
                    default="/mnt/lab-tank/uber/Tactile-Reward/checkpoints_3ch_globalnorm/tactile_rewind_epoch10.pth")
    ap.add_argument("--baseline_summary",
                    default="/mnt/lab-tank/uber/Tactile-Reward/exp_taskcompare/summary.json",
                    help="single-task baseline summary.json for the comparison plot.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps_per_epoch", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--lr_scratch", type=float, default=1e-4)
    ap.add_argument("--lr_finetune", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--inits", nargs="+", default=["scratch", "at2ft"])
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    # Write the task specs JSON the multi-task trainer reads.
    spec_path = os.path.join(args.root, "task_specs.json")
    spec_for_writer = {t: {k: v for k, v in s.items() if k != "multi_dir"}
                       for t, s in TASK_SPECS.items()}
    with open(spec_path, "w") as f:
        json.dump(spec_for_writer, f, indent=2)
    print(f"=== root: {args.root} ===")
    print(f"=== gpu: {args.gpu}  at2_ckpt: {args.at2_ckpt} ===")
    print(f"=== wrote task spec: {spec_path} ===")

    grand_t0 = time.time()
    summary: dict = {"args": vars(args), "task_spec_path": spec_path, "results": {}}
    for init in args.inits:
        print(f"\n========== multitask_{init} ==========")
        train_info = run_train(init, args.gpu, args, args.root, spec_path)
        per_task: dict = {}
        for task, spec in TASK_SPECS.items():
            task_text = spec["task_texts"][0]
            per_task[task] = {}
            # held-out
            per_task[task]["heldout"] = run_eval(
                init, task, "heldout",
                ckpt=train_info["best_ckpt"],
                data_dir=spec["data_dirs"][0],
                test_files_json=train_info["test_files_json"][task],
                task_text=task_text, gpu=args.gpu, root=args.root,
            )
            # multi-pos (peg has none)
            if spec["multi_dir"]:
                per_task[task]["multipos"] = run_eval(
                    init, task, "multipos",
                    ckpt=train_info["best_ckpt"],
                    data_dir=spec["multi_dir"],
                    test_files_json=None,
                    task_text=task_text, gpu=args.gpu, root=args.root,
                )
        summary["results"][init] = {"train": train_info, "per_task": per_task}

    summary["total_duration_s"] = time.time() - grand_t0
    with open(os.path.join(args.root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    baseline = None
    if args.baseline_summary and os.path.exists(args.baseline_summary):
        with open(args.baseline_summary) as f:
            baseline = json.load(f)
        print(f"loaded baseline summary from {args.baseline_summary}")
    plot_path = os.path.join(args.root, "summary.png")
    make_compare_plot(summary, baseline, plot_path)

    print("\n=== MULTITASK SUMMARY ===")
    print(f"{'cell':<22} {'best_ep':>7} {'train_avg':>10} "
          f"{'held_succ':>9} {'held_fail':>9} {'held_gap':>9} "
          f"{'multi_succ':>10} {'multi_fail':>10} {'multi_gap':>9}")
    for init, e in summary["results"].items():
        for task in ("peg", "gear", "nut"):
            row = e["per_task"].get(task, {})
            h = row.get("heldout", {})
            m = row.get("multipos")
            held_s = f"{h.get('success_final', float('nan')):.3f}"
            held_f = f"{h.get('fail_final', float('nan')):.3f}"
            held_g = f"{h.get('gap', float('nan')):+.3f}" if h.get("gap") is not None else "—"
            if m:
                multi_s = f"{m['success_final']:.3f}"
                multi_f = f"{m['fail_final']:.3f}"
                multi_g = f"{m['gap']:+.3f}"
            else:
                multi_s = multi_f = multi_g = "—"
            print(f"multitask_{init}/{task:<6} ep{e['train']['best_epoch']:>4} "
                  f"{e['train']['train_avg_gap']:>+10.3f} "
                  f"{held_s:>9} {held_f:>9} {held_g:>9} "
                  f"{multi_s:>10} {multi_f:>10} {multi_g:>9}")
    print(f"\ntotal duration: {summary['total_duration_s']:.0f}s")
    print(f"summary JSON: {os.path.join(args.root, 'summary.json')}")
    print(f"summary PNG:  {plot_path}")


if __name__ == "__main__":
    main()
