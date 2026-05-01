"""Build a small metadata H5 from AnyTouch2 tactile npy files.

Each AnyTouch2 file is `{task_name}__{traj_idx}.npy` of shape (N, 320, 480, 2)
in float16 (left+right hand shear forces concat along width). The raw npy
files stay where they are; we only write a tiny metadata H5 containing
language embeddings and trajectory file lists per task.

Default split is **within-task held-out**: every task appears in BOTH the
train and eval H5s, but `--eval_per_task N` trajectories per task are held
out for eval. This is a sanity-check split — it tests whether the model
generalises to unseen rollouts of *the same* task. For the harder
held-out-task language-generalisation split, pass `--held_out_tasks A B C`.

Output H5 schema (per task group):
  minilm_lang_embedding : float32 [num_instructions, 384]
  instructions          : utf-8 string [num_instructions]
  trajectory_files      : utf-8 string [num_trajectories]   (filenames only)
Top-level attribute:
  data_dir              : str  (root dir containing the npy files)
"""
import os
import re
import json
import random
import argparse
from collections import defaultdict

import h5py
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def humanize(task_name: str) -> str:
    return task_name.replace("_", " ").replace("'", "").strip()


def discover_tasks(data_dir: str):
    pattern = re.compile(r"^(.+)__(\d+)\.npy$")
    task_to_files = defaultdict(list)
    for f in sorted(os.listdir(data_dir)):
        m = pattern.match(f)
        if m:
            task_to_files[m.group(1)].append(f)
    return {t: sorted(fs) for t, fs in task_to_files.items()}


def encode_minilm(texts, tokenizer, model, device):
    with torch.no_grad():
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model(**enc)
        emb = mean_pooling(out, enc["attention_mask"]).cpu().numpy()
    return emb.astype(np.float32)


def write_split_h5(out_path: str, task_to_traj: dict, instructions_map: dict,
                   data_dir: str, tokenizer, model, device):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as h5:
        h5.attrs["data_dir"] = data_dir
        for task in tqdm(sorted(task_to_traj.keys()),
                         desc=os.path.basename(out_path)):
            files = task_to_traj[task]
            if not files:
                continue
            grp = h5.create_group(task)
            instructions = instructions_map.get(task, [humanize(task)])
            emb = encode_minilm(instructions, tokenizer, model, device)
            grp.create_dataset("minilm_lang_embedding", data=emb)
            grp.create_dataset(
                "instructions",
                data=np.array(instructions, dtype=object),
                dtype=str_dt,
            )
            grp.create_dataset(
                "trajectory_files",
                data=np.array(files, dtype=object),
                dtype=str_dt,
            )
    print(f"wrote {out_path} "
          f"({sum(1 for v in task_to_traj.values() if v)} tasks, "
          f"{sum(len(v) for v in task_to_traj.values())} trajectories)")


def split_within_task(all_tasks: dict, eval_per_task: int, seed: int):
    """Per task, hold out `eval_per_task` trajectories for eval.

    Both train and eval contain ALL tasks. Selection is deterministic
    given `seed` so re-running the conversion is reproducible.
    """
    rng = random.Random(seed)
    train, eval_ = {}, {}
    for task in sorted(all_tasks.keys()):
        files = list(all_tasks[task])
        if len(files) <= 1:
            print(f"WARNING: task {task!r} has {len(files)} traj; "
                  f"using it for train only (no eval).")
            n_eval = 0
        elif len(files) <= eval_per_task:
            print(f"WARNING: task {task!r} has {len(files)} traj <= eval_per_task; "
                  f"holding out {len(files) - 1} for eval, keeping 1 for train.")
            n_eval = len(files) - 1
        else:
            n_eval = eval_per_task
        rng.shuffle(files)
        eval_files = sorted(files[:n_eval])
        train_files = sorted(files[n_eval:])
        if train_files:
            train[task] = train_files
        if eval_files:
            eval_[task] = eval_files
    return train, eval_


def split_held_out_tasks(all_tasks: dict, held_out: set):
    train = {t: list(v) for t, v in all_tasks.items() if t not in held_out}
    eval_ = {t: list(v) for t, v in all_tasks.items() if t in held_out}
    return train, eval_


def main(args):
    all_tasks = discover_tasks(args.data_dir)
    print(f"discovered {len(all_tasks)} tasks, "
          f"{sum(len(v) for v in all_tasks.values())} trajectories")

    instructions_map = {}
    if args.instructions_json:
        with open(args.instructions_json) as fh:
            instructions_map = json.load(fh)
        print(f"loaded paraphrases for {len(instructions_map)} tasks "
              f"from {args.instructions_json}")

    if args.held_out_tasks:
        held = set(args.held_out_tasks) & set(all_tasks)
        train_tasks, eval_tasks = split_held_out_tasks(all_tasks, held)
        print(f"split mode: held-out tasks  "
              f"(train={len(train_tasks)}, eval={len(eval_tasks)})")
    else:
        train_tasks, eval_tasks = split_within_task(
            all_tasks, args.eval_per_task, args.seed,
        )
        print(f"split mode: within-task held-out "
              f"({args.eval_per_task} traj/task held out for eval)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(device)
    model.eval()

    write_split_h5(
        os.path.join(args.output_dir, "tactile_metadata_train.h5"),
        train_tasks, instructions_map, args.data_dir, tokenizer, model, device,
    )
    write_split_h5(
        os.path.join(args.output_dir, "tactile_metadata_eval.h5"),
        eval_tasks, instructions_map, args.data_dir, tokenizer, model, device,
    )

    summary = {
        "split_mode": "held_out_tasks" if args.held_out_tasks else "within_task",
        "eval_per_task": None if args.held_out_tasks else args.eval_per_task,
        "seed": args.seed,
        "train_tasks": {t: len(v) for t, v in sorted(train_tasks.items())},
        "eval_tasks": {t: len(v) for t, v in sorted(eval_tasks.items())},
    }
    summary_path = os.path.join(args.output_dir, "task_split.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote task split summary to {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="/mnt/tank/uber/AnyTouch2/tactile_dataset",
                    help="Directory containing AnyTouch2 *.npy trajectory files.")
    ap.add_argument("--output_dir",
                    default="/mnt/tank/uber/Tactile-Reward",
                    help="Where to write metadata H5 + task split.")
    ap.add_argument("--instructions_json", default=None,
                    help="JSON {task_name: [instr, ...]} (e.g. from "
                         "generate_instructions_via_llm.py).")
    ap.add_argument("--eval_per_task", type=int, default=1,
                    help="Within-task split: trajectories per task held out for eval.")
    ap.add_argument("--held_out_tasks", nargs="*", default=None,
                    help="Switch to held-out-task split with these task names.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the within-task selection.")
    args = ap.parse_args()
    main(args)
