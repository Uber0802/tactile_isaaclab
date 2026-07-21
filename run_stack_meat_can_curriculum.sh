#!/bin/bash
# Curriculum rollout: load each baseline-A snapshot, briefly run in single_pos
# to collect tactile trajectories at that skill level, save to a per-ckpt subdir.

set -e

CKPTS_DIR=/home/kim/ml/tactile-irl/tactile_isaaclab/logs/rl_games/franka_stack_potted_meat_can/2026-06-18_00-25-01/nn/
BASE_SAVE_DIR=./tactile_dataset/stack_meat_can/new_robot_v3_multipos

# Uniformly sample checkpoints across the episode range from the whole given directory.
NUM_SAMPLES=48

CKPTS=($(python3 -c "
import os, re
ckpts_dir = os.path.expanduser('$CKPTS_DIR')
num_samples = int('$NUM_SAMPLES')
files = []
if os.path.isdir(ckpts_dir):
    for f in os.listdir(ckpts_dir):
        if f.endswith('.pth'):
            match = re.search(r'ep_(\d+)', f)
            if match:
                files.append((int(match.group(1)), f))
files.sort()
if files:
    n = min(num_samples, len(files))
    if n <= 1:
        sampled = files
    else:
        indices = [int(round(i * (len(files) - 1) / (n - 1))) for i in range(n)]
        seen = set()
        sampled = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                sampled.append(files[idx])
    for _, f in sampled:
        print(f)
"))

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "Error: No checkpoints found matching 'ep_[0-9]+' in $CKPTS_DIR"
    exit 1
fi

if [ "$1" = "--dry-run" ] || [ "$1" = "-d" ]; then
    echo "Dry run. Selected $NUM_SAMPLES checkpoints to run:"
    for ckpt_name in "${CKPTS[@]}"; do
        echo "  $ckpt_name"
    done
    exit 0
fi



# ITERS_PER_CKPT = EPISODES_PER_ENV * 2 ensures each env completes at least EPISODES_PER_ENV episodes per checkpoint
EPISODES_PER_ENV=1
ITERS_PER_CKPT=2

for ckpt_name in "${CKPTS[@]}"; do
    ckpt_path="$CKPTS_DIR/$ckpt_name"
    if [ ! -f "$ckpt_path" ]; then
        echo "[skip] missing ckpt: $ckpt_path"
        continue
    fi
    echo $ckpt_path
    # Label = ep_NNN portion of the filename, e.g. ep_300
    label=$(echo "$ckpt_name" | grep -oE 'ep_[0-9]+')
    ckpt_epoch=${label#ep_}
    # rl_games loads the ckpt's epoch counter on restore, so `--max_iterations N`
    # (which sets `max_epochs = N`, absolute) would exit immediately if N <=
    # ckpt_epoch. Compute the absolute target so we actually run ITERS_PER_CKPT
    # PPO iters relative to the loaded ckpt.
    max_iters_abs=$((ckpt_epoch + ITERS_PER_CKPT))
    save_dir="$BASE_SAVE_DIR/$label"
    mkdir -p "$save_dir"
    echo "==== [$label] ckpt=$ckpt_name → $save_dir  (max_epochs=$max_iters_abs) ===="

    FORGE_FIXED_OBJECT_POS=0 \
    FORGE_SAVE_TACTILE_FORCE_FIELD=1 \
    FORGE_SAVE_TACTILE_ALL_ENVS=1 \
    FORGE_ENABLE_FRONT_CAM=0 \
    FORGE_ENABLE_SENSOR=1 \
    FORGE_TACTILE_SAVE_DIR="$save_dir" \
    FORGE_TACTILE_EPISODES_PER_ENV=$EPISODES_PER_ENV \
    FORGE_TACTILE_REWARD_INSTRUCTION="grasp the meat can and stack it on the red box" \
    ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
        --task Isaac-Stack-Cube-Franka-Gelsight-v0 \
        --checkpoint "$ckpt_path" \
        --num_envs 32 \
        --max_iterations $max_iters_abs \
        --enable_cameras \
        --headless \
        agent.params.config.learning_rate=0 \
        agent.params.config.lr_schedule=fixed \
        agent.params.config.save_frequency=999999 \
        agent.params.config.save_best_after=999999 \
        +agent.params.config.full_experiment_name=Stack_box_curriculum_rollout_tmp
done

echo "==== done. dataset root: $BASE_SAVE_DIR ===="
