#!/bin/bash
# Curriculum rollout: load each baseline-A snapshot, briefly run in single_pos
# to collect tactile trajectories at that skill level, save to a per-ckpt subdir.

set -e

CKPTS_DIR=~/ml/tactile-irl/tactile_isaaclab/nn
BASE_SAVE_DIR=./tactile_dataset/stack_box/curriculum_multipos

#Curriculum spectrum: sample snapshots across the whole skill range.
#Set CKPT_STRIDE=1 to roll out EVERY checkpoint (77 snapshots),
#or CKPT_STRIDE=4 for a quicker sparse rollout (~20 snapshots).
CKPTS=(
    "last_franka_stack_ep_100_rew_6.677147.pth"
    "last_franka_stack_ep_400_rew_7.2512236.pth"
    "last_franka_stack_ep_700_rew_7.757937.pth"
    "last_franka_stack_ep_1000_rew_8.924516.pth"
    "last_franka_stack_ep_1300_rew_11.001369.pth"
    "last_franka_stack_ep_1600_rew_12.429459.pth"
    "last_franka_stack_ep_1900_rew_13.496348.pth"
    "last_franka_stack_ep_2200_rew_14.337699.pth"
    "last_franka_stack_ep_2500_rew_17.27967.pth"
    "last_franka_stack_ep_2800_rew_19.043896.pth"
    "last_franka_stack_ep_3100_rew_22.553942.pth"
    "last_franka_stack_ep_3400_rew_25.260508.pth"
    "last_franka_stack_ep_3700_rew_28.634111.pth"
    "last_franka_stack_ep_4000_rew_29.927746.pth"
    "last_franka_stack_ep_4300_rew_37.87523.pth"
    "last_franka_stack_ep_4600_rew_39.98565.pth"
    "last_franka_stack_ep_4900_rew_51.786015.pth"
    "last_franka_stack_ep_5200_rew_52.69835.pth"
    "last_franka_stack_ep_5500_rew_50.053093.pth"
    "last_franka_stack_ep_5700_rew_73.564156.pth"
)


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
    FORGE_ENABLE_FRONT_CAM=1 \
    FORGE_TACTILE_SAVE_DIR="$save_dir" \
    FORGE_TACTILE_EPISODES_PER_ENV=$EPISODES_PER_ENV \
    FORGE_TACTILE_REWARD_INSTRUCTION="grasp the blue box and stack it on the red box" \
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
