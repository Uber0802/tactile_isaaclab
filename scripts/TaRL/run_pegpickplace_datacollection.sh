#!/bin/bash
# Peg pickplace paired data collection — load the best baseline ckpt, roll out
# one trajectory per env, save tactile force field + RGB front camera. Two
# passes: single_pos (deterministic init) and baseline (full randomization,
# saved under multipos/). Output: $BASE_SAVE_DIR/{single_pos,multipos}/ep_NNN/
set -e
source "$(dirname "$0")/_common.sh"

# Training checkpoint
CKPTS_DIR=/mnt/home/tactile/tactile_isaaclab/logs/rl_games/ForgePickPlace/PegInsert_PickPlace_baselineA_legacy/nn
# Directory to save
BASE_SAVE_DIR=/mnt/tank/tactile/tactile_dataset/pegpickplace_paired
BEST_EP=2700               # best peg baseline snapshot (r4tddjlv, ~86% success)

# Date-filter to r4tddjlv's ckpt — nn/ also holds older same-epoch ckpts.
CKPT_NAME=$(find "$CKPTS_DIR" -name "last_ForgePickPlace_ep_${BEST_EP}_rew_*.pth" \
    -newermt "2026-05-22 11:30" ! -newermt "2026-05-24 03:00" 2>/dev/null | head -1)
[ -z "$CKPT_NAME" ] && { echo "[fatal] r4tddjlv ep_${BEST_EP} ckpt not found in $CKPTS_DIR"; exit 1; }
CKPT_PATH="$CKPTS_DIR/$(basename "$CKPT_NAME")"
LABEL="ep_${BEST_EP}"

NUM_ENVS=64                # 64 trajectories per pass (per-env quota = 1)
SIGMA=0.3
MAX_ITERS_ABS=$((BEST_EP + 5))   # +5 = safety upper bound; early-exits when saved

for BASELINE in single_pos baseline; do
    [ "$BASELINE" = "baseline" ] && SUBDIR="multipos" || SUBDIR="$BASELINE"
    SAVE_DIR="$BASE_SAVE_DIR/$SUBDIR/$LABEL"
    mkdir -p "$SAVE_DIR"
    echo "==== [$SUBDIR / $LABEL] baseline=$BASELINE → $SAVE_DIR ===="

    FORGE_ENABLE_FRONT_CAM=1 \
    ./isaaclab.sh -p "$TRAIN" \
    "env.tactile_save.force_field=true" \
    "env.tactile_save.all_envs=true" \
    "env.tactile_save.camera=true" \
    "env.tactile_save.episodes_per_env=1" \
    "env.tactile_save.save_dir=$SAVE_DIR" \
        --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
        --baseline "$BASELINE" \
        --checkpoint "$CKPT_PATH" \
        --num_envs $NUM_ENVS --max_iterations $MAX_ITERS_ABS --sigma $SIGMA \
        --headless --enable_cameras \
        agent.params.config.learning_rate=0 \
        agent.params.config.lr_schedule=fixed \
        agent.params.config.save_frequency=999999 \
        agent.params.config.save_best_after=999999 \
        agent.params.config.full_experiment_name=PegInsert_PickPlace_datacollection_tmp
done

echo "==== done. dataset root: $BASE_SAVE_DIR ===="
