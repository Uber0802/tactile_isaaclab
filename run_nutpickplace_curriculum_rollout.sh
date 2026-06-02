#!/bin/bash
# Curriculum rollout for nut pickplace — best-ckpt only, one trajectory per
# env, BOTH tactile force field AND RGB front camera saved. Runs twice:
#   1. single_pos baseline → deterministic init pose
#   2. A baseline           → full randomization (multipos)
# Same ckpt, same num_envs, same per-env quota. Output structure:
#
#   pegpickplace_paired/
#     single_pos/ep_NNN/
#       ep<N>_env<XXX>.npy          ← tactile (T, 40, 25, 3) float16
#       ep<N>_env<XXX>_camera.npy   ← RGB     (T, 224, 224, 3) uint8
#     multipos/ep_NNN/
#       (same files)
#
# Tactile sensors stay ON (we want tactile force fields). Front camera also on.

set -e

CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

CKPTS_DIR=/mnt/home/tactile/tactile_isaaclab/logs/rl_games/ForgeNutPickPlace/NutThread_PickPlace_baselineA/nn
BASE_SAVE_DIR=/mnt/tank/tactile/tactile_dataset/nutpickplace_paired

# Use the BEST nut baselineA snapshot (ep_260, ~57% success).
BEST_EP=260
CKPT_NAME=$(find "$CKPTS_DIR" -name "last_ForgeNutPickPlace_ep_${BEST_EP}_rew_*.pth" 2>/dev/null | head -1)
if [ -z "$CKPT_NAME" ]; then
    echo "[fatal] ep_${BEST_EP} ckpt not found in $CKPTS_DIR"; exit 1
fi
CKPT_NAME=$(basename "$CKPT_NAME")
CKPT_PATH="$CKPTS_DIR/$CKPT_NAME"
LABEL="ep_${BEST_EP}"
CKPT_EPOCH=$BEST_EP

NUM_ENVS=64                # → 64 trajectories per pass (per-env quota = 1)
ITERS_PER_CKPT=5           # safety upper bound; early-exits when all envs saved
SIGMA=0.3
MAX_ITERS_ABS=$((CKPT_EPOCH + ITERS_PER_CKPT))

# Run twice — one per init-pose mode.
for BASELINE in single_pos A; do
    if [ "$BASELINE" = "A" ]; then
        SUBDIR="multipos"
    else
        SUBDIR="$BASELINE"
    fi
    SAVE_DIR="$BASE_SAVE_DIR/$SUBDIR/$LABEL"
    mkdir -p "$SAVE_DIR"
    echo "==== [$SUBDIR / $LABEL] baseline=$BASELINE → $SAVE_DIR ===="

    TMPDIR="$CACHE_DIR/tmp" \
    XDG_CACHE_HOME="$CACHE_DIR/cache" \
    OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
    OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
    TORCH_HOME="$CACHE_DIR/torch" \
    TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
    TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
    FORGE_SAVE_TACTILE_FORCE_FIELD=1 \
    FORGE_SAVE_TACTILE_ALL_ENVS=1 \
    FORGE_SAVE_CAMERA=1 \
    FORGE_ENABLE_FRONT_CAM=1 \
    FORGE_TACTILE_EPISODES_PER_ENV=1 \
    FORGE_TACTILE_SAVE_DIR="$SAVE_DIR" \
    ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
        --task Isaac-Forge-NutThread-PickPlace-Direct-v0 \
        --baseline "$BASELINE" \
        --checkpoint "$CKPT_PATH" \
        --num_envs $NUM_ENVS \
        --max_iterations $MAX_ITERS_ABS \
        --sigma $SIGMA \
        --headless \
        --enable_cameras \
        agent.params.config.learning_rate=0 \
        agent.params.config.lr_schedule=fixed \
        agent.params.config.save_frequency=999999 \
        agent.params.config.save_best_after=999999 \
        agent.params.config.full_experiment_name=NutThread_PickPlace_curriculum_rollout_tmp
done

echo "==== done. dataset root: $BASE_SAVE_DIR ===="
