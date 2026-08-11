#!/bin/bash
# Train the ReWiND visual (RGB) reward model on the gear seed2 single_pos rollouts.
#
# Two stages, one script:
#   1. precompute_dino_features.py — encode every `*_camera.npy` with the frozen
#      DINOv2 ViT-B/14 into a (T,768) fp16 cache. ~2 episodes/s, so ~7 min for
#      832 episodes; ~380 MB of cache from 37 GB of RGB. Already-cached episodes
#      are skipped, so a re-run is nearly free.
#   2. train_visual_taRL.py — train ReWiNDTransformer on the cached features.
#      Same training signal as the tactile trainer (forward/rewind sampling,
#      success/fail 50-50, failures labelled progress 0, held-out gap eval).
#
# The resulting .pth is a drop-in FORGE_VISUAL_REWARD_CKPT — same state_dict
# layout as the existing /mnt/tank/uber/Tactile-Reward/ckpt_visual/*.pth.
#
# ---------------------------------------------------------------------------
# Two caveats specific to the single_pos gear dataset
# ---------------------------------------------------------------------------
#  1. Label balance: 832 episodes, 658 success / 174 fail (79%), and every
#     failure comes from ep_20 / ep_200 / ep_400 — the plateau snapshots are
#     97-100% success because the fixed pose is the easiest one seed2 ever
#     solved. The multipos twin is the better-balanced set; consider training
#     on both (share one CACHE_DIR, then SKIP_PRECOMPUTE=1).
#  2. Success criterion: this data was collected under `--baseline single_pos`,
#     which leaves gear's success_threshold at the frozen-A default +0.05
#     ("gear hovering 2.5mm above the shaft top counts"), NOT the 0.0 that
#     A_hard_success_yaw01 — the baseline seed2 trained under — uses. The
#     multipos rollouts used 0.0, so the two sets' Success labels are not
#     strictly comparable. Mild for gear (2.5mm on a 50mm shaft); it is the
#     reason the plateau reads ~100% here vs ~55% on the training curve.
# ---------------------------------------------------------------------------
#
# Usage:
#   GPU_ID=5 ./run_gearpickplace_train_visual_reward_seed2_singlepos.sh
#   SKIP_PRECOMPUTE=1 EPOCHS=30 ./...     # features already cached
#   DATA_DIR=... CACHE_DIR=... CKPT_DIR=... ./...   # retarget to nut / peg
#
# NOTE: TORCH_HOME is deliberately NOT overridden here (unlike the Isaac run
# scripts). torch.hub resolves the DINOv2 weights from ~/.cache/torch/hub, where
# they are already downloaded; pointing it at a per-machine scratch dir would
# force a fresh download.

set -e

REPO=/mnt/home/tactile/tactile_isaaclab
TAREWIND=$REPO/external/third-party/Tactile-ReWiND
PYTHON=/mnt/home/tactile/miniconda3/envs/tactile_isaaclab/bin/python

DATA_DIR=${DATA_DIR:-/mnt/scratch/tactile/gearpickplace_curriculum_seed2_paired}
CACHE_DIR=${CACHE_DIR:-/mnt/scratch/tactile/dino_cache/gear_seed2_singlepos}
CKPT_DIR=${CKPT_DIR:-/mnt/lab-tank/tactile/Tactile-Reward/ckpt_visual/gear_seed2_singlepos}
RUN_NAME=${RUN_NAME:-gear_rgb_seed2_singlepos}

# The FIRST paraphrase is what eval uses and what you should later pass as
# FORGE_VISUAL_REWARD_INSTRUCTION. The rest are language-robustness augmentation
# (same five strings the tactile gear model was trained on).
TASK_TEXTS=(
    "pick up the gear and mesh it onto the shaft"
    "grasp the gear and slide it down onto the gear shaft"
    "use the gripper to pick the gear and engage it on the shaft"
    "grab the gear and place it onto the meshing shaft"
    "lift the gear and mesh it onto the rotating shaft"
)

# Defaults mirror the args stored inside the shipped peg_rgb_multipos.pth.
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-128}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-100}
LR=${LR:-8e-4}
REWIND_RATIO=${REWIND_RATIO:-0.8}
SUCCESS_PROB=${SUCCESS_PROB:-0.5}

if [ -n "${GPU_ID:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "[info] pinned to GPU $GPU_ID"
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "[fatal] dataset not found: $DATA_DIR"; exit 1
fi

cd "$TAREWIND"

# ---------------------------------------------------------------------------
# Stage 1 — DINOv2 feature cache
# ---------------------------------------------------------------------------
if [ "${SKIP_PRECOMPUTE:-0}" = "1" ]; then
    echo "==== [1/2] SKIP_PRECOMPUTE=1 — reusing $CACHE_DIR ===="
else
    n_cam=$(find "$DATA_DIR" -name "*_camera.npy" | wc -l)
    echo "==== [1/2] DINOv2 features: $n_cam camera episodes -> $CACHE_DIR ===="
    $PYTHON scripts/precompute_dino_features.py \
        --data_dirs "$DATA_DIR" \
        --cache_dir "$CACHE_DIR" \
        --backbone dinov2_vitb14 \
        --frame_batch 64 \
        --amp bf16
fi

n_feat=$(find "$CACHE_DIR" -name "*.npz" 2>/dev/null | wc -l)
if [ "$n_feat" -eq 0 ]; then
    echo "[fatal] no cached features under $CACHE_DIR"; exit 1
fi
echo "[info] $n_feat cached episodes ready"

# ---------------------------------------------------------------------------
# Stage 2 — train the ReWiND reward head on the cached features
# ---------------------------------------------------------------------------
mkdir -p "$CKPT_DIR"
echo "==== [2/2] training $RUN_NAME -> $CKPT_DIR ===="
$PYTHON scripts/train_visual_taRL.py \
    --feat_dirs "$CACHE_DIR" \
    --ckpt_dir "$CKPT_DIR" \
    --run_name "$RUN_NAME" \
    --task_texts "${TASK_TEXTS[@]}" \
    --rewind_root "$REPO/external/third-party/ReWiND" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --steps_per_epoch "$STEPS_PER_EPOCH" \
    --lr "$LR" \
    --rewind_ratio "$REWIND_RATIO" \
    --success_prob "$SUCCESS_PROB" \
    --test_ratio 0.1 \
    --test_eval_every 1 \
    --amp bf16 \
    --seed 42

echo
echo "==== done ===="
echo "Pick the epoch with the best test_gap (NOT the lowest train_loss):"
echo "  $PYTHON -c \"import json;h=json.load(open('$CKPT_DIR/history.json'));\\"
echo "    print(max(h,key=lambda r:r.get('test_gap',-9)))\""
echo
echo "Then wire it into RL:"
echo "  FORGE_ENABLE_FRONT_CAM=1 \\"
echo "  FORGE_VISUAL_REWARD_CKPT=$CKPT_DIR/${RUN_NAME}_epochN.pth \\"
echo "  FORGE_VISUAL_REWARD_SCALE=1.0 \\"
echo "  FORGE_VISUAL_REWARD_INSTRUCTION=\"${TASK_TEXTS[0]}\" \\"
echo "  FORGE_VISUAL_REWARD_ROOT=$REPO/external/third-party/ReWiND \\"
echo "  FORGE_VISUAL_REWARD_BACKBONE=dinov2_vitb14  ... --enable_cameras"
