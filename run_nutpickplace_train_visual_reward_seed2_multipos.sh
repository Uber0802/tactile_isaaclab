#!/bin/bash
# Train the ReWiND visual (RGB) reward model on the nut seed2 multipos rollouts.
#
# Two stages, one script:
#   1. precompute_dino_features.py — encode every `*_camera.npy` with the frozen
#      DINOv2 ViT-B/14 into a (T,768) fp16 cache. Nut episodes are T=449
#      (episode_length_s=30) vs gear's 299, so ~1.3 episodes/s -> ~8 min for the
#      640 episodes, producing ~440 MB of cache from 42 GB of RGB.
#      Already-cached episodes are skipped, so a re-run is nearly free.
#   2. train_visual_taRL.py — train ReWiNDTransformer on the cached features.
#      Same training signal as the tactile trainer (forward/rewind sampling,
#      success/fail 50-50, failures labelled progress 0, held-out gap eval).
#
# The resulting .pth is a drop-in FORGE_VISUAL_REWARD_CKPT — same state_dict
# layout as the existing /mnt/tank/uber/Tactile-Reward/ckpt_visual/*.pth.
#
# Usage:
#   GPU_ID=5 ./run_nutpickplace_train_visual_reward_seed2_multipos.sh
#   SKIP_PRECOMPUTE=1 EPOCHS=30 ./...     # features already cached
#   DATA_DIR=... CACHE_DIR=... CKPT_DIR=... ./...   # retarget to another dataset
#
# To also fold in the single_pos rollouts once that collection finishes, point
# both stages at the two roots: run this script once per root with a shared
# CACHE_DIR, then re-run with SKIP_PRECOMPUTE=1 (train_visual_taRL.py globs the
# whole cache dir recursively).
#
# NOTE: TORCH_HOME is deliberately NOT overridden here (unlike the Isaac run
# scripts). torch.hub resolves the DINOv2 weights from ~/.cache/torch/hub, where
# they are already downloaded; pointing it at a per-machine scratch dir would
# force a fresh download.

set -e

REPO=/mnt/home/tactile/tactile_isaaclab
TAREWIND=$REPO/external/third-party/Tactile-ReWiND
PYTHON=/mnt/home/tactile/miniconda3/envs/tactile_isaaclab/bin/python

DATA_DIR=${DATA_DIR:-/mnt/scratch/tactile/nutpickplace_curriculum_seed2_paired_multipos}
CACHE_DIR=${CACHE_DIR:-/mnt/scratch/tactile/dino_cache/nut_seed2_multipos}
CKPT_DIR=${CKPT_DIR:-/mnt/lab-tank/tactile/Tactile-Reward/ckpt_visual/nut_seed2_multipos}
RUN_NAME=${RUN_NAME:-nut_rgb_seed2_multipos}

# The FIRST paraphrase is what eval uses and what you should later pass as
# FORGE_VISUAL_REWARD_INSTRUCTION. The rest are language-robustness augmentation
# (same five strings the tactile nut model was trained on — see
# Tactile-ReWiND/scripts/run_multitask_experiment.py TASK_SPECS["nut"]).
TASK_TEXTS=(
    "pick up the nut and thread it onto the bolt"
    "grasp the nut and screw it onto the threaded shaft"
    "use the gripper to pick the nut and thread it onto the rod"
    "grab the nut and thread it onto the bolt"
    "lift the nut and thread it onto the screw"
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
