#!/bin/bash
# Curriculum rollout: load each baseline-A snapshot, briefly run in single_pos
# to collect tactile trajectories at that skill level, save to a per-ckpt subdir.
# Output: /mnt/tank/tactile/tactile_dataset/pegpickplace_curriculum/<label>/ep*.npy
# Init pose is fixed (baseline single_pos), so tactile variation across the
# combined dataset comes purely from policy-skill progression.

set -e

# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

CKPTS_DIR=/mnt/home/tactile/tactile_isaaclab/logs/rl_games/ForgePickPlace/PegInsert_PickPlace_baselineA_legacy/nn
BASE_SAVE_DIR=/mnt/tank/tactile/tactile_dataset/pegpickplace_curriculum_rgb

# Explicit ep list picked across the r4tddjlv (2026-05-22 ~ 05-24) skill curve:
#   ep_100..1000  = pre-crack (0% success)
#   ep_1100..1500 = crack region (5-50%)
#   ep_1600..2100 = consolidation (50-74%)
#   ep_2400..2700 = peak region (74-91%)
# This dir mixes two runs (May 20-21 vhhj6i3c + May 22-24 r4tddjlv) — the date
# filter `-newermt 2026-05-22 11:30` picks only r4tddjlv's snapshots so we get
# a consistent skill trajectory rather than mixed-policy ckpts.
TARGET_EPS=(100 300 500 700 1000 1100 1500 1600 1900 2100 2400 2700)
CKPTS=()
for ep in "${TARGET_EPS[@]}"; do
    match=$(find "$CKPTS_DIR" -name "last_ForgePickPlace_ep_${ep}_rew_*.pth" \
        -newermt "2026-05-22 11:30" ! -newermt "2026-05-24 03:00" 2>/dev/null | head -1)
    if [ -n "$match" ]; then
        CKPTS+=("$(basename "$match")")
    else
        echo "[warn] missing r4tddjlv ep_$ep — skipping"
    fi
done
echo "Selected ${#CKPTS[@]} ckpts: ${CKPTS[*]}"

# Per-ckpt rollout budget. One PPO iter ≈ horizon_length (256) × num_envs steps.
# At 20 s episode length and 15 Hz control, ~300 steps per episode, so 20 iters
# × 256 horizon × 128 envs ≈ ~2.7k episodes per ckpt. Tune to taste.
ITERS_PER_CKPT=5
SIGMA=0.3   # mild action noise so the 128 envs aren't identical; lower=more deterministic

for ckpt_name in "${CKPTS[@]}"; do
    ckpt_path="$CKPTS_DIR/$ckpt_name"
    if [ ! -f "$ckpt_path" ]; then
        echo "[skip] missing ckpt: $ckpt_path"
        continue
    fi
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

    TMPDIR="$CACHE_DIR/tmp" \
    XDG_CACHE_HOME="$CACHE_DIR/cache" \
    OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
    OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
    TORCH_HOME="$CACHE_DIR/torch" \
    TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
    TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
    FORGE_SKIP_TACTILE_SENSORS=1 \
    FORGE_SAVE_TACTILE_ALL_ENVS=1 \
    FORGE_SAVE_CAMERA=1 \
    FORGE_ENABLE_FRONT_CAM=1 \
    FORGE_TACTILE_SAVE_DIR="$save_dir" \
    ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
        --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
        --baseline single_pos \
        --checkpoint "$ckpt_path" \
        --num_envs 128 \
        --max_iterations $max_iters_abs \
        --sigma $SIGMA \
        --headless \
        --enable_cameras \
        agent.params.config.learning_rate=0 \
        agent.params.config.lr_schedule=fixed \
        agent.params.config.save_frequency=999999 \
        agent.params.config.save_best_after=999999 \
        agent.params.config.full_experiment_name=PegInsert_PickPlace_curriculum_rollout_tmp
done

echo "==== done. dataset root: $BASE_SAVE_DIR ===="
