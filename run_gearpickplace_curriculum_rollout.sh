#!/bin/bash
# Curriculum rollout: load each baseline-A snapshot, briefly run in single_pos
# to collect tactile trajectories at that skill level, save to a per-ckpt subdir.
# Output: /mnt/tank/tactile/tactile_dataset/gearpickplace_curriculum/<label>/ep*.npy
# Init pose is fixed (baseline single_pos), so tactile variation across the
# combined dataset comes purely from policy-skill progression.

set -e

# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

CKPTS_DIR=/mnt/home/tactile/tactile_isaaclab/logs/rl_games/ForgeGearPickPlace/GearMesh_PickPlace_baselineA/nn
BASE_SAVE_DIR=/mnt/tank/tactile/tactile_dataset/gearpickplace_curriculum

# Curriculum spectrum: sample 1-out-of-every-STRIDE saved snapshots so we hit
# the whole skill range without rolling out every single ckpt.
# baselineA save_frequency=20 → ckpts every 20 epochs → ~40 snapshots over
# 800 epochs. Stride 4 → 10 snapshots spanning unskilled → expert.
CKPT_STRIDE=4
mapfile -t ALL_CKPTS < <(ls -v "$CKPTS_DIR"/last_ForgeGearPickPlace_ep_*.pth 2>/dev/null)
CKPTS=()
for ((i=0; i<${#ALL_CKPTS[@]}; i+=CKPT_STRIDE)); do
    CKPTS+=("$(basename "${ALL_CKPTS[$i]}")")
done

# Per-ckpt rollout budget. One PPO iter ≈ horizon_length × num_envs steps.
# Tune ITERS_PER_CKPT for the trajectory count you want per skill level.
ITERS_PER_CKPT=50
SIGMA=0.3   # mild action noise so the 128 envs aren't identical; lower=more deterministic

for ckpt_name in "${CKPTS[@]}"; do
    ckpt_path="$CKPTS_DIR/$ckpt_name"
    if [ ! -f "$ckpt_path" ]; then
        echo "[skip] missing ckpt: $ckpt_path"
        continue
    fi
    # Label = ep_NNN portion of the filename, e.g. ep_300
    label=$(echo "$ckpt_name" | grep -oE 'ep_[0-9]+')
    save_dir="$BASE_SAVE_DIR/$label"
    mkdir -p "$save_dir"
    echo "==== [$label] ckpt=$ckpt_name → $save_dir ===="

    TMPDIR="$CACHE_DIR/tmp" \
    XDG_CACHE_HOME="$CACHE_DIR/cache" \
    OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
    OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
    TORCH_HOME="$CACHE_DIR/torch" \
    TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
    TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
    FORGE_SAVE_TACTILE_FORCE_FIELD=1 \
    FORGE_TACTILE_SAVE_DIR="$save_dir" \
    ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
        --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
        --baseline single_pos \
        --checkpoint "$ckpt_path" \
        --num_envs 128 \
        --max_iterations $ITERS_PER_CKPT \
        --sigma $SIGMA \
        --enable_cameras \
        --headless
done

echo "==== done. dataset root: $BASE_SAVE_DIR ===="
