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

CKPTS_DIR=/mnt/home/uber/tactile_isaaclab/logs/rl_games/ForgeGearPickPlace/GearMesh_PickPlace_baselineA/nn
BASE_SAVE_DIR=/mnt/tank/tactile/tactile_dataset/gearpickplace_curriculum_rgb

# Explicit ep list spanning the gear baselineA skill curve (gearmesh-5-10-y3pgpab6):
#   ep_20   ~ 0%   (random init)
#   ep_100  ~ 1%   (pre-crack)
#   ep_260  ~ 0%   (still pre-crack)
#   ep_340  ~ 1%
#   ep_420  ~ 25%  (first crack)
#   ep_500  ~ 50%
#   ep_580  ~ 71%
#   ep_660  ~ 75%
#   ep_740  ~ 79%  (peak)
TARGET_EPS=(20 100 260 340 420 500 580 660 740)
CKPTS=()
for ep in "${TARGET_EPS[@]}"; do
    match=$(find "$CKPTS_DIR" -name "last_ForgeGearPickPlace_ep_${ep}_rew_*.pth" 2>/dev/null | head -1)
    if [ -n "$match" ]; then
        CKPTS+=("$(basename "$match")")
    else
        echo "[warn] missing ep_$ep — skipping"
    fi
done
echo "Selected ${#CKPTS[@]} ckpts: ${CKPTS[*]}"

# Per-ckpt rollout budget. One PPO iter ≈ horizon_length × num_envs steps.
# Tune ITERS_PER_CKPT for the trajectory count you want per skill level.
ITERS_PER_CKPT=20
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
    FORGE_DISABLE_YAW_DIFF_OBS=1 \
    FORGE_TACTILE_EPISODES_PER_ENV=1 \
    FORGE_TACTILE_SAVE_DIR="$save_dir" \
    ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
        --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
        --baseline single_pos \
        --checkpoint "$ckpt_path" \
        --num_envs 64 \
        --max_iterations $max_iters_abs \
        --sigma $SIGMA \
        --enable_cameras \
        --headless \
        agent.params.config.learning_rate=0 \
        agent.params.config.lr_schedule=fixed \
        agent.params.config.save_frequency=999999 \
        agent.params.config.save_best_after=999999 \
        agent.params.config.full_experiment_name=GearMesh_PickPlace_curriculum_rollout_tmp
done

echo "==== done. dataset root: $BASE_SAVE_DIR ===="
