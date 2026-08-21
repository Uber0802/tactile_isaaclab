#!/bin/bash
# Gear pickplace data collection — curriculum sweep: roll out each baselineA
# snapshot (worst→best) in single_pos so tactile/RGB variation across the
# dataset comes purely from policy-skill progression. RGB saved (tactile
# sensors off). Output: $BASE_SAVE_DIR/<ep_label>/ep*.npy
set -e
source "$(dirname "$0")/_common.sh"

CKPTS_DIR=/mnt/home/uber/tactile_isaaclab/logs/rl_games/ForgeGearPickPlace/GearMesh_PickPlace_baselineA/nn
BASE_SAVE_DIR=/mnt/tank/tactile/tactile_dataset/gearpickplace_curriculum_rgb

# Ep list spanning the gear baselineA skill curve (0% → ~79% at ep_740).
TARGET_EPS=(20 100 260 340 420 500 580 660 740)
ITERS_PER_CKPT=20
SIGMA=0.3   # mild action noise so the 64 envs aren't identical

CKPTS=()
for ep in "${TARGET_EPS[@]}"; do
    match=$(find "$CKPTS_DIR" -name "last_ForgeGearPickPlace_ep_${ep}_rew_*.pth" 2>/dev/null | head -1)
    [ -n "$match" ] && CKPTS+=("$(basename "$match")") || echo "[warn] missing ep_$ep — skipping"
done
echo "Selected ${#CKPTS[@]} ckpts: ${CKPTS[*]}"

for ckpt_name in "${CKPTS[@]}"; do
    ckpt_path="$CKPTS_DIR/$ckpt_name"
    [ -f "$ckpt_path" ] || { echo "[skip] missing ckpt: $ckpt_path"; continue; }
    label=$(echo "$ckpt_name" | grep -oE 'ep_[0-9]+')
    # rl_games restores the ckpt's epoch counter, so --max_iterations is absolute;
    # add ITERS_PER_CKPT to actually run that many iters past the loaded ckpt.
    max_iters_abs=$(( ${label#ep_} + ITERS_PER_CKPT ))
    save_dir="$BASE_SAVE_DIR/$label"
    mkdir -p "$save_dir"
    echo "==== [$label] ckpt=$ckpt_name → $save_dir  (max_epochs=$max_iters_abs) ===="

    FORGE_SKIP_TACTILE_SENSORS=1 \
    FORGE_ENABLE_FRONT_CAM=1 \
    FORGE_DISABLE_YAW_DIFF_OBS=1 \
    ./isaaclab.sh -p "$TRAIN" \
    "env.tactile_save.all_envs=true" \
    "env.tactile_save.camera=true" \
    "env.tactile_save.episodes_per_env=1" \
    "env.tactile_save.save_dir=$save_dir" \
        --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
        --baseline single_pos \
        --checkpoint "$ckpt_path" \
        --num_envs 64 --max_iterations $max_iters_abs --sigma $SIGMA \
        --headless --enable_cameras \
        agent.params.config.learning_rate=0 \
        agent.params.config.lr_schedule=fixed \
        agent.params.config.save_frequency=999999 \
        agent.params.config.save_best_after=999999 \
        agent.params.config.full_experiment_name=GearMesh_PickPlace_datacollection_tmp
done

echo "==== done. dataset root: $BASE_SAVE_DIR ===="
