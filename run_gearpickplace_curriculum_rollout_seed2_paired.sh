#!/bin/bash
# Curriculum rollout over the seed2 skill curve — load each snapshot from
# "barely moves" to "peak success", run briefly in single_pos, and save BOTH
# the tactile force field AND the RGB front camera per trajectory.
#
# Source run: GearMesh_PickPlace_baselineA_hard_success_yaw01_yawinput0.5_seed2
#             (trained --baseline A_hard_success_yaw01, seed 2, 256 envs,
#              ep 20..4500, finished 2026-08-02 ~06:10)
#
# Output (on /mnt/scratch, not /mnt/tank — 58 TB free vs tank's 4.4 TB):
#   /mnt/scratch/tactile/gearpickplace_curriculum_seed2_paired/ep_NNN/
#     ep<N>_env<XXX>.npy          <- tactile (T, 40, 25, 3) float16   ~1.8 MB
#     ep<N>_env<XXX>_camera.npy   <- RGB     (T, 224, 224, 3) uint8   ~45 MB
#
# Init pose is fixed (single_pos), so variation across the combined dataset
# comes purely from policy-skill progression, not from the reset distribution.
#
# Differences vs run_gearpickplace_curriculum_rollout.sh (the baselineA / RGB-only
# version whose output already sits in gearpickplace_curriculum_rgb/):
#   1. Tactile sensors stay ON  -> no FORGE_SKIP_TACTILE_SENSORS.
#   2. NO FORGE_DISABLE_YAW_DIFF_OBS. That flag strips `yaw_diff_to_fixed` to
#      recreate the legacy 37-dim obs of the May-10 baselineA ckpts. seed2 was
#      trained with the flag unset, so its actor expects the full 39-dim layout
#      and setting it here would silently load a shape-mismatched policy.
#   3. Duplicate-epoch guard, see RUN2_AFTER below.
#
# Why `--baseline single_pos` is safe for an A_hard_success_yaw01 ckpt:
# single_pos only zeroes reset-pose randomizers; it does not touch obs_order,
# and obs_order carries no success/engagement flags, so the differing
# success_threshold / yaw_reward_scale affect only logged metrics and reward —
# both irrelevant at learning_rate=0. Episodes are fixed-length (no
# terminate-on-success), so trajectory lengths are unaffected too. The rollout
# distribution is a strict subset of what seed2 trained on, not OOD.

set -e

# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

CKPTS_DIR=/mnt/home/tactile/tactile_isaaclab/logs/rl_games/ForgeGearPickPlace/GearMesh_PickPlace_baselineA_hard_success_yaw01_yawinput0.5_seed2/nn
BASE_SAVE_DIR=/mnt/scratch/tactile/gearpickplace_curriculum_seed2_paired

# All six GPUs on this box are usually busy with training. Pin the rollout with
# e.g. `GPU_ID=5 ./run_gearpickplace_curriculum_rollout_seed2_paired.sh`
# (2 and 5 are the 98 GB Blackwells and normally have the most free memory).
if [ -n "${GPU_ID:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "[info] pinned to GPU $GPU_ID"
fi

# ---------------------------------------------------------------------------
# Duplicate-epoch guard.
#
# This nn/ directory holds TWO runs that were both written under the same
# full_experiment_name: a first attempt (2026-07-29 15:33 -> ep 1040, died) and
# the real run (2026-07-30 07:17 -> ep 4500). Every epoch from 20 to 1040
# therefore exists twice with different rewards, e.g.
#   ep_1000_rew_752.4615.pth   (attempt 1)
#   ep_1000_rew_999.7833.pth   (run 2)
# A bare `find ... | head -1` would mix weights from two different learning
# curves. Filter by mtime so every selected ckpt comes from run 2.
# ---------------------------------------------------------------------------
RUN2_AFTER='2026-07-30 07:00'

# Epochs chosen off the seed2 success curve (successes/iter, smoothed +/-25 iter).
# Dense through the 500->600 crack, sparse across the post-2000 plateau.
#   ep_20   ~  0%   random policy, barely lifts the gear
#   ep_200  ~  0%   pre-crack, transport learned but no meshing
#   ep_400  ~  0%   pre-crack, reward climbing (584) but still 0 success
#   ep_500  ~  2%   crack begins
#   ep_560  ~ 15%   mid-crack
#   ep_600  ~ 29%   crack complete
#   ep_700  ~ 32%
#   ep_900  ~ 39%
#   ep_1200 ~ 40%
#   ep_1600 ~ 47%
#   ep_2000 ~ 49%
#   ep_2900 ~ 52%
#   ep_4100 ~ 55%   peak
# Override for smoke tests / partial re-runs, e.g. `EPS="4100" ./thisscript.sh`
if [ -n "${EPS:-}" ]; then
    read -r -a TARGET_EPS <<< "$EPS"
    echo "[info] EPS override: ${TARGET_EPS[*]}"
else
    TARGET_EPS=(20 200 400 500 560 600 700 900 1200 1600 2000 2900 4100)
fi

CKPTS=()
for ep in "${TARGET_EPS[@]}"; do
    matches=$(find "$CKPTS_DIR" -maxdepth 1 -name "last_ForgeGearPickPlace_ep_${ep}_rew_*.pth" \
                   -newermt "$RUN2_AFTER" -printf '%f\n' | sort)
    n=$(printf '%s' "$matches" | grep -c . || true)
    if [ "$n" -eq 1 ]; then
        CKPTS+=("$matches")
    elif [ "$n" -eq 0 ]; then
        echo "[warn] ep_$ep has no run-2 ckpt (newer than $RUN2_AFTER) — skipping"
    else
        echo "[fatal] ep_$ep matched $n run-2 ckpts, expected 1:"
        printf '          %s\n' $matches
        exit 1
    fi
done
echo "Selected ${#CKPTS[@]}/${#TARGET_EPS[@]} ckpts:"
printf '  %s\n' "${CKPTS[@]}"

NUM_ENVS=64                # -> 64 trajectories per ckpt (per-env quota = 1)
ITERS_PER_CKPT=5           # safety upper bound; early-exits once every env has saved
SIGMA=0.3                  # mild action noise so the 64 envs aren't identical
EXPECTED_FILES=$((NUM_ENVS * 2))   # tactile + camera per env

for ckpt_name in "${CKPTS[@]}"; do
    ckpt_path="$CKPTS_DIR/$ckpt_name"
    label=$(echo "$ckpt_name" | grep -oE 'ep_[0-9]+')
    ckpt_epoch=${label#ep_}
    save_dir="$BASE_SAVE_DIR/$label"

    # Resume-friendly: a full run is ~13 ckpts x several minutes, so skip any
    # ckpt whose output directory is already complete.
    if [ -d "$save_dir" ] && [ "$(ls "$save_dir" | wc -l)" -ge "$EXPECTED_FILES" ]; then
        echo "==== [$label] already has $EXPECTED_FILES files — skipping ===="
        continue
    fi

    # rl_games loads the ckpt's epoch counter on restore, so `--max_iterations N`
    # (which sets `max_epochs = N`, absolute) would exit immediately if N <=
    # ckpt_epoch. Compute the absolute target so we actually run ITERS_PER_CKPT
    # PPO iters relative to the loaded ckpt.
    max_iters_abs=$((ckpt_epoch + ITERS_PER_CKPT))
    mkdir -p "$save_dir"
    echo "==== [$label] ckpt=$ckpt_name -> $save_dir  (max_epochs=$max_iters_abs) ===="

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
    FORGE_TACTILE_SAVE_DIR="$save_dir" \
    ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
        --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
        --baseline single_pos \
        --checkpoint "$ckpt_path" \
        --num_envs $NUM_ENVS \
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
