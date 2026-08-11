#!/bin/bash
# Multi-position twin of run_gearpickplace_curriculum_rollout_seed2_paired.sh.
#
# Same seed2 skill curve, same 13 snapshots, same 64 envs, same paired
# tactile + RGB output — the ONLY difference is the reset distribution:
#
#   single_pos version  -> --baseline single_pos            (every pose zeroed)
#   this version        -> --baseline A_hard_success_yaw01  (positions vary)
#
# Output (separate root so the existing single_pos dataset stays usable as-is
# with a plain `--data_dirs <root>` glob):
#   /mnt/scratch/tactile/gearpickplace_curriculum_seed2_paired_multipos/ep_NNN/
#     ep<N>_env<XXX>.npy          <- tactile (T, 40, 25, 3) float16   ~1.8 MB
#     ep<N>_env<XXX>_camera.npy   <- RGB     (T, 224, 224, 3) uint8   ~45 MB
#
# Why A_hard_success_yaw01 and NOT A / A_hard:
#   A_hard_success_yaw01 is exactly what seed2 was trained with, so the rollout
#   stays in-distribution. It keeps the position randomizers live
#   (gear_table_pos_noise [0.03,0.03,0], fixed_asset_init_pos_noise
#   [0.05,0.05,0], hand_init_pos_noise [0.02,0.02,0.01]) but inherits
#   A_hard_success's yaw zeroing (gear_table_yaw_range = 0,
#   fixed_asset_init_orn_range_deg = 0, hand_init_orn_noise = 0).
#   -> "multi-POSITION", not "multi-yaw". Baseline A would restore 0..360 deg
#   yaw, which seed2 never saw; every episode would fail and the tactile
#   patterns would be OOD for the reward model.
#   See forge_gearpickplace_env_cfg.py:_apply_baseline_A_hard_success.
#
# Expect a LOWER success rate than the single_pos pass (that one hits ~100% on
# the post-ep_1200 plateau because the fixed pose is the easiest one seed2 ever
# solved; the training curve itself peaks near 55%). That is desirable here:
# it gives the reward model failure examples from LATE checkpoints too, instead
# of negatives that only exist at ep_20 / ep_200 / ep_400.
#
# Usage:
#   GPU_ID=5 ./run_gearpickplace_curriculum_rollout_seed2_paired_multipos.sh
#   EPS="4100" ./...._multipos.sh        # best-ckpt-only eval split
#   NUM_ENVS=128 ./...._multipos.sh      # denser coverage of the pose distribution

set -e

# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

CKPTS_DIR=/mnt/home/tactile/tactile_isaaclab/logs/rl_games/ForgeGearPickPlace/GearMesh_PickPlace_baselineA_hard_success_yaw01_yawinput0.5_seed2/nn
BASE_SAVE_DIR=/mnt/scratch/tactile/gearpickplace_curriculum_seed2_paired_multipos
BASELINE=A_hard_success_yaw01

# All six GPUs on this box are usually busy with training. Pin the rollout with
# e.g. `GPU_ID=5 ./run_gearpickplace_curriculum_rollout_seed2_paired_multipos.sh`
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

# Same epoch list as the single_pos pass so the two datasets are paired
# checkpoint-for-checkpoint.
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

NUM_ENVS=${NUM_ENVS:-64}   # -> NUM_ENVS trajectories per ckpt (per-env quota = 1)
ITERS_PER_CKPT=5           # safety upper bound; early-exits once every env has saved
SIGMA=0.3                  # mild action noise so the envs aren't identical
EXPECTED_FILES=$((NUM_ENVS * 2))   # tactile + camera per env

echo "baseline=$BASELINE  num_envs=$NUM_ENVS  -> $BASE_SAVE_DIR"

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
        --baseline "$BASELINE" \
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
        agent.params.config.full_experiment_name=GearMesh_PickPlace_curriculum_rollout_multipos_tmp
done

echo "==== done. dataset root: $BASE_SAVE_DIR ===="
