#!/bin/bash
# Multi-position twin of run_nutpickplace_curriculum_rollout_seed2_paired.sh.
#
# Same seed2 skill curve, same 10 snapshots, same 64 envs, same paired
# tactile + RGB output — the ONLY difference is the reset distribution:
#
#   single_pos version  -> --baseline A_hard_success_single_pos  (poses zeroed)
#   this version        -> --baseline A_hard_success             (as trained)
#
# Source run: NutThread_PickPlace_baselineA_hard_success_-3.5_seed2
#             (wandb ysze8n16, 2026-06-14 11:50 -> 2026-06-25, 256 envs,
#              save_frequency=500 -> ckpts at ep 500..5000)
#
# Output (separate root so each dataset stays usable as-is with a plain
# `--data_dirs <root>` glob):
#   /mnt/scratch/tactile/nutpickplace_curriculum_seed2_paired_multipos/ep_NNN/
#     ep<N>_env<XXX>.npy          <- tactile (T, 40, 25, 3) float16
#     ep<N>_env<XXX>_camera.npy   <- RGB     (T, 224, 224, 3) uint8   ~45 MB
#
# ---------------------------------------------------------------------------
# Why A_hard_success (i.e. the training baseline itself) is the multipos mode
# ---------------------------------------------------------------------------
# Unlike gear, nut's A_hard_success does NOT zero any randomizer — it WIDENS
# them (hand_init_pos_noise 2cm->8cm xy / 1cm->3cm z, fixed_asset_init_pos_noise
# 5cm->10cm xy +3cm z) and leaves nut_thread's wide orientation defaults live
# (hand_init_orn_noise yaw = 1.57 rad, fixed_asset_init_orn_range_deg = 360).
# So running the ckpt under its own training baseline already gives full
# position AND yaw randomization — there is nothing extra to switch on, and
# using frozen `A` instead would change success_threshold back to +0.375 and
# re-enable unidirectional_rot, corrupting the saved Success labels.
#
# Expect success rates close to the wandb curve below (~85% at plateau) and a
# much higher failure count than the single_pos pass — which is exactly what
# the reward model needs.
# ---------------------------------------------------------------------------
#
# Tactile sensors stay ON (no FORGE_SKIP_TACTILE_SENSORS, unlike the training
# script) — the sensors are what we are here to record, and they do not enter
# the obs for an A_hard_success policy.

set -e

# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

CKPTS_DIR=/mnt/home/tactile/tactile_isaaclab/logs/rl_games/ForgeNutPickPlace/NutThread_PickPlace_baselineA_hard_success_-3.5_seed2/nn
BASE_SAVE_DIR=/mnt/scratch/tactile/nutpickplace_curriculum_seed2_paired_multipos
BASELINE=A_hard_success

# All six GPUs on this box are usually busy with training. Pin the rollout with
# e.g. `GPU_ID=5 ./run_nutpickplace_curriculum_rollout_seed2_paired_multipos.sh`
# (2 and 5 are the 98 GB Blackwells and normally have the most free memory).
if [ -n "${GPU_ID:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "[info] pinned to GPU $GPU_ID"
fi

# Every ckpt this run wrote (save_frequency=500). Success rates read off the
# seed2 wandb log (mean over +/-25 iters):
#   ep_500  ~  0.2%   nut never threads
#   ep_1000 ~  8.3%   crack starting
#   ep_1500 ~ 76.6%   crack complete (the 1000->1500 transition is NOT
#                     resolvable — no ckpt was saved inside it)
#   ep_2000 ~ 83.3%
#   ep_2500 ~ 84.6%
#   ep_3000 ~ 84.0%
#   ep_3500 ~ 86.7%
#   ep_4000 ~ 85.2%
#   ep_4500 ~ 86.4%
#   ep_5000 ~ 86.1%   peak
# NOTE: unlike gear (13 ckpts, 3 of them at 0% success) this curve gives only
# ~2 failure-heavy snapshots against 8 plateau ones, so the combined dataset
# skews success-heavy. The multipos twin is where most of the failures come
# from — collect both before training a reward model on this task.
# Override for smoke tests / partial re-runs, e.g. `EPS="5000" ./thisscript.sh`
if [ -n "${EPS:-}" ]; then
    read -r -a TARGET_EPS <<< "$EPS"
    echo "[info] EPS override: ${TARGET_EPS[*]}"
else
    TARGET_EPS=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000)
fi

CKPTS=()
for ep in "${TARGET_EPS[@]}"; do
    matches=$(find "$CKPTS_DIR" -maxdepth 1 -name "last_ForgeNutPickPlace_ep_${ep}_rew_*.pth" \
                   -printf '%f\n' | sort)
    n=$(printf '%s' "$matches" | grep -c . || true)
    if [ "$n" -eq 1 ]; then
        CKPTS+=("$matches")
    elif [ "$n" -eq 0 ]; then
        echo "[warn] ep_$ep has no ckpt — skipping"
    else
        # This run's nn/ currently holds exactly one file per epoch. If a rerun
        # ever writes under the same full_experiment_name, add an mtime filter
        # like the gear seed2 script's RUN2_AFTER guard rather than picking one.
        echo "[fatal] ep_$ep matched $n ckpts, expected 1:"
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

    # Resume-friendly: a full run is ~10 ckpts x several minutes, so skip any
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
        --task Isaac-Forge-NutThread-PickPlace-Direct-v0 \
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
        agent.params.config.full_experiment_name=NutThread_PickPlace_curriculum_rollout_multipos_tmp
done

echo "==== done. dataset root: $BASE_SAVE_DIR ===="
