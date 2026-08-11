#!/bin/bash
# Curriculum rollout over the nut seed2 skill curve — load each snapshot from
# "never threads" to "peak success", run briefly with a FIXED init pose, and
# save BOTH the tactile force field AND the RGB front camera per trajectory.
#
# Source run: NutThread_PickPlace_baselineA_hard_success_-3.5_seed2
#             (wandb ysze8n16, 2026-06-14 11:50 -> 2026-06-25, 256 envs,
#              save_frequency=500 -> ckpts at ep 500..5000)
#
# Output (on /mnt/scratch, not /mnt/tank — 58 TB free vs tank's 4.4 TB):
#   /mnt/scratch/tactile/nutpickplace_curriculum_seed2_paired/ep_NNN/
#     ep<N>_env<XXX>.npy          <- tactile (T, 40, 25, 3) float16
#     ep<N>_env<XXX>_camera.npy   <- RGB     (T, 224, 224, 3) uint8   ~45 MB
#
# The multi-position twin is run_nutpickplace_curriculum_rollout_seed2_paired_multipos.sh.
#
# ---------------------------------------------------------------------------
# Why --baseline A_hard_success_single_pos and NOT plain single_pos
# ---------------------------------------------------------------------------
# The gear script uses `single_pos`, but that is unsafe for nut. `single_pos`
# only zeroes the pose randomizers; everything else reverts to frozen-A
# defaults, including two fields that change what lands on disk:
#
#   1. success_threshold: nut_thread's default is +0.375
#      (= thread_pitch 2mm * 0.375 = nut 0.75mm ABOVE target, "barely
#      touching"), while seed2 was trained/evaluated at -3.5 (7mm DEEP).
#      The `Success` int saved into every .npy is `ep_succeeded`, which is
#      computed from `cfg_task.success_threshold` — so under plain single_pos
#      essentially every episode that reaches the bolt gets labelled success
#      and the reward model would train on garbage labels.
#   2. unidirectional_rot: reverts to True, which silently clamps positive
#      commanded delta_yaw to 0. seed2 trained with it False.
#
# `A_hard_success_single_pos` (added in forge_nutpickplace_env_cfg.py) applies
# the full A_hard_success config first and zeroes the six pose randomizers
# after, so only the reset distribution changes.
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
BASE_SAVE_DIR=/mnt/scratch/tactile/nutpickplace_curriculum_seed2_paired
BASELINE=A_hard_success_single_pos

# All six GPUs on this box are usually busy with training. Pin the rollout with
# e.g. `GPU_ID=5 ./run_nutpickplace_curriculum_rollout_seed2_paired.sh`
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
        agent.params.config.full_experiment_name=NutThread_PickPlace_curriculum_rollout_tmp
done

echo "==== done. dataset root: $BASE_SAVE_DIR ===="
