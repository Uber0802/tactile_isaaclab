#!/bin/bash
# Multi-position twin of run_pegpickplace_curriculum_rollout_seed2_paired.sh.
#
# Same ckpt selection, same 64 envs, same paired tactile + RGB output — the
# ONLY difference is the reset distribution:
#
#   single_pos version  -> --baseline single_pos  (poses zeroed, source hole
#                                                  pinned at +10cm X)
#   this version        -> --baseline A_legacy    (as trained: hand/peg pose
#                                                  noise live, source hole
#                                                  U(+/-10cm) in xy with a 5cm
#                                                  minimum separation)
#
# Source run: PegInsert_PickPlace_baselineA_legacy_seed2
#             (wandb dfk2g15p, 2026-06-08 09:06 -> 2026-06-12, 256 envs,
#              save_frequency=200)
#
# Output (separate root so each dataset stays usable as-is with a plain
# `--data_dirs <root>` glob):
#   /mnt/scratch/tactile/pegpickplace_curriculum_seed2_paired_multipos/<run>_ep_NNN/
#     ep<N>_env<XXX>.npy          <- tactile (T=299, 40, 25, 3) float16  ~1.8 MB
#     ep<N>_env<XXX>_camera.npy   <- RGB     (T=299, 224, 224, 3) uint8  ~45 MB
#
# A_legacy is the baseline seed2 was trained with, so this pass is exactly its
# training distribution. Do NOT substitute frozen `A`: that restores the full
# 1.5 xy/z align bridges and the 4cm descent gate, which is a different task.
#
# ---------------------------------------------------------------------------
# WHY THIS IS NOT A REAL CURRICULUM ROLLOUT
# ---------------------------------------------------------------------------
# Unlike gear (13 snapshots) and nut (10), the peg runs kept only their LAST
# TWO checkpoints — everything earlier was deleted, and no backup exists on
# /mnt/lab-tank, /mnt/scratch or /mnt/tank. What survives per run:
#
#   PegInsert_PickPlace_baselineA_legacy_seed2   ep_5200, ep_5600
#   PegInsert_PickPlace_baselineA_legacy_seed1   ep_5060, ep_5200
#   PegInsert_PickPlace_baselineA_legacy         ep_5280, ep_5300
#   PegInsert_PickPlace_baselineA_legacy_seed3   ep_320,  ep_480
#
# seed2's own wandb success curve (from the local output.log) is fine —
#   ep<=1000 0% | ep_1400 4.5% | ep_1800 42% | ep_2600 56% | ep_3000 72%
#   ep_3400 84% | ep_4200 86% | ep_5200 90.3% | ep_5600 86.7%
# — but every snapshot below the plateau is gone. Both surviving seed2 ckpts
# sit at ~90% success, so a seed2-only dataset is ~87% positives and gives the
# reward model almost no failure trajectories to contrast against.
#
# WITH_SEED3=1 appends seed3's ep_320 / ep_480. That run died at ep_486 and
# both snapshots are at 0% success, so they supply the negative half. They come
# from a different seed, but for reward-model training what matters is the
# behaviour distribution (never-inserts vs inserts), not seed identity.
# Recommended unless you are re-collecting purely as an eval split.
# ---------------------------------------------------------------------------
#
# Usage:
#   WITH_SEED3=1 GPU_ID=5 ./run_pegpickplace_curriculum_rollout_seed2_paired_multipos.sh
#   TARGETS="seed2:5200" ./...            # explicit <run>:<epoch> list
#   NUM_ENVS=128 ./...                    # more trajectories per snapshot

set -e

# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

LOG_ROOT=/mnt/home/tactile/tactile_isaaclab/logs/rl_games/ForgePickPlace
BASE_SAVE_DIR=/mnt/scratch/tactile/pegpickplace_curriculum_seed2_paired_multipos
BASELINE=A_legacy

# `single_pos` is label-safe for peg (unlike nut). peg's A_legacy only rescales
# reward shaping (xy_align_reward_scale / z_align_reward_scale 1.5->0.5,
# descent_xy_threshold 4cm->1cm) and leaves `success_threshold` at the frozen-A
# default of 0.04 — the same value `single_pos` uses. The `Success` int written
# into each .npy therefore means the same thing in both passes.
# Note `single_pos` ALSO pins the source hole at +10cm X from the destination
# (source_hole_fixed_offset), on top of zeroing the pose randomizers; under
# A_legacy that offset is U(+/-10cm) in xy with a 5cm minimum separation.

# All six GPUs on this box are usually busy with training. Pin the rollout with
# e.g. `GPU_ID=5 ./run_pegpickplace_curriculum_rollout_seed2_paired_multipos.sh`
# (2 and 5 are the 98 GB Blackwells and normally have the most free memory).
if [ -n "${GPU_ID:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "[info] pinned to GPU $GPU_ID"
fi

run_dir_for() {
    case "$1" in
        seed2) echo "$LOG_ROOT/PegInsert_PickPlace_baselineA_legacy_seed2" ;;
        seed1) echo "$LOG_ROOT/PegInsert_PickPlace_baselineA_legacy_seed1" ;;
        seed3) echo "$LOG_ROOT/PegInsert_PickPlace_baselineA_legacy_seed3" ;;
        base)  echo "$LOG_ROOT/PegInsert_PickPlace_baselineA_legacy" ;;
        *)     echo "" ;;
    esac
}

# Targets are `<run>:<epoch>` pairs so the list can span runs — necessary here
# because no single peg run retained both a low-skill and a high-skill ckpt.
if [ -n "${TARGETS:-}" ]; then
    read -r -a TARGET_LIST <<< "$TARGETS"
    echo "[info] TARGETS override: ${TARGET_LIST[*]}"
else
    TARGET_LIST=(seed2:5200 seed2:5600)          # ~90.3% / ~86.7% success
    if [ "${WITH_SEED3:-0}" = "1" ]; then
        TARGET_LIST+=(seed3:320 seed3:480)       # ~0% / ~0% success
        echo "[info] WITH_SEED3=1 — appending seed3 ep_320 / ep_480 as negatives"
    else
        echo "[warn] seed2-only: both ckpts are ~90% success, so this dataset"
        echo "[warn] will be almost all positives. Re-run with WITH_SEED3=1 to"
        echo "[warn] add the 0%-success seed3 snapshots."
    fi
fi

SELECTED=()   # "<run>|<ckpt filename>"
for target in "${TARGET_LIST[@]}"; do
    run="${target%%:*}"
    ep="${target##*:}"
    dir=$(run_dir_for "$run")
    if [ -z "$dir" ]; then
        echo "[fatal] unknown run key '$run' in target '$target'"; exit 1
    fi
    matches=$(find "$dir/nn" -maxdepth 1 -name "last_ForgePickPlace_ep_${ep}_rew_*.pth" \
                   -printf '%f\n' 2>/dev/null | sort)
    n=$(printf '%s' "$matches" | grep -c . || true)
    if [ "$n" -eq 1 ]; then
        SELECTED+=("$run|$matches")
    elif [ "$n" -eq 0 ]; then
        echo "[fatal] $run ep_$ep not found under $dir/nn"
        echo "        (peg runs keep only their last two ckpts — see header)"
        exit 1
    else
        echo "[fatal] $run ep_$ep matched $n ckpts, expected 1:"
        printf '          %s\n' $matches
        exit 1
    fi
done
echo "Selected ${#SELECTED[@]}/${#TARGET_LIST[@]} ckpts:"
printf '  %s\n' "${SELECTED[@]}"

NUM_ENVS=${NUM_ENVS:-64}   # -> NUM_ENVS trajectories per ckpt (per-env quota = 1)
ITERS_PER_CKPT=5           # safety upper bound; early-exits once every env has saved
SIGMA=0.3                  # mild action noise so the envs aren't identical
EXPECTED_FILES=$((NUM_ENVS * 2))   # tactile + camera per env

echo "baseline=$BASELINE  num_envs=$NUM_ENVS  -> $BASE_SAVE_DIR"

for entry in "${SELECTED[@]}"; do
    run="${entry%%|*}"
    ckpt_name="${entry##*|}"
    ckpt_path="$(run_dir_for "$run")/nn/$ckpt_name"
    ckpt_epoch=$(echo "$ckpt_name" | grep -oE 'ep_[0-9]+' | cut -d_ -f2)
    # Labelled with the run key too — targets can span runs, and ep numbers
    # collide across them (seed1 and seed2 both have an ep_5200).
    label="${run}_ep_${ckpt_epoch}"
    save_dir="$BASE_SAVE_DIR/$label"

    # Resume-friendly: skip any ckpt whose output directory is already complete.
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
        --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
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
        agent.params.config.full_experiment_name=PegInsert_PickPlace_curriculum_rollout_multipos_tmp
done

echo "==== done. dataset root: $BASE_SAVE_DIR ===="
