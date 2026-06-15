# Repro of wandb run b11902127-ntu/tactile-rewind/5knf2i2t
#   name : PegInsert_PickPlace_baselineTacReward_legacy0.1
#   date : 2026-05-23, git state 8b0b576 (working tree had uncommitted edits:
#          the committed script at 8b0b576 was scale=0.3 / num_envs=256 / name …0.3,
#          but the actual run was the 0.1 variant — so it ran from local edits
#          that were never committed; git alone cannot reconstruct it).
#
# This is the TacReward (tactile-reward) experiment, NOT baseline A.
# --baseline A_legacy only sets the reward shaping (half XY/Z bridges, 1cm
# descent gate); the tactile reward itself comes from the FORGE_TACTILE_REWARD_*
# env vars below. The reward ckpt is peg_curr_epoch17.pth (the run's actual
# ckpt — ignore the stale comment in run_pegpickplace_baselineTacReward_legacy.sh
# that mentions isaaclab_overfit_epoch99.pth).
#
# !!! TWO VALUES NOT RECORDED BY WANDB — FILL THESE BEFORE RUNNING !!!
#   --num_envs : the sibling 0.3 run used 256; the later 0.1_seed2 used 128.
#                The original 0.1 run's value is unknown. Defaulting to 256.
#   --seed     : the original 0.1 run almost certainly passed NO --seed
#                (default None = random), so an exact rerun is impossible.
#                Leave it unset to match, or pin one to make YOUR rerun
#                deterministic (it just won't match the original bit-for-bit).
#
# For the closest reproduction, also: `git checkout 8b0b576` first, since
# forge_env.py drifted ~500 lines afterwards (mostly opt-in visual-reward code,
# but not guaranteed identical).

CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
FORGE_TACTILE_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/peg_curriculum_retrain/peg_curr_epoch17.pth \
FORGE_TACTILE_REWARD_SCALE=0.1 \
FORGE_TACTILE_REWARD_INSTRUCTION="grasp peg and insert to another hole" \
FORGE_TACTILE_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/Tactile-ReWiND \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
    --baseline A_legacy \
    --headless \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --wandb-entity b11902127-ntu \
    --wandb-project-name tactile-rewind \
    --wandb-name PegInsert_PickPlace_baselineTacReward_legacy0.1 \
    agent.params.config.full_experiment_name=PegInsert_PickPlace_baselineTacReward_legacy0.1 \
    agent.params.config.save_frequency=20
