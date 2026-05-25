# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# LEGACY peg TacReward script — reproduces the May 17 run `q50f4175` that hit
# 62.5% success at ep 900 / 16h 40m before crashing. Uses the older
# `checkpoints_data_3_baseline_3ch_r03_flip_split/isaaclab_overfit_epoch99.pth`
# tactile reward ckpt (pre-fulltrajonly, pre-curriculum-retrain) at scale 0.1.
# Kept around as a baseline reference; new experiments should use the
# curriculum-retrained or fulltrajonly ckpts and live in
# run_pegpickplace_baselineTacReward.sh / _hard_success.sh.
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
    --num_envs 128 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --wandb-entity b11902127-ntu \
    --wandb-project-name tactile-rewind \
    --wandb-name PegInsert_PickPlace_baselineTacReward_legacy0.1 \
    agent.params.config.full_experiment_name=PegInsert_PickPlace_baselineTacReward_legacy0.1 \
    agent.params.config.save_frequency=20