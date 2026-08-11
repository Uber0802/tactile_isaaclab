# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# Baseline A_hard_success for peg pickplace (transport shapings cut + tight
# success: peg must touch hole bottom) + tactile reward shaping.
# Ckpt is curriculum-retrained (data from peg curriculum rollout, multi-skill)
# with zero-contact aug and global normalize — covers RL-mid-state distribution
# better than the original taskcompare ckpts.
TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
FORGE_TACTILE_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/peg_curriculum_retrain/peg_curr_epoch17.pth \
FORGE_TACTILE_REWARD_SCALE=0.175 \
FORGE_TACTILE_REWARD_INSTRUCTION="grasp peg and insert to another hole" \
FORGE_TACTILE_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/Tactile-ReWiND \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
    --baseline A_hard_success \
    --headless \
    --num_envs 128 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --wandb-entity b11902127-ntu \
    --wandb-project-name tactile-rewind \
    --wandb-name PegInsert_PickPlace_baselineTacReward_hard_success \
    agent.params.config.full_experiment_name=PegInsert_PickPlace_baselineTacReward_hard_success \
    agent.params.config.save_frequency=20
