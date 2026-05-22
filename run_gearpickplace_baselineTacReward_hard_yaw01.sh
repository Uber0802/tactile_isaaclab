# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# Baseline A_hard_success_yaw01 + tactile reward shaping. A_hard_success
# ablations (yaw_reward=0 → 0.1, gear yaw fixed, success_threshold=-0.3) plus
# the curriculum-distribution-matched tactile reward signal at scale 0.3.
# Pair with run_gearpickplace_baselineA_hard_yaw01.sh to measure tactile's
# marginal benefit when a mild yaw hint is already present.
TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
FORGE_TACTILE_REWARD_CKPT=/mnt/lab-tank/uber/Tactile-Reward/exp_taskcompare/gear_scratch/gear_scratch_epoch18.pth \
FORGE_TACTILE_REWARD_LOG_DIR=/mnt/lab-home/tactile/tactile_isaaclab/logs/tactile_curves/GearMesh_TacReward_hard_success_yaw01_0.3 \
FORGE_TACTILE_REWARD_SCALE=0.3 \
FORGE_TACTILE_REWARD_SMOOTH_ALPHA=0.2 \
FORGE_TACTILE_REWARD_INSTRUCTION="pick up the gear and mesh it onto the shaft" \
FORGE_TACTILE_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/Tactile-ReWiND \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
    --baseline A_hard_success_yaw01 \
    --headless \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --wandb-entity b11902127-ntu \
    --wandb-project-name tactile-rewind \
    --wandb-name GearMesh_PickPlace_baselineTacReward_hard_success_yaw01_0.3 \
    agent.params.config.full_experiment_name=GearMesh_PickPlace_baselineTacReward_hard_success_yaw01_0.3 \
    agent.params.config.save_frequency=20
