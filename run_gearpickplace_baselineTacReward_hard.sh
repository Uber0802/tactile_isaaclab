# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# Baseline A_hard (yaw_reward=0 + wider pose noise) + tactile reward shaping.
# Pair with run_gearpickplace_baselineA_hard.sh to test whether tactile reward
# can replace the cut yaw shaping signal in a regime where baselineA alone
# struggles to crack the gear-teeth alignment.
# Ckpt path uses the immutable `exp_taskcompare_fulltrajonly_1779004236` backup
# (no zero-contact aug), which RL-deployment-proved beats zeroaug on gear.
TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
FORGE_TACTILE_REWARD_CKPT=/mnt/lab-tank/uber/Tactile-Reward/exp_taskcompare/gear_scratch/gear_scratch_epoch18.pth \
FORGE_TACTILE_REWARD_LOG_DIR=/mnt/lab-home/tactile/tactile_isaaclab/logs/tactile_curves/GearMesh_TacReward_hard_success_0.175 \
FORGE_TACTILE_REWARD_SCALE=0.175 \
FORGE_TACTILE_REWARD_INSTRUCTION="pick up the gear and mesh it onto the shaft" \
FORGE_TACTILE_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/Tactile-ReWiND \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
    --baseline A_hard_success \
    --headless \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --wandb-entity b11902127-ntu \
    --wandb-project-name tactile-rewind \
    --wandb-name GearMesh_PickPlace_baselineTacReward_hard_success_0.175 \
    agent.params.config.full_experiment_name=GearMesh_PickPlace_baselineTacReward_hard_success_0.175 \
    agent.params.config.save_frequency=20
