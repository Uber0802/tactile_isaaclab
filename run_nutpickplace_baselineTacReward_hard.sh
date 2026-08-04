# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# Baseline A_hard (yaw_reward=0 + wider pose noise) + tactile reward shaping.
# Pair with run_nutpickplace_baselineA_hard.sh to test whether tactile reward
# can replace the cut yaw shaping signal in a regime where baselineA alone
# struggles to crack the task.
#
# Tactile reward annealing (success-triggered): hold scale at 0.175 until the
# running episode success rate first crosses 0.01, then fade 0.175 -> 0.0 over
# 200 PPO iters (nut horizon_length=256 -> 200*256=51200 env control steps).
# This keeps the tactile bonus bootstrapping right up until the policy starts
# to solve the task, then converges it away fast so it finishes on task reward
# alone. Switch ANNEAL_MODE=linear for the old from-step-0 ramp; ANNEAL_STEPS=0
# disables annealing entirely (constant scale).
TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
FORGE_TACTILE_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/exp_taskcompare/nut_scratch/nut_scratch_epoch12.pth \
FORGE_TACTILE_REWARD_SCALE=0.175 \
FORGE_TACTILE_REWARD_SCALE_END=0.0 \
FORGE_TACTILE_REWARD_ANNEAL_MODE=success \
FORGE_TACTILE_REWARD_ANNEAL_SUCCESS_THRESH=0.1 \
FORGE_TACTILE_REWARD_ANNEAL_STEPS=5120 \
FORGE_TACTILE_REWARD_INSTRUCTION="pick up the nut and thread it onto the bolt" \
FORGE_TACTILE_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/Tactile-ReWiND \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-NutThread-PickPlace-Direct-v0 \
    --baseline A_hard_success \
    --headless \
    --seed 0 \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --wandb-entity b11902127-ntu \
    --wandb-project-name tactile-rewind \
    --wandb-name NutThread_PickPlace_baselineTacReward_hard_success_0.175_anneal_-3.5_seed0 \
    agent.params.config.full_experiment_name=NutThread_PickPlace_baselineTacReward_hard_success_0.175_anneal_-3.5_seed0 \
    agent.params.config.save_frequency=20
