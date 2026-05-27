# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# Baseline A_hard_success_yaw01: A_hard_success ablations (yaw_reward=0, gear
# yaw fixed, success_threshold=-0.3) but with a TINY yaw_reward bump back to
# 0.1 (10% of original 1.0). Mild hint so baseline can slowly learn yaw
# without making yaw_reward dominate. Pair with the strict yaw_reward=0
# `run_gearpickplace_baselineA_hard.sh` to ablate the effect of the hint.
TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
FORGE_SKIP_TACTILE_SENSORS=1 \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
    --baseline A_hard_success_yaw01 \
    --headless \
    --num_envs 256 \
    --max_iterations 10000 \
    --track \
    --wandb-entity b11902127-ntu \
    --wandb-project-name tactile-rewind \
    --wandb-name GearMesh_PickPlace_baselineA_hard_success_yaw01_yawinput \
    agent.params.config.full_experiment_name=GearMesh_PickPlace_baselineA_hard_success_yaw01_yawinput \
    agent.params.config.save_frequency=20 \
    agent.params.config.entropy_coef=0.02 \
    agent.params.network.space.continuous.sigma_init.val=0.5
