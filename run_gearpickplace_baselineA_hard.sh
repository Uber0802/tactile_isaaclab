# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# Baseline A_hard: same obs/state as A but yaw_reward=0 and wider initial pose
# randomization (see _apply_baseline_A_hard in forge_gearpickplace_env_cfg.py).
# Acts as the "shaping-insufficient" reference for evaluating whether tactile
# reward shaping (run_gearpickplace_baselineTacReward_hard.sh) can fill the gap
# left by the cut yaw signal + wider exploration requirement.
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
    --baseline A_hard_success \
    --headless \
    --num_envs 256 \
    --max_iterations 10000 \
    --track \
    --wandb-entity b11902127-ntu \
    --wandb-project-name tactile-rewind \
    --wandb-name GearMesh_PickPlace_baselineA_hard_success_new \
    agent.params.config.full_experiment_name=GearMesh_PickPlace_baselineA_hard_success \
    agent.params.config.save_frequency=20
