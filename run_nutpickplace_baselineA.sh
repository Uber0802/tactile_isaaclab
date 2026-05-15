# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# Baseline A (no tactile in obs/state) + tactile force-field dump for downstream
# model training. Two hydra overrides:
#   full_experiment_name=NutThread_PickPlace_baselineA  → ckpts go to
#     logs/rl_games/ForgeNutPickPlace/NutThread_PickPlace_baselineA/nn/
#     instead of the default `test/` dir which gets clobbered by B/B2/single_pos.
#   save_frequency=20  → ckpt every 20 epochs (vs default 100) so curriculum
#     rollout has fine-grained skill-level snapshots.
TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-NutThread-PickPlace-Direct-v0 \
    --baseline A \
    --headless \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --wandb-entity b11902127-ntu \
    --wandb-project-name tactile-rewind \
    --wandb-name NutThread_PickPlace_baselineA \
    agent.params.config.full_experiment_name=NutThread_PickPlace_baselineA \
    agent.params.config.save_frequency=20
