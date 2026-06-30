# Shared setup sourced by every TaRL run script.
#   source "$(dirname "$0")/_common.sh"
# Run the scripts from the repo root (they call ./isaaclab.sh with relative paths).

# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"
export TMPDIR="$CACHE_DIR/tmp"
export XDG_CACHE_HOME="$CACHE_DIR/cache"
export OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov"
export OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov"
export TORCH_HOME="$CACHE_DIR/torch"
export TRITON_CACHE_DIR="$CACHE_DIR/torch/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor"

# Common paths / args reused across scripts.
TRAIN="scripts/reinforcement_learning/rl_games/train.py"
WANDB="--track --wandb-entity b11902127-ntu --wandb-project-name tactile-rewind"
TACTILE_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/Tactile-ReWiND
VISUAL_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/ReWiND
