# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# Baseline A_legacy for peg pickplace: reproduces the May 15 commit ed32dd8
# reward shaping (no coarse XY bridge, no Z bridge, tight 1cm descent gate).
# This is the "right difficulty" regime per the q50f4175 run result —
# baselineA can eventually solve the task in ~16h but with room for tactile
# reward shaping to demonstrate measurable improvement. Pair with
# run_pegpickplace_baselineTacReward_legacy.sh for the +tactile comparison.
TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
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
    --wandb-name PegInsert_PickPlace_baselineA_legacy \
    agent.params.config.full_experiment_name=PegInsert_PickPlace_baselineA_legacy \
    agent.params.config.save_frequency=20
