# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# Peg pickplace + ReWiND visual reward shaping (RGB → DINOv2 → ReWiNDTransformer).
# Mirrors run_pegpickplace_baselineTacReward_legacy.sh but uses the front_cam RGB
# stream instead of the GelSight tactile force fields, and a visual ReWiND ckpt
# trained on per-episode HDF5 demos (success/fail) via train_reward_human_demos.py.
#
# Tactile reward DISABLED here so we isolate the contribution of visual shaping
# (`FORGE_TACTILE_REWARD_CKPT` left unset → tactile reward off). Pair this with
# run_pegpickplace_baselineA_legacy.sh as the no-shaping baseline.
TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
FORGE_ENABLE_FRONT_CAM=1 \
FORGE_VISUAL_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/ckpt_visual/rewind_pegpickplace_epoch_019.pth \
FORGE_VISUAL_REWARD_SCALE=0.1 \
FORGE_VISUAL_REWARD_INSTRUCTION="grasp peg and insert to another hole" \
FORGE_VISUAL_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/ReWiND \
FORGE_VISUAL_REWARD_BACKBONE=dinov2_vitb14 \
FORGE_VISUAL_REWARD_DINO_INTERVAL=1 \
FORGE_SKIP_TACTILE_SENSORS=1 \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
    --baseline A_legacy \
    --headless \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --wandb-entity b11902127-ntu \
    --wandb-project-name tactile-rewind \
    --wandb-name PegInsert_PickPlace_baselineVisualReward_legacy0.1 \
    agent.params.config.full_experiment_name=PegInsert_PickPlace_baselineVisualReward_legacy0.1 \
    agent.params.config.save_frequency=20
