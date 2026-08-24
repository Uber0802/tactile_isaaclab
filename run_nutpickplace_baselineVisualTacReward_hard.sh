# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# Straight merge of run_nutpickplace_baselineVisualReward_hard.sh and
# run_nutpickplace_baselineTacReward_hard.sh — same baseline (A_hard_success,
# success_threshold -3.5), same env count / iteration budget, both reward heads
# active with the scales each script already used:
#
#   tactile ReWiND (GelSight -> TactileReWiNDTransformer)          x 0.175
#   visual  ReWiND (front_cam -> DINOv2 ViT-B/14 -> ReWiNDTransformer) x 0.175
#
# forge_env._get_rewards adds each as its own `*_progress` entry, so wandb gets
# logs_rew_tactile_progress and logs_rew_visual_progress separately, and each
# anneals on independent state (same success-triggered schedule: hold until the
# episode success EMA crosses 0.1, then fade to 0 over 5120 control steps).
#
# Two things had to be resolved because the sources disagree:
#   * FORGE_SKIP_TACTILE_SENSORS=1 is DROPPED (the visual-only script sets it).
#     The tactile reward model reads GelSight every step, so the sensors must
#     stay in the scene.
#   * seed: visual-only uses 2, tactile-only uses 0. Taking 0 to match the
#     tactile run, since that is the reference this is meant to build on.
TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
FORGE_ENABLE_FRONT_CAM=1 \
FORGE_VISUAL_REWARD_CKPT=/mnt/tank/tactile/Tactile-Reward/ckpt_visual/nut_seed2_multipos/nut_rgb_seed2_multipos_epoch13.pth \
FORGE_VISUAL_REWARD_SCALE=0.15 \
FORGE_VISUAL_REWARD_SCALE_END=0.0 \
FORGE_VISUAL_REWARD_ANNEAL_MODE=success \
FORGE_VISUAL_REWARD_ANNEAL_SUCCESS_THRESH=0.1 \
FORGE_VISUAL_REWARD_ANNEAL_STEPS=5120 \
FORGE_VISUAL_REWARD_INSTRUCTION="pick up the nut and thread it onto the bolt" \
FORGE_VISUAL_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/ReWiND \
FORGE_VISUAL_REWARD_BACKBONE=dinov2_vitb14 \
FORGE_VISUAL_REWARD_DINO_INTERVAL=1 \
FORGE_TACTILE_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/exp_taskcompare/nut_scratch/nut_scratch_epoch12.pth \
FORGE_TACTILE_REWARD_SCALE=0.2 \
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
    --wandb-name NutThread_PickPlace_baselineVisualTacReward_yaw01_tac0.2_vis0.15_seed0 \
    agent.params.config.full_experiment_name=NutThread_PickPlace_baselineVisualTacReward_yaw01_tac0.2_vis0.15_seed0 \
    agent.params.config.save_frequency=100
