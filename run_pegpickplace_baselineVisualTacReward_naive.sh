# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# Peg pickplace + BOTH visual ReWiND + tactile ReWiND on the NAIVE reward
# backbone (all peg-specific dense shaping = 0). Combined setup:
#   - Visual reward  (RGB front_cam → DINOv2 → ReWiNDTransformer)        × 1.0
#       ckpt: peg_rgb_multipos.pth (multi-pos trained, matches RL randomization)
#   - Tactile reward (GelSight → TactileReWiNDTransformer)               × 0.1
#       ckpt: peg_curr_epoch17.pth (curriculum-retrained)
#   - Baseline A_naive — all peg dense shaping (approach/lift/xy/z/descent) = 0
#   - Factory base only: kp_*, curr_engaged, curr_success
# Two reward heads run side-by-side; each adds its own `*_progress` term to
# rew_dict. Tactile sensors are KEPT (no FORGE_SKIP_TACTILE_SENSORS) because
# the tactile reward model needs GelSight readings every step.
#
# Pair with run_pegpickplace_baselineVisualReward_naive.sh (visual only) and
# run_pegpickplace_baselineA_naive.sh (no shaping) to ablate each head.
TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
FORGE_ENABLE_FRONT_CAM=1 \
FORGE_VISUAL_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/ckpt_visual/peg_rgb_multipos.pth \
FORGE_VISUAL_REWARD_SCALE=1.0 \
FORGE_VISUAL_REWARD_INSTRUCTION="grasp peg and insert to another hole" \
FORGE_VISUAL_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/ReWiND \
FORGE_VISUAL_REWARD_BACKBONE=dinov2_vitb14 \
FORGE_VISUAL_REWARD_DINO_INTERVAL=4 \
FORGE_TACTILE_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/peg_curriculum_retrain/peg_curr_epoch17.pth \
FORGE_TACTILE_REWARD_SCALE=0.1 \
FORGE_TACTILE_REWARD_INSTRUCTION="grasp peg and insert to another hole" \
FORGE_TACTILE_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/Tactile-ReWiND \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
    --baseline A_naive \
    --headless \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --wandb-entity b11902127-ntu \
    --wandb-project-name tactile-rewind \
    --wandb-name PegInsert_PickPlace_baselineVisualTacReward_naive \
    agent.params.config.full_experiment_name=PegInsert_PickPlace_baselineVisualTacReward_naive \
    agent.params.config.save_frequency=20
