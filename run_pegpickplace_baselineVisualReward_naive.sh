# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# Peg pickplace + Visual reward (multipos ckpt) on the NAIVE reward backbone:
#   - All peg-specific dense shaping (approach / lift / xy / z / descent) = 0
#   - Factory base only: kp_*, curr_engaged, curr_success
#   - Plus ReWiND visual reward at scale 1.0
# Isolates the visual reward's contribution — no hand-crafted shaping competing.
#
# Compare against run_pegpickplace_baselineA_naive.sh (no visual) to read the
# Δ from visual reward alone.
TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
FORGE_SKIP_TACTILE_SENSORS=1 \
FORGE_ENABLE_FRONT_CAM=1 \
FORGE_VISUAL_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/ckpt_visual/peg_rgb_multipos.pth \
FORGE_VISUAL_REWARD_SCALE=1.0 \
FORGE_VISUAL_REWARD_INSTRUCTION="grasp peg and insert to another hole" \
FORGE_VISUAL_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/ReWiND \
FORGE_VISUAL_REWARD_BACKBONE=dinov2_vitb14 \
FORGE_VISUAL_REWARD_DINO_INTERVAL=1 \
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
    --wandb-name PegInsert_PickPlace_baselineVisualReward_naive1.0 \
    agent.params.config.full_experiment_name=PegInsert_PickPlace_baselineVisualReward_naive1.0 \
    agent.params.config.save_frequency=20
