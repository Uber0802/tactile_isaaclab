# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# RGB twin of run_gearpickplace_baselineTacReward_hard_yaw01.sh: identical
# baseline (A_hard_success_yaw01), seed, env count and iteration budget — the
# dense shaping bonus comes from the ReWiND VISUAL reward model instead of the
# tactile one, so the pair isolates modality.
#
#   tactile: gear_seed2_scratch_epoch29.pth   (T,40,25,3) force field
#   visual : gear_rgb_seed2_multipos_epoch19  DINOv2 ViT-B/14 -> ReWiNDTransformer
#            trained on gearpickplace_curriculum_seed2_paired_multipos
#            (832 episodes; held-out gap +0.344 at epoch 19, best was +0.381 at
#            epoch 9 — swap to _epoch9.pth if you prefer the peak-gap ckpt).
#
# TACTILE PERCEPTION IS OFF (FORGE_SKIP_TACTILE_SENSORS=1). The GelSight sensors
# cost real sim time and nothing here reads them: baseline A_hard_success_yaw01
# carries no tactile obs, and the reward now comes from the front camera.
#
# Annealing matches the tactile twin exactly: hold scale at 0.2 until the running
# episode success rate (EMA) first crosses 0.1, then fade 0.2 -> 0.0 over 200 PPO
# iters (gear horizon_length=128 -> 200*128=25600 env control steps). The visual
# FORGE_VISUAL_REWARD_ANNEAL_* hooks mirror the tactile ones in forge_env
# (_compute_visual_reward); ANNEAL_STEPS=0 disables for a constant scale.
TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
FORGE_SKIP_TACTILE_SENSORS=1 \
FORGE_ENABLE_FRONT_CAM=1 \
FORGE_VISUAL_REWARD_CKPT=/mnt/lab-tank/tactile/Tactile-Reward/ckpt_visual/gear_seed2_multipos/gear_rgb_seed2_multipos_epoch19.pth \
FORGE_VISUAL_REWARD_SCALE=0.2 \
FORGE_VISUAL_REWARD_SCALE_END=0.0 \
FORGE_VISUAL_REWARD_ANNEAL_MODE=success \
FORGE_VISUAL_REWARD_ANNEAL_SUCCESS_THRESH=0.1 \
FORGE_VISUAL_REWARD_ANNEAL_STEPS=25600 \
FORGE_VISUAL_REWARD_INSTRUCTION="pick up the gear and mesh it onto the shaft" \
FORGE_VISUAL_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/ReWiND \
FORGE_VISUAL_REWARD_BACKBONE=dinov2_vitb14 \
FORGE_VISUAL_REWARD_DINO_INTERVAL=1 \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
    --baseline A_hard_success_yaw01 \
    --headless \
    --seed 1 \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --wandb-entity b11902127-ntu \
    --wandb-project-name tactile-rewind \
    --wandb-name GearMesh_PickPlace_baselineVisualReward_yaw01_0.2_seed1 \
    agent.params.config.full_experiment_name=GearMesh_PickPlace_baselineVisualReward_yaw01_0.2_seed1 \
    agent.params.config.save_frequency=20
