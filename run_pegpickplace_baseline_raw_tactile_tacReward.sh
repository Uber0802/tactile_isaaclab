# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# Baseline raw_tactile_reward for peg pickplace: everything raw_tactile has
# (A_legacy reward shaping — half-strength transport bridges, 1cm descent gate —
# plus a frozen AE tactile latent appended to actor obs AND critic state) with
# the Tactile-ReWiND progress reward bonus added on top. Pair with
# run_pegpickplace_baseline_raw_tactile.sh: same encoder ckpt / num_envs /
# save_frequency, so the only delta is the reward model. That isolates
# "tactile as reward" on top of "tactile as state input" — the two pathways are
# independent (AE latent → policy, reward model → return).
#
# The baseline hard-requires FORGE_TACTILE_REWARD_CKPT: without it the reward
# model stays silently disabled and the run would just duplicate raw_tactile.
#
# Pretrain the encoder first (see external/third-party/Tactile-ReWiND/train_tactile_ae.py):
#   python train_tactile_ae.py \
#       --data_dir /mnt/tank/tactile/tactile_dataset/pegpickplace_paired \
#       --out_dir  /mnt/tank/uber/Tactile-Reward/tactile_ae_peg \
#       --per_hand_dim 64 --epochs 40
# FORGE_TACTILE_ENCODER_DIM must equal 2*per_hand_dim (startup asserts this).
#
# Note: no --enable_cameras — both the encoder and the reward model read the
# (40, 25, C) GelSight force fields (SDF queries, no RTX renderer), not the
# tactile RGB, so this runs on compute-only cloud GPUs.
TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
FORGE_TACTILE_ENCODER_CKPT=/mnt/scratch/kimnai/research/tarl/ae_16/ae_best.pth \
FORGE_TACTILE_ENCODER_DIM=32 \
FORGE_TACTILE_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/peg_curriculum_retrain/peg_curr_epoch17.pth \
FORGE_TACTILE_REWARD_SCALE=0.1 \
FORGE_TACTILE_REWARD_INSTRUCTION="grasp peg and insert to another hole" \
FORGE_TACTILE_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/Tactile-ReWiND \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
    --baseline raw_tactile \
    --headless \
    --num_envs 128 \
    --seed 0 \
    --max_iterations 5000 \
    --track \
    --wandb-entity b06902045-national-taiwan-university \
    --wandb-project-name tactile-rewind \
    --wandb-name PegInsert_PickPlace_baseline_raw_tactile32_seed0_tacReward \
    --checkpoint /mnt/home/kimnai/research/tactile_isaaclab/logs/rl_games/ForgePickPlace/PegInsert_PickPlace_baseline_raw_tactile32_seed0_tacReward/nn/ForgePickPlace.pth \
    agent.params.config.full_experiment_name=PegInsert_PickPlace_baseline_raw_tactile32_seed0_tacReward \
    agent.params.config.save_frequency=100
