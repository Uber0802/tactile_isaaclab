# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# Baseline raw_tactile for peg pickplace: A_legacy reward shaping (half-strength
# transport bridges, 1cm descent gate) + a frozen tactile latent appended to
# actor obs AND critic state. The latent comes from an autoencoder-pretrained
# TactileCNNEncoder (reconstruction objective — no reward-model info, unlike
# B2's progress-trained encoder). Pair with run_pegpickplace_baselineA_legacy.sh:
# same seed / num_envs / save_frequency, so the only delta is the tactile input.
#
# Pretrain the encoder first (see external/third-party/Tactile-ReWiND/train_tactile_ae.py):
#   python train_tactile_ae.py \
#       --data_dir /mnt/tank/tactile/tactile_dataset/pegpickplace_paired \
#       --out_dir  /mnt/tank/uber/Tactile-Reward/tactile_ae_peg \
#       --per_hand_dim 64 --epochs 40
# FORGE_TACTILE_ENCODER_DIM must equal 2*per_hand_dim (startup asserts this).
#
# Note: no --enable_cameras — the baseline switches the GelSight sensors to
# force-field-only (SDF queries, no RTX renderer), so this runs on
# compute-only cloud GPUs.
TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
FORGE_TACTILE_ENCODER_CKPT=/mnt/scratch/kimnai/research/tarl/gear_ae_16/ae_best.pth \
FORGE_TACTILE_ENCODER_DIM=32 \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
    --baseline raw_tactile \
    --headless \
    --num_envs 256 \
    --seed 0 \
    --max_iterations 3500 \
    --track \
    --wandb-entity b06902045-national-taiwan-university \
    --wandb-project-name tactile-rewind \
    --wandb-name GearMesh_PickPlace_baseline_tactile_state_seed0 \
    agent.params.config.full_experiment_name=GearMesh_PickPlace_baseline_tactile_state_seed0 \
    agent.params.config.save_frequency=20 \
    agent.params.config.entropy_coef=0.005 \
    agent.params.network.space.continuous.sigma_init.val=0.5
