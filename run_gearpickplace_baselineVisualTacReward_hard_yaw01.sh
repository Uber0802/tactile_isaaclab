# Per-machine local cache so two hosts sharing the same NAS $HOME don't fight
# over Omniverse / Triton / Torch caches (NFS lock contention slows training to a crawl).
CACHE_DIR="/tmp/${USER}_${HOSTNAME%%.*}_isaac"
mkdir -p "$CACHE_DIR/tmp" "$CACHE_DIR/cache/ov" "$CACHE_DIR/torch/triton" "$CACHE_DIR/torch/inductor"

# Gear pickplace, baseline A_hard_success_yaw01 + BOTH reward heads:
#   - tactile ReWiND (GelSight force field -> TactileReWiNDTransformer)  x 0.2
#   - visual  ReWiND (front_cam RGB -> DINOv2 ViT-B/14 -> ReWiNDTransformer) x 0.1
# forge_env._get_rewards adds each as its own `*_progress` entry in rew_dict, so
# both appear separately in wandb (logs_rew_tactile_progress / _visual_progress)
# and each anneals on its own independent state.
#
# Scale choice: the tactile head keeps the 0.2 that the tactile-only run
# (run_gearpickplace_baselineTacReward_hard_yaw01.sh) is known to crack with at
# ~ep300, and the visual head is added at HALF that (0.1) so the combined
# shaping budget is 0.3 rather than 0.4 -- this is meant to test whether visual
# adds anything on top of tactile, not to re-tune the tactile recipe. Same
# asymmetric-scale convention as run_pegpickplace_baselineVisualTacReward.sh.
#
# TACTILE SENSORS STAY ON (no FORGE_SKIP_TACTILE_SENSORS, unlike the
# visual-only script) -- the tactile reward model reads GelSight every step.
#
# Annealing: both heads hold their scale until the running episode success EMA
# first crosses 0.1, then fade to 0 over 200 PPO iters (gear horizon_length=128
# -> 200*128 = 25600 env control steps). Independent trigger state per head,
# but with a shared success signal they fire together in practice.
#
# COST: this is the most expensive gear config -- GelSight sensors AND the front
# camera AND the ViT. Measured per-control-step at 256 envs: 0.43 s physics/PPO
# + 0.25 s GelSight + 0.22 s camera render + 0.10 s DINOv2(bf16) ~= 1.0 s,
# i.e. roughly 257 fps vs 598 for bare baselineA. To trade a little staleness
# for speed, add `FORGE_VISUAL_REWARD_DINO_INTERVAL=2` and
# `env.sim.render_interval=16` (~+25% fps); left at 1/8 here so the numbers stay
# comparable with the existing VisualReward-only runs.
#
# CAVEAT on the visual ckpt: held-out eval of gear_rgb_seed2_multipos separates
# success from failure well overall (epoch13 gap +0.381, AUC 0.913) but that is
# mostly driven by the ep_20/ep_200 random-policy failures. Restricted to
# failures from competent checkpoints (ckpt >= 400) it drops to gap +0.303 /
# AUC 0.847 -- and epoch19 is markedly worse there (+0.219 / 0.744), which is
# why this script uses epoch13 rather than epoch19. The visual-only runs also
# showed logs_rew_visual_progress stuck near 0.02 during RL, i.e. the model
# reads the on-policy states as out-of-distribution. Treat the visual term here
# as unproven; the tactile term is the one carrying the run.
TMPDIR="$CACHE_DIR/tmp" \
XDG_CACHE_HOME="$CACHE_DIR/cache" \
OMNI_KIT_CACHE_DIR="$CACHE_DIR/cache/ov" \
OV_CACHE_DIRECTORY="$CACHE_DIR/cache/ov" \
TORCH_HOME="$CACHE_DIR/torch" \
TRITON_CACHE_DIR="$CACHE_DIR/torch/triton" \
TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torch/inductor" \
FORGE_ENABLE_FRONT_CAM=1 \
FORGE_VISUAL_REWARD_CKPT=/mnt/lab-tank/tactile/Tactile-Reward/ckpt_visual/gear_seed2_multipos/gear_rgb_seed2_multipos_epoch13.pth \
FORGE_VISUAL_REWARD_SCALE=0.175 \
FORGE_VISUAL_REWARD_SCALE_END=0.0 \
FORGE_VISUAL_REWARD_ANNEAL_MODE=success \
FORGE_VISUAL_REWARD_ANNEAL_SUCCESS_THRESH=0.1 \
FORGE_VISUAL_REWARD_ANNEAL_STEPS=25600 \
FORGE_VISUAL_REWARD_INSTRUCTION="pick up the gear and mesh it onto the shaft" \
FORGE_VISUAL_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/ReWiND \
FORGE_VISUAL_REWARD_BACKBONE=dinov2_vitb14 \
FORGE_VISUAL_REWARD_DINO_INTERVAL=1 \
FORGE_TACTILE_REWARD_CKPT=/mnt/lab-tank/tactile/Tactile-Reward/exp_gear_seed2/gear_seed2_scratch/gear_seed2_scratch_epoch29.pth \
FORGE_TACTILE_REWARD_SCALE=0.175 \
FORGE_TACTILE_REWARD_SCALE_END=0.0 \
FORGE_TACTILE_REWARD_ANNEAL_MODE=success \
FORGE_TACTILE_REWARD_ANNEAL_SUCCESS_THRESH=0.1 \
FORGE_TACTILE_REWARD_ANNEAL_STEPS=25600 \
FORGE_TACTILE_REWARD_INSTRUCTION="pick up the gear and mesh it onto the shaft" \
FORGE_TACTILE_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/Tactile-ReWiND \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
    --baseline A_hard_success_yaw01 \
    --headless \
    --seed 2 \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --wandb-entity b11902127-ntu \
    --wandb-project-name tactile-rewind \
    --wandb-name GearMesh_PickPlace_baselineVisualTacReward_yaw01_tac0.175_vis0.175_seed2 \
    agent.params.config.full_experiment_name=GearMesh_PickPlace_baselineVisualTacReward_yaw01_tac0.175_vis0.175_seed2 \
    agent.params.config.save_frequency=100
