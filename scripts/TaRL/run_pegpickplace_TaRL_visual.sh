#!/bin/bash
# Peg pickplace — naive backbone + BOTH visual (x1.0) and tactile (x0.1) ReWiND
# reward heads side by side. Tactile sensors KEPT (the tactile model needs
# GelSight every step). Ablate against the visual-only and naive runs.
source "$(dirname "$0")/_common.sh"

FORGE_ENABLE_FRONT_CAM=1 \
FORGE_VISUAL_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/ckpt_visual/peg_rgb_multipos.pth \
FORGE_VISUAL_REWARD_SCALE=1.0 \
FORGE_VISUAL_REWARD_INSTRUCTION="grasp peg and insert to another hole" \
FORGE_VISUAL_REWARD_ROOT="$VISUAL_ROOT" \
FORGE_VISUAL_REWARD_BACKBONE=dinov2_vitb14 \
FORGE_VISUAL_REWARD_DINO_INTERVAL=4 \
FORGE_TACTILE_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/peg_curriculum_retrain/peg_curr_epoch17.pth \
FORGE_TACTILE_REWARD_SCALE=0.1 \
FORGE_TACTILE_REWARD_INSTRUCTION="grasp peg and insert to another hole" \
FORGE_TACTILE_REWARD_ROOT="$TACTILE_ROOT" \
./isaaclab.sh -p "$TRAIN" \
    --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
    --baseline naive \
    --headless --num_envs 256 --max_iterations 10000 --enable_cameras \
    agent.params.config.full_experiment_name=PegInsert_PickPlace_VisualTacReward_naive \
    agent.params.config.save_frequency=20
