#!/bin/bash
# Peg pickplace — naive backbone (all peg-specific dense shaping = 0, factory
# base only) + ReWiND visual reward at scale 1.0. Isolates the visual reward's
# contribution. Pair with the plain naive run to read the visual-only delta.
source "$(dirname "$0")/_common.sh"

FORGE_SKIP_TACTILE_SENSORS=1 \
FORGE_ENABLE_FRONT_CAM=1 \
FORGE_VISUAL_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/ckpt_visual/peg_rgb_multipos.pth \
FORGE_VISUAL_REWARD_SCALE=1.0 \
FORGE_VISUAL_REWARD_INSTRUCTION="grasp peg and insert to another hole" \
FORGE_VISUAL_REWARD_ROOT="$VISUAL_ROOT" \
FORGE_VISUAL_REWARD_BACKBONE=dinov2_vitb14 \
FORGE_VISUAL_REWARD_DINO_INTERVAL=1 \
./isaaclab.sh -p "$TRAIN" \
    --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
    --baseline naive \
    --headless --num_envs 256 --max_iterations 10000 --enable_cameras \
    agent.params.config.full_experiment_name=PegInsert_PickPlace_VisualReward_naive1.0 \
    agent.params.config.save_frequency=20
