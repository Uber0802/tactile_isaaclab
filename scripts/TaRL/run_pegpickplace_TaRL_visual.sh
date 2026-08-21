#!/bin/bash
# Peg pickplace — naive backbone + BOTH visual (x1.0) and tactile (x0.1) ReWiND
# reward heads side by side. Tactile sensors KEPT (the tactile model needs
# GelSight every step). Ablate against the visual-only and naive runs.
source "$(dirname "$0")/_common.sh"

FORGE_ENABLE_FRONT_CAM=1 \
./isaaclab.sh -p "$TRAIN" \
    "env.visual_reward.ckpt=/mnt/tank/uber/Tactile-Reward/ckpt_visual/peg_rgb_multipos.pth" \
    "env.visual_reward.scale=1.0" \
    "env.visual_reward.instruction=grasp peg and insert to another hole" \
    "env.visual_reward.root=$VISUAL_ROOT" \
    "env.visual_reward.backbone=dinov2_vitb14" \
    "env.visual_reward.dino_interval=4" \
    "env.tactile_reward.ckpt=/mnt/tank/uber/Tactile-Reward/peg_curriculum_retrain/peg_curr_epoch17.pth" \
    "env.tactile_reward.scale=0.1" \
    "env.tactile_reward.instruction=grasp peg and insert to another hole" \
    "env.tactile_reward.rewind_root=$TACTILE_ROOT" \
    --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
    --baseline naive \
    --headless --num_envs 256 --max_iterations 10000 --enable_cameras \
    agent.params.config.full_experiment_name=PegInsert_PickPlace_VisualTacReward_naive \
    agent.params.config.save_frequency=20
