#!/bin/bash
# Nut pickplace — BOTH reward heads on the baseline shaping:
#   tactile ReWiND (GelSight force field)            x 0.2
#   visual  ReWiND (front_cam -> DINOv2 -> ReWiND)   x 0.15
# Independent success-triggered anneal per head (hold until success EMA crosses
# 0.1, then fade to 0 over 5120 control steps). Tactile sensors STAY ON.
source "$(dirname "$0")/_common.sh"

FORGE_ENABLE_FRONT_CAM=1 \
./isaaclab.sh -p "$TRAIN" \
    "env.visual_reward.ckpt=assets/TactileModel/nut_rgb_seed2_multipos_epoch13.pth" \
    "env.visual_reward.scale=0.15" \
    "env.visual_reward.scale_end=0.0" \
    "env.visual_reward.anneal_mode=success" \
    "env.visual_reward.anneal_success_thresh=0.1" \
    "env.visual_reward.anneal_steps=5120" \
    "env.visual_reward.instruction=pick up the nut and thread it onto the bolt" \
    "env.visual_reward.root=$VISUAL_ROOT" \
    "env.visual_reward.backbone=dinov2_vitb14" \
    "env.visual_reward.dino_interval=1" \
    "env.tactile_reward.ckpt=assets/TactileModel/nut_scratch_epoch12.pth" \
    "env.tactile_reward.scale=0.2" \
    "env.tactile_reward.scale_end=0.0" \
    "env.tactile_reward.anneal_mode=success" \
    "env.tactile_reward.anneal_success_thresh=0.1" \
    "env.tactile_reward.anneal_steps=5120" \
    "env.tactile_reward.instruction=pick up the nut and thread it onto the bolt" \
    "env.tactile_reward.rewind_root=$TACTILE_ROOT" \
    --task Isaac-Forge-NutThread-PickPlace-Direct-v0 \
    --baseline baseline \
    --headless --seed 0 --num_envs 256 --max_iterations 10000 --enable_cameras \
    agent.params.config.full_experiment_name=NutThread_PickPlace_VisualTacReward_tac0.2_vis0.15_seed0 \
    agent.params.config.save_frequency=100
