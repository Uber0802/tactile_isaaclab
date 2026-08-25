#!/bin/bash
# Gear pickplace — BOTH reward heads on the baseline shaping:
#   tactile ReWiND (GelSight force field)            x 0.175
#   visual  ReWiND (front_cam -> DINOv2 -> ReWiND)   x 0.175
# Each head anneals on its own independent success-triggered state (hold until
# success EMA crosses 0.1, then fade to 0 over 25600 control steps); with a
# shared success signal they fire together in practice. Tactile sensors STAY ON
# (the tactile model reads GelSight every step). Most expensive gear config:
# GelSight + front camera + ViT per control step.
source "$(dirname "$0")/_common.sh"

FORGE_ENABLE_FRONT_CAM=1 \
./isaaclab.sh -p "$TRAIN" \
    "env.visual_reward.ckpt=assets/TactileModel/gear_rgb_seed2_multipos_epoch13.pth" \
    "env.visual_reward.scale=0.175" \
    "env.visual_reward.scale_end=0.0" \
    "env.visual_reward.anneal_mode=success" \
    "env.visual_reward.anneal_success_thresh=0.1" \
    "env.visual_reward.anneal_steps=25600" \
    "env.visual_reward.instruction=pick up the gear and mesh it onto the shaft" \
    "env.visual_reward.root=$VISUAL_ROOT" \
    "env.visual_reward.backbone=dinov2_vitb14" \
    "env.visual_reward.dino_interval=1" \
    "env.tactile_reward.ckpt=assets/TactileModel/gear_seed2_scratch_epoch29.pth" \
    "env.tactile_reward.scale=0.175" \
    "env.tactile_reward.scale_end=0.0" \
    "env.tactile_reward.anneal_mode=success" \
    "env.tactile_reward.anneal_success_thresh=0.1" \
    "env.tactile_reward.anneal_steps=25600" \
    "env.tactile_reward.instruction=pick up the gear and mesh it onto the shaft" \
    "env.tactile_reward.rewind_root=$TACTILE_ROOT" \
    --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
    --baseline baseline \
    --headless --seed 2 --num_envs 256 --max_iterations 10000 --enable_cameras \
    agent.params.config.full_experiment_name=GearMesh_PickPlace_VisualTacReward_tac0.175_vis0.175_seed2 \
    agent.params.config.save_frequency=100
