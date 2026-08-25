#!/bin/bash
# Nut pickplace — RGB twin of run_nutpickplace_TaRL.sh: identical baseline
# shaping and success-triggered anneal, but the dense bonus comes from the
# ReWiND VISUAL reward model, so the pair isolates modality. Tactile OFF.
# Visual ckpt: nut_rgb_seed2_multipos_epoch13 (DINOv2 ViT-B/14 -> ReWiND),
# trained on nutpickplace_curriculum_seed2_paired_multipos.
source "$(dirname "$0")/_common.sh"

FORGE_SKIP_TACTILE_SENSORS=1 \
FORGE_ENABLE_FRONT_CAM=1 \
./isaaclab.sh -p "$TRAIN" \
    "env.visual_reward.ckpt=assets/TactileModel/nut_rgb_seed2_multipos_epoch13.pth" \
    "env.visual_reward.scale=0.175" \
    "env.visual_reward.scale_end=0.0" \
    "env.visual_reward.anneal_mode=success" \
    "env.visual_reward.anneal_success_thresh=0.1" \
    "env.visual_reward.anneal_steps=5120" \
    "env.visual_reward.instruction=pick up the nut and thread it onto the bolt" \
    "env.visual_reward.root=$VISUAL_ROOT" \
    "env.visual_reward.backbone=dinov2_vitb14" \
    "env.visual_reward.dino_interval=1" \
    --task Isaac-Forge-NutThread-PickPlace-Direct-v0 \
    --baseline baseline \
    --headless --seed 2 --num_envs 256 --max_iterations 10000 --enable_cameras \
    agent.params.config.full_experiment_name=NutThread_PickPlace_VisualReward_0.175_seed2 \
    agent.params.config.save_frequency=100
