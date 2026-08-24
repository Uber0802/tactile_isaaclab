#!/bin/bash
# Gear pickplace — baseline + tactile as state + tactile reward shaping.

source "$(dirname "$0")/_common.sh"

./isaaclab.sh -p "$TRAIN" \
    "env.tactile_reward.ckpt=assets/TactileModel/gear_seed2_scratch_epoch29.pth" \
    "env.tactile_reward.scale=0.2" \
    "env.tactile_reward.scale_end=0.0" \
    "env.tactile_reward.anneal_mode=success" \
    "env.tactile_reward.anneal_success_thresh=0.1" \
    "env.tactile_reward.anneal_steps=25600" \
    "env.tactile_reward.instruction=pick up the gear and mesh it onto the shaft" \
    "env.tactile_reward.rewind_root=$TACTILE_ROOT" \
    "env.tactile_encoder.ckpt=assets/TactileModel/gear_ae_best.pth" \
    "env.tactile_encoder.dim=32" \
    "env.tactile_encoder.root=$TACTILE_ROOT" \
    --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
    --baseline tactile_state \
    --headless --seed 0 --num_envs 256 --max_iterations 10000 \
    agent.params.config.full_experiment_name=GearMesh_PickPlace_tactile_state_TaRL \
    agent.params.config.save_frequency=20 \
