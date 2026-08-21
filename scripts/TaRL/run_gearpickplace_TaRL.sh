#!/bin/bash
# Gear pickplace — baseline + tactile reward shaping (TaRL).
source "$(dirname "$0")/_common.sh"

./isaaclab.sh -p "$TRAIN" \
    "env.tactile_reward.ckpt=assets/TactileModel/gear_scratch_epoch18.pth" \
    "env.tactile_reward.scale=0.1" \
    "env.tactile_reward.smooth_alpha=0.2" \
    "env.tactile_reward.instruction=pick up the gear and mesh it onto the shaft" \
    "env.tactile_reward.rewind_root=$TACTILE_ROOT" \
    --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
    --baseline baseline \
    --headless --num_envs 128 --max_iterations 10000 --enable_cameras \
    agent.params.config.full_experiment_name=GearMesh_PickPlace_TaRL_0.1 \
    agent.params.config.save_frequency=20
