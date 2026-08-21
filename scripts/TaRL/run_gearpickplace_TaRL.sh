#!/bin/bash
# Gear pickplace — baseline + tactile reward shaping (TaRL).
source "$(dirname "$0")/_common.sh"

FORGE_TACTILE_REWARD_CKPT=assets/TactileModel/gear_scratch_epoch18.pth \
FORGE_TACTILE_REWARD_SCALE=0.1 \
FORGE_TACTILE_REWARD_SMOOTH_ALPHA=0.2 \
FORGE_TACTILE_REWARD_INSTRUCTION="pick up the gear and mesh it onto the shaft" \
FORGE_TACTILE_REWARD_ROOT="$TACTILE_ROOT" \
./isaaclab.sh -p "$TRAIN" \
    --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
    --baseline baseline \
    --headless --num_envs 256 --max_iterations 10000 --enable_cameras \
    agent.params.config.full_experiment_name=GearMesh_PickPlace_TaRL_0.1 \
    agent.params.config.save_frequency=20
