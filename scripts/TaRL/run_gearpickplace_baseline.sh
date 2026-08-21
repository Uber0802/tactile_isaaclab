#!/bin/bash
# Gear pickplace — baseline (no tactile): yaw_reward=0 + tighter success.
# The "shaping-insufficient" reference for the TaRL comparison.
source "$(dirname "$0")/_common.sh"

FORGE_SKIP_TACTILE_SENSORS=1 \
./isaaclab.sh -p "$TRAIN" \
    --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
    --baseline baseline \
    --headless --num_envs 256 --max_iterations 10000 \
    agent.params.config.full_experiment_name=GearMesh_PickPlace_baseline \
    agent.params.config.save_frequency=20
