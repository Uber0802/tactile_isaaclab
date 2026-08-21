#!/bin/bash
# Nut pickplace — baseline (no tactile): yaw_reward=0 + wider pose noise +
# tighter success. The "shaping-insufficient" reference for the TaRL comparison.
source "$(dirname "$0")/_common.sh"

FORGE_SKIP_TACTILE_SENSORS=1 \
./isaaclab.sh -p "$TRAIN" \
    --task Isaac-Forge-NutThread-PickPlace-Direct-v0 \
    --baseline baseline \
    --headless --num_envs 256 --max_iterations 10000 --seed 1 \
    agent.params.config.full_experiment_name=NutThread_PickPlace_baseline_seed1 \
    agent.params.config.save_frequency=500
