#!/bin/bash
# Peg pickplace — baseline (no tactile): half transport bridges + tight 1cm
# descent gate. Solvable but slow, leaving headroom for the TaRL comparison.
source "$(dirname "$0")/_common.sh"

./isaaclab.sh -p "$TRAIN" \
    --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
    --baseline baseline \
    --headless --num_envs 128 --max_iterations 10000 --enable_cameras --seed 2 \
    agent.params.config.full_experiment_name=PegInsert_PickPlace_baseline_seed2 \
    agent.params.config.save_frequency=200
