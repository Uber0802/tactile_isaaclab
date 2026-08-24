#!/bin/bash
# Gear pickplace — baseline + tactile as state + tactile reward shaping.

source "$(dirname "$0")/_common.sh"

./isaaclab.sh -p "$TRAIN" \
    "env.tactile_encoder.ckpt=assets/TactileModel/gear_ae_best.pth" \
    "env.tactile_encoder.dim=32" \
    "env.tactile_encoder.root=$TACTILE_ROOT" \
    --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
    --baseline tactile_state \
    --headless --seed 0 --num_envs 256 --max_iterations 10000 \
    agent.params.config.full_experiment_name=GearMesh_PickPlace_tactile_state \
    agent.params.config.save_frequency=20 \
    agent.params.config.entropy_coef=0.005 \
    agent.params.network.space.continuous.sigma_init.val=0.5
