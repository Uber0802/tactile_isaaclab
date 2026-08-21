#!/bin/bash
# Peg pickplace — baseline + tactile reward shaping (TaRL).
source "$(dirname "$0")/_common.sh"

./isaaclab.sh -p "$TRAIN" \
    "env.tactile_reward.ckpt=assets/TactileModel/peg_curr_epoch17.pth" \
    "env.tactile_reward.scale=0.1" \
    "env.tactile_reward.instruction=grasp peg and insert to another hole" \
    "env.tactile_reward.rewind_root=$TACTILE_ROOT" \
    --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
    --baseline baseline \
    --headless --num_envs 256 --max_iterations 10000 --enable_cameras --seed 2 \
    agent.params.config.full_experiment_name=PegInsert_PickPlace_TaRL_0.1_seed2 \
    agent.params.config.save_frequency=20
