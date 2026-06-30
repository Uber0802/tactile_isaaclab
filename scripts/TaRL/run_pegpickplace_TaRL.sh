#!/bin/bash
# Peg pickplace — baseline + tactile reward shaping (TaRL).
source "$(dirname "$0")/_common.sh"

FORGE_TACTILE_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/peg_curriculum_retrain/peg_curr_epoch17.pth \
FORGE_TACTILE_REWARD_SCALE=0.1 \
FORGE_TACTILE_REWARD_INSTRUCTION="grasp peg and insert to another hole" \
FORGE_TACTILE_REWARD_ROOT="$TACTILE_ROOT" \
./isaaclab.sh -p "$TRAIN" \
    --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
    --baseline baseline \
    --headless --num_envs 256 --max_iterations 10000 --enable_cameras --seed 2 \
    $WANDB --wandb-name PegInsert_PickPlace_TaRL_0.1_seed2 \
    agent.params.config.full_experiment_name=PegInsert_PickPlace_TaRL_0.1_seed2 \
    agent.params.config.save_frequency=20
