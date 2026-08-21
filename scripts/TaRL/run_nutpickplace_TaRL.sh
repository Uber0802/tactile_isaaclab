#!/bin/bash
# Nut pickplace — baseline + tactile reward shaping (TaRL).
# Annealing: fade scale 0.175 -> 0.0 over the first 1000 PPO iters
# (horizon_length=256 -> 1000*256=256000 env control steps), then hold at 0.
# Bootstraps early learning with tactile, then converges on task reward alone.
# Set ANNEAL_STEPS=0 to disable (constant scale).
source "$(dirname "$0")/_common.sh"

FORGE_TACTILE_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/exp_taskcompare/nut_scratch/nut_scratch_epoch12.pth \
FORGE_TACTILE_REWARD_SCALE=0.175 \
FORGE_TACTILE_REWARD_SCALE_END=0.0 \
FORGE_TACTILE_REWARD_ANNEAL_STEPS=256000 \
FORGE_TACTILE_REWARD_INSTRUCTION="pick up the nut and thread it onto the bolt" \
FORGE_TACTILE_REWARD_ROOT="$TACTILE_ROOT" \
./isaaclab.sh -p "$TRAIN" \
    --task Isaac-Forge-NutThread-PickPlace-Direct-v0 \
    --baseline baseline \
    --headless --num_envs 256 --max_iterations 10000 --enable_cameras --seed 1 \
    agent.params.config.full_experiment_name=NutThread_PickPlace_TaRL_0.175_anneal1000it_seed1 \
    agent.params.config.save_frequency=20
