#!/bin/bash
# Nut pickplace — baseline + tactile reward shaping (TaRL).
# Annealing: fade scale 0.175 -> 0.0 over the first 1000 PPO iters
# (horizon_length=256 -> 1000*256=256000 env control steps), then hold at 0.
# Bootstraps early learning with tactile, then converges on task reward alone.
# Set ANNEAL_STEPS=0 to disable (constant scale).
source "$(dirname "$0")/_common.sh"

./isaaclab.sh -p "$TRAIN" \
    "env.tactile_reward.ckpt=assets/TactileModel/nut_scratch_epoch12.pth" \
    "env.tactile_reward.scale=0.175" \
    "env.tactile_reward.scale_end=0.0" \
    "env.tactile_reward.anneal_steps=256000" \
    "env.tactile_reward.instruction=pick up the nut and thread it onto the bolt" \
    "env.tactile_reward.rewind_root=$TACTILE_ROOT" \
    --task Isaac-Forge-NutThread-PickPlace-Direct-v0 \
    --baseline baseline \
    --headless --num_envs 256 --max_iterations 10000 --enable_cameras --seed 1 \
    agent.params.config.full_experiment_name=NutThread_PickPlace_TaRL_0.175_anneal1000it_seed1 \
    agent.params.config.save_frequency=20
