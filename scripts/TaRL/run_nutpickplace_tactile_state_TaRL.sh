#!/bin/bash
# Nut pickplace — tactile as state AND as reward: the frozen AE latent in
# obs+state plus the TaRL progress reward on top.
# Success-triggered annealing — see run_gearpickplace_tactile_state_TaRL.sh.
source "$(dirname "$0")/_common.sh"

./isaaclab.sh -p "$TRAIN" \
    "env.tactile_reward.ckpt=assets/TactileModel/nut_scratch_epoch12.pth" \
    "env.tactile_reward.scale=0.175" \
    "env.tactile_reward.scale_end=0.0" \
    "env.tactile_reward.anneal_mode=success" \
    "env.tactile_reward.anneal_success_thresh=0.1" \
    "env.tactile_reward.anneal_steps=5120" \
    "env.tactile_reward.instruction=pick up the nut and thread it onto the bolt" \
    "env.tactile_reward.rewind_root=$TACTILE_ROOT" \
    "env.tactile_encoder.ckpt=assets/TactileModel/nut_ae_best.pth" \
    "env.tactile_encoder.dim=32" \
    "env.tactile_encoder.root=$TACTILE_ROOT" \
    --task Isaac-Forge-NutThread-PickPlace-Direct-v0 \
    --baseline tactile_state \
    --headless --seed 1 --num_envs 256 --max_iterations 10000 \
    agent.params.config.full_experiment_name=NutThread_PickPlace_tactile_state_TaRL \
    agent.params.config.save_frequency=20
