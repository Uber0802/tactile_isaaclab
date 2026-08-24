#!/bin/bash
# Peg pickplace — tactile as state AND as reward: the frozen AE latent in
# obs+state plus the TaRL progress reward on top.
# Success-triggered annealing — see run_gearpickplace_tactile_state_TaRL.sh.
source "$(dirname "$0")/_common.sh"

./isaaclab.sh -p "$TRAIN" \
    "env.tactile_reward.ckpt=assets/TactileModel/peg_curr_epoch17.pth" \
    "env.tactile_reward.scale=0.1" \
    "env.tactile_reward.instruction=grasp peg and insert to another hole" \
    "env.tactile_reward.rewind_root=$TACTILE_ROOT" \
    "env.tactile_encoder.ckpt=assets/TactileModel/peg_ae_best.pth" \
    "env.tactile_encoder.dim=32" \
    "env.tactile_encoder.root=$TACTILE_ROOT" \
    --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
    --baseline tactile_state \
    --headless --seed 0 --num_envs 128 --max_iterations 10000 \
    agent.params.config.full_experiment_name=PegInsert_PickPlace_tactile_state_TaRL \
    agent.params.config.save_frequency=20
