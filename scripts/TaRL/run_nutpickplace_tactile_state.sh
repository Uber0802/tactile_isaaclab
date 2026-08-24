#!/bin/bash
# Nut pickplace — tactile-as-state: `baseline` reward shaping + a frozen
# autoencoder tactile latent appended to actor obs AND critic state.
# Pairs with run_nutpickplace_baseline.sh; see
# run_gearpickplace_tactile_state.sh for the encoder pretraining recipe.
source "$(dirname "$0")/_common.sh"

./isaaclab.sh -p "$TRAIN" \
    "env.tactile_encoder.ckpt=assets/TactileModel/nut_ae_best.pth" \
    "env.tactile_encoder.dim=32" \
    "env.tactile_encoder.root=$TACTILE_ROOT" \
    --task Isaac-Forge-NutThread-PickPlace-Direct-v0 \
    --baseline tactile_state \
    --headless --num_envs 256 --max_iterations 10000 \
    agent.params.config.full_experiment_name=NutThread_PickPlace_tactile_state \
    agent.params.config.save_frequency=20
