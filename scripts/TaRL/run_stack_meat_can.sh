WANDB_DIR=/mnt/scratch/kimnai/research/tarl \
ISAACLAB_LOG_DIR=/mnt/scratch/kimnai/research/tarl/logs \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    "env.tactile_save.force_field=true" \
    "env.tactile_save.save_dir=/mnt/scratch/kimnai/research/tarl/tactile_dataset/stack_meat_can/baselineA_43" \
    "env.tactile_reward.scale=1.0" \
    "env.tactile_reward.instruction=grasp the potted meat can and stack it on the box" \
    --task Isaac-Stack-Potted-Meat-Can-Franka-Gelsight-v0 \
    --num_envs 256 \
    --max_iterations 15000 \
    --enable_cameras \
    --seed 42 \
    --headless