FORGE_TACTILE_ENCODER_CKPT=/mnt/scratch/kimnai/research/tarl/box_ae_16/ae_best.pth \
FORGE_TACTILE_ENCODER_DIM=32 \
WANDB_DIR=/mnt/scratch/kimnai/research/tarl \
ISAACLAB_LOG_DIR=/mnt/scratch/kimnai/research/tarl/logs \
FORGE_ENABLE_SENSOR=1 \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    "env.tactile_save.force_field=true" \
    "env.tactile_save.save_dir=/mnt/scratch/kimnai/research/tarl/tactile_dataset/stack_box/baselineA_123" \
    "env.tactile_reward.scale=1.0" \
    "env.tactile_reward.instruction=grasp the blue box and stack it on the red box" \
    --task Isaac-Stack-Cube-Franka-Gelsight-v0 \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --seed 123 \
    --headless \
    --track \
    --wandb-entity b06902045-national-taiwan-university \
    --wandb-project-name tactile-rewind \
    --wandb-name Stack_box_new_robot_baselineB_v3_123
