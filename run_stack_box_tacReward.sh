FORGE_ENABLE_SENSOR=1 \
WANDB_DIR=/mnt/scratch/kimnai/research/tarl \
ISAACLAB_LOG_DIR=/mnt/scratch/kimnai/research/tarl/logs \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    "env.tactile_save.force_field=true" \
    "env.tactile_save.save_dir=/mnt/scratch/kimnai/research/tarl/tactile_dataset/stack_box/tacReward_123" \
    "env.tactile_reward.ckpt=/mnt/scratch/kimnai/Tactile-Reward/v3_box_curriculum/v3_box_curriculum_epoch25.pth" \
    "env.tactile_reward.scale=0.3" \
    "env.tactile_reward.instruction=grasp the blue box and stack it on the red box" \
    "env.tactile_reward.rewind_root=/mnt/home/tactile/tactile_isaaclab/external/third-party/Tactile-ReWiND" \
    "env.tactile_reward.curve_log_dir=/mnt/scratch/kimnai/research/tarl/tactile_curves/stack_box/tacReward" \
    --task Isaac-Stack-Cube-Franka-Gelsight-v0 \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --headless \
    --seed 123 \
    --track \
    --wandb-entity b06902045-national-taiwan-university \
    --wandb-project-name tactile-rewind \
    --wandb-name Stack_box_tacReward_new_robot_v3_123_03
