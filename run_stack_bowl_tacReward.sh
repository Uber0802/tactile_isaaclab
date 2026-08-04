FORGE_TACTILE_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/box_kimnai_curriculum/box_kimnai_curr_epoch25.pth \
FORGE_SAVE_TACTILE_FORCE_FIELD=1 \
FORGE_TACTILE_SAVE_DIR=/mnt/scratch/kimnai/research/tarl/tactile_dataset/stack_bowl/tacReward_42 \
FORGE_TACTILE_REWARD_SCALE=0.3 \
FORGE_TACTILE_REWARD_INSTRUCTION="grasp a bowl and stack it on a box" \
FORGE_ENABLE_SENSOR=1 \
FORGE_TACTILE_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/Tactile-ReWiND \
WANDB_DIR=/mnt/scratch/kimnai/research/tarl \
ISAACLAB_LOG_DIR=/mnt/scratch/kimnai/research/tarl/logs \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Stack-Bowl-Franka-Gelsight-v0 \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --headless \
    --seed 42 \
    --wandb-entity b06902045-national-taiwan-university \
    --wandb-project-name tactile-rewind \
    --wandb-name Stack_bowl_tacReward_42

