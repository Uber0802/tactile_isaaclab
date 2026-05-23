FORGE_TACTILE_REWARD_CKPT=/mnt/tank/uber/Tactile-Reward/box_stack_scratch/box_stack_epoch19.pth \
FORGE_SAVE_TACTILE_FORCE_FIELD=1 \
FORGE_TACTILE_SAVE_DIR=./tactile_dataset/stack_box/baselineA \
FORGE_TACTILE_REWARD_SCALE=0.3 \
FORGE_TACTILE_REWARD_INSTRUCTION="grasp a box and stack it on another box" \
FORGE_TACTILE_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/Tactile-ReWiND \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Stack-Cube-Franka-Gelsight-v0 \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --headless \
    --track \
    --wandb-entity b06902045-national-taiwan-university \
    --wandb-project-name tactile-rewind \
    --wandb-name Stack_box_tacReward
