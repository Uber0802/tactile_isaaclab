FORGE_TACTILE_REWARD_CKPT=/home/kim/ml/tactile-irl/tactile_isaaclab/box_kimnai_curr_epoch25.pth \
FORGE_SAVE_TACTILE_FORCE_FIELD=1 \
FORGE_TACTILE_SAVE_DIR=./tactile_dataset/stack_bowl/tacReward_501 \
FORGE_TACTILE_REWARD_SCALE=0.3 \
FORGE_TACTILE_REWARD_INSTRUCTION="grasp a bowl and stack it on a box" \
FORGE_TACTILE_REWARD_ROOT=/home/kim/ml/tactile-irl/Tactile-ReWiND \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Stack-Bowl-Franka-Gelsight-v0 \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --headless \
    --seed 501 \
    --wandb-entity b06902045-national-taiwan-university \
    --wandb-project-name tactile-rewind \
    --wandb-name Stack_bowl_tacReward_501

