FORGE_SAVE_TACTILE_FORCE_FIELD=1 \
FORGE_TACTILE_SAVE_DIR=/mnt/scratch/kimnai/research/tarl/tactile_dataset/stack_bowl/baselineA_42 \
FORGE_TACTILE_REWARD_SCALE=1.0 \
FORGE_TACTILE_REWARD_INSTRUCTION="grasp a bowl and stack it on a box" \
WANDB_DIR=/mnt/scratch/kimnai/research/tarl \
ISAACLAB_LOG_DIR=/mnt/scratch/kimnai/research/tarl/logs \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Stack-Bowl-Franka-Gelsight-v0 \
    --num_envs 768 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --headless \
    --seed 42 \
    --wandb-entity b06902045-national-taiwan-university \
    --wandb-project-name tactile-rewind \
    --wandb-name Stack_bowl_baselineA_42
