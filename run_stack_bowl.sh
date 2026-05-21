FORGE_SAVE_TACTILE_FORCE_FIELD=1 \
FORGE_TACTILE_SAVE_DIR=./tactile_dataset/stack_bowl/baselineA \
FORGE_TACTILE_REWARD_SCALE=1.0 \
FORGE_TACTILE_REWARD_INSTRUCTION="grasp a bowl and stack it on a box" \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Stack-Bowl-Franka-Gelsight-v0 \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --wandb-entity b06902045-national-taiwan-university \
    --wandb-project-name tactile-rewind \
    --wandb-name Stack_bowl_baselineA
