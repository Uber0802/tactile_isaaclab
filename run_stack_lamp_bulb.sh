FORGE_SAVE_TACTILE_FORCE_FIELD=1 \
FORGE_TACTILE_SAVE_DIR=./tactile_dataset/stack_lamp_bulb/baselineA \
FORGE_TACTILE_REWARD_SCALE=1.0 \
FORGE_TACTILE_REWARD_INSTRUCTION="grasp the lamp bulb and stack it on the box" \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Stack-Lamp-Bulb-Franka-Gelsight-v0 \
    --num_envs 1024 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --wandb-entity b06902045-national-taiwan-university \
    --wandb-project-name tactile-rewind \
    --wandb-name Stack_lamp_bulb_baselineA
