FORGE_SAVE_TACTILE_FORCE_FIELD=1 \
FORGE_TACTILE_SAVE_DIR=./tactile_dataset/data_2 \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-PegInsert-PickPlace-Direct-v0 \
    --num_envs 128 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --wandb-entity b11902127-ntu \
    --wandb-project-name tactile-rewind \
    --wandb-name Peg_In_2Hole_baseline

