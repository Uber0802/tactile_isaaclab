FORGE_SAVE_TACTILE_FORCE_FIELD=1 \
FORGE_TACTILE_SAVE_DIR=./tactile_dataset/gearmesh_1 \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-GearMesh-Direct-v0 \
    --num_envs 256 \
    --max_iterations 10000 \
    --enable_cameras \
    --track \
    --wandb-entity b11902127-ntu \
    --wandb-project-name tactile-rewind \
    --wandb-name GearMesh_baseline
