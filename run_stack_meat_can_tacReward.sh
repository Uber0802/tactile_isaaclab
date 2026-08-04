FORGE_TACTILE_REWARD_CKPT=/mnt/scratch/kimnai/Tactile-Reward/v3_box_curriculum/v3_box_curriculum_epoch25.pth \
FORGE_SAVE_TACTILE_FORCE_FIELD=1 \
FORGE_TACTILE_SAVE_DIR=/mnt/scratch/kimnai/research/tarl/tactile_dataset/stack_box/tacReward_42 \
FORGE_TACTILE_REWARD_SCALE=0.3 \
FORGE_TACTILE_REWARD_INSTRUCTION="grasp the potted meat can and stack it on the box" \
FORGE_TACTILE_REWARD_ROOT=/mnt/home/tactile/tactile_isaaclab/external/third-party/Tactile-ReWiND \
FORGE_TACTILE_REWARD_LOG_DIR=/mnt/scratch/kimnai/research/tarl/tactile_curves/stack_box/tacReward \
WANDB_DIR=/mnt/scratch/kimnai/research/tarl \
ISAACLAB_LOG_DIR=/mnt/scratch/kimnai/research/tarl/logs \
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Stack-Potted-Meat-Can-Franka-Gelsight-v0 \
    --num_envs 256 \
    --max_iterations 20000 \
    --enable_cameras \
    --headless \
    --track \
    --seed 42 \
    --checkpoint /mnt/home/kimnai/research/tactile_isaaclab/last_franka_stack_potted_meat_can_ep_13300_rew_83.377144.pth \
    --wandb-entity b06902045-national-taiwan-university \
    --wandb-project-name tactile-rewind \
    --wandb-name Stack_meat_can_tacReward_v3_03_annealing_continue_2
