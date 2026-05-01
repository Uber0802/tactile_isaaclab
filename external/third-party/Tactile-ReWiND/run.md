conda activate rewind
pip install anthropic    # 第一次跑 LLM script 要裝
export ANTHROPIC_API_KEY=...

cd external/third-party/Tactile-ReWiND

# 1) LLM 產 paraphrase JSON（一次性）
python scripts/generate_instructions_via_llm.py
# → /mnt/tank/uber/Tactile-Reward/instructions.json

# 2) 建 metadata（within-task held-out，預設 1 traj/task 留 eval）
python scripts/anytouch2_to_h5.py \
  --instructions_json /mnt/tank/uber/Tactile-Reward/instructions.json

# 3) 訓練
python scripts/train_tactile_reward.py \
  --rewind --clip_grad \
  --batch_size 64 --max_length 16 \
  --epochs 20 --steps_per_epoch 200 \
  --num_workers 6 \
  --isaaclab_aligned \
  --run_name tactile_rewind_aligned_long2long \
  --ckpt_dir /mnt/tank/uber/Tactile-Reward/checkpoints_aligned_long2long

or

python scripts/precompute_aligned.py
python scripts/train_tactile_reward.py \
  --train_metadata /mnt/tank/uber/Tactile-Reward/tactile_metadata_train_aligned.h5 \
  --rewind --clip_grad \
  --batch_size 256 --max_length 16 \
  --epochs 20 --steps_per_epoch 200 \
  --num_workers 8 \
  --isaaclab_aligned \
  --run_name tactile_rewind_aligned_pre \
  --ckpt_dir /mnt/tank/uber/Tactile-Reward/checkpoints_aligned_pre

# 4) Within-task eval（檢查有沒有學起來）
python scripts/eval_tactile_reward.py \
  --ckpt /mnt/tank/uber/Tactile-Reward/checkpoints/tactile_rewind_epoch19.pth \
  --output /mnt/tank/uber/Tactile-Reward/eval_within_task.json

# visualize evaluate
python scripts/visualize_eval.py   --ckpt /mnt/tank/uber/Tactile-Reward/checkpoints/tactile_rewind_epoch19.pth   --limit 5 --overwrite --step 8 --middle_frames 50

# 5) 跨資料集 generalization 測試
python scripts/eval_cross_dataset.py \
  --ckpt /mnt/tank/uber/Tactile-Reward/checkpoints/tactile_rewind_epoch19.pth \
  --instruction "grasp peg and insert to another hole" \
  --mismatched_instruction "fold the towel into a square" \
  --save_curves
