#!/usr/bin/env bash
set -euo pipefail
cd /mnt/shared-storage-user/xiejiaqing/spectrogen_v2
mkdir -p logs results/verify_fig1_seed2

run_cmd() {
  echo
  echo "============================================================"
  echo "[RUN] $*"
  echo "============================================================"
  "$@"
}

# QM9S VAE
run_cmd python src/test.py \
  --data_dir /mnt/shared-storage-user/xiejiaqing/spectrogen_v2/data/processed \
  --checkpoint_dir checkpoints/qm9s \
  --source_mode ir \
  --target_mode raman \
  --heatmap_size 3600 \
  --resize_shape 60 60 \
  --seed 2 \
  --batch_size 32 \
  --save_dir results/verify_fig1_seed2/qm9s_vae_ir2raman_seed2

run_cmd python src/test.py \
  --data_dir /mnt/shared-storage-user/xiejiaqing/spectrogen_v2/data/processed \
  --checkpoint_dir checkpoints/qm9s \
  --source_mode raman \
  --target_mode ir \
  --heatmap_size 3600 \
  --resize_shape 60 60 \
  --seed 2 \
  --batch_size 32 \
  --save_dir results/verify_fig1_seed2/qm9s_vae_raman2ir_seed2

# QM9S VibraFlow, Euler 8 steps
run_cmd python src/test_flow.py \
  --data_dir /mnt/shared-storage-user/xiejiaqing/spectrogen_v2/data/processed \
  --checkpoint_dir checkpoints/qm9s \
  --source_mode ir \
  --target_mode raman \
  --heatmap_size 3600 \
  --resize_shape 60 60 \
  --seed 2 \
  --batch_size 32 \
  --save_dir results/verify_fig1_seed2/qm9s_flow_ir2raman_vibradit_seed2 \
  --backbone vibradit \
  --dit_patch_size 20 \
  --dit_hidden_dim 384 \
  --dit_depth 8 \
  --dit_num_heads 6 \
  --no_rk4 \
  --num_steps 8

run_cmd python src/test_flow.py \
  --data_dir /mnt/shared-storage-user/xiejiaqing/spectrogen_v2/data/processed \
  --checkpoint_dir checkpoints/qm9s \
  --source_mode raman \
  --target_mode ir \
  --heatmap_size 3600 \
  --resize_shape 60 60 \
  --seed 2 \
  --batch_size 32 \
  --save_dir results/verify_fig1_seed2/qm9s_flow_raman2ir_vibradit_seed2 \
  --backbone vibradit \
  --dit_patch_size 20 \
  --dit_hidden_dim 384 \
  --dit_depth 8 \
  --dit_num_heads 6 \
  --no_rk4 \
  --num_steps 8

# QMe14S VAE
run_cmd python src/test.py \
  --data_dir /mnt/shared-storage-user/xiejiaqing/data/QMe14S/processed \
  --checkpoint_dir checkpoints/qme14s \
  --source_mode ir \
  --target_mode raman \
  --heatmap_size 3600 \
  --resize_shape 60 60 \
  --seed 2 \
  --batch_size 32 \
  --save_dir results/verify_fig1_seed2/qme14s_vae_ir2raman_seed2

run_cmd python src/test.py \
  --data_dir /mnt/shared-storage-user/xiejiaqing/data/QMe14S/processed \
  --checkpoint_dir checkpoints/qme14s \
  --source_mode raman \
  --target_mode ir \
  --heatmap_size 3600 \
  --resize_shape 60 60 \
  --seed 2 \
  --batch_size 32 \
  --save_dir results/verify_fig1_seed2/qme14s_vae_raman2ir_seed2

# QMe14S VibraFlow, Euler 8 steps
run_cmd python src/test_flow.py \
  --data_dir /mnt/shared-storage-user/xiejiaqing/data/QMe14S/processed \
  --checkpoint_dir checkpoints/qme14s \
  --source_mode ir \
  --target_mode raman \
  --heatmap_size 3600 \
  --resize_shape 60 60 \
  --seed 2 \
  --batch_size 32 \
  --save_dir results/verify_fig1_seed2/qme14s_flow_ir2raman_vibradit_seed2 \
  --backbone vibradit \
  --dit_patch_size 20 \
  --dit_hidden_dim 384 \
  --dit_depth 8 \
  --dit_num_heads 6 \
  --no_rk4 \
  --num_steps 8

run_cmd python src/test_flow.py \
  --data_dir /mnt/shared-storage-user/xiejiaqing/data/QMe14S/processed \
  --checkpoint_dir checkpoints/qme14s \
  --source_mode raman \
  --target_mode ir \
  --heatmap_size 3600 \
  --resize_shape 60 60 \
  --seed 2 \
  --batch_size 32 \
  --save_dir results/verify_fig1_seed2/qme14s_flow_raman2ir_vibradit_seed2 \
  --backbone vibradit \
  --dit_patch_size 20 \
  --dit_hidden_dim 384 \
  --dit_depth 8 \
  --dit_num_heads 6 \
  --no_rk4 \
  --num_steps 8
