#!/usr/bin/env bash
set -eo pipefail

source /mnt/shared-storage-user/xiejiaqing/miniconda3/etc/profile.d/conda.sh
conda activate azr
cd /mnt/shared-storage-user/xiejiaqing/spectrogen_v2

EXTRA_ARGS=()
if [[ "${USE_CPU:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--cpu --no_amp)
fi
if [[ -n "${CPU_THREADS:-}" ]]; then
  EXTRA_ARGS+=(--cpu_threads "${CPU_THREADS}")
fi
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SOURCE_MODE="${SOURCE_MODE:-ir}"
TARGET_MODE="${TARGET_MODE:-raman}"
SAVE_DIR="${SAVE_DIR:-checkpoints/qm9s_direct}"

python src/train_direct.py \
  --data_dir /mnt/shared-storage-user/xiejiaqing/spectrogen_v2/data/processed \
  --save_dir "${SAVE_DIR}" \
  --source_mode "${SOURCE_MODE}" \
  --target_mode "${TARGET_MODE}" \
  --heatmap_size 3600 \
  --resize_shape 60 60 \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --learning_rate 2e-4 \
  --seed 2 \
  --backbone vibradit \
  --dit_hidden_dim 384 \
  --dit_depth 8 \
  --dit_num_heads 6 \
  --dit_patch_size 20 \
  --direct_time 0 \
  --endpoint_loss_weight 0.1 \
  --endpoint_loss_prob 0.1 \
  --amp_weight 0.0 \
  --grad_weight 0 \
  --curv_weight 0 \
  --lambda_shape 0 \
  --lambda_ot 0 \
  --lambda_pos 0 \
  --ot_window_size 64 \
  "${EXTRA_ARGS[@]}"
