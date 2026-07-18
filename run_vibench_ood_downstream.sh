#!/usr/bin/env bash
set -euo pipefail

# Run on pjH200-ma4science from anywhere:
#   bash run_vibench_ood_downstream.sh
#
# Optional overrides:
#   DATASETS="pahs nist_ir" bash run_vibench_ood_downstream.sh
#   FORCE_EXTRACT=1 bash run_vibench_ood_downstream.sh

cd /mnt/shared-storage-user/xiejiaqing/spectrogen_v2

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate azr
elif [ -f "$HOME/.bashrc" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc"
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate azr
  else
    echo "Warning: conda command not found after sourcing ~/.bashrc; using current environment."
  fi
else
  echo "Warning: conda command not found; using current environment."
fi

SOURCE_MODE="${SOURCE_MODE:-ir}"
TARGET_MODE="${TARGET_MODE:-raman}"
DIRECTION="${SOURCE_MODE}2${TARGET_MODE}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/vibench_ood}"
SEED="${SEED:-0}"
TRAIN_SEED="${TRAIN_SEED:-42}"
RIDGE_ALPHA="${RIDGE_ALPHA:-10}"
NUM_SEEDS="${NUM_SEEDS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"

DATASETS="${DATASETS:-pahs nist_ir peptide_mod peptide geom zinc15 qm9 mols}"
PROPERTIES="${PROPERTIES:-MolWt LogP TPSA NumHDonors NumHAcceptors NumRotatableBonds RingCount HeavyAtomCount FractionCSP3 NumAromaticRings MolMR LabuteASA}"

RESULT_TXT="downstream/vibench_ood_${DIRECTION}_t0p5_vibradit_hidden_seed${SEED}_all_results.txt"
mkdir -p downstream
: > "$RESULT_TXT"

echo "VibraDiT hidden downstream run" | tee -a "$RESULT_TXT"
echo "direction=${DIRECTION} checkpoint_dir=${CHECKPOINT_DIR} seed=${SEED}" | tee -a "$RESULT_TXT"
echo "datasets=${DATASETS}" | tee -a "$RESULT_TXT"
echo "properties=${PROPERTIES}" | tee -a "$RESULT_TXT"
echo | tee -a "$RESULT_TXT"

needs_extract() {
  local feature_dir="$1"
  local props_csv="${feature_dir}/molecular_properties_${DIRECTION}.csv"
  local flow_npy="${feature_dir}/flow_embedding_${DIRECTION}.npy"

  if [ "$FORCE_EXTRACT" = "1" ]; then
    return 0
  fi
  if [ ! -f "$flow_npy" ] || [ ! -f "$props_csv" ]; then
    return 0
  fi

  PROPS_CSV="$props_csv" PROPERTIES="$PROPERTIES" python - <<'PY'
import os
import pandas as pd

props_csv = os.environ["PROPS_CSV"]
required = os.environ["PROPERTIES"].split()
cols = set(pd.read_csv(props_csv, nrows=0).columns)
missing = [p for p in required if p not in cols]
if missing:
    print("Missing property columns:", " ".join(missing))
    raise SystemExit(0)
raise SystemExit(1)
PY
}

echo "================================================================" | tee -a "$RESULT_TXT"
echo "PHASE 1: Extract all requested datasets" | tee -a "$RESULT_TXT"
echo "================================================================" | tee -a "$RESULT_TXT"

for DS in $DATASETS; do
  FEATURE_DIR="downstream/${DS}_ood_${DIRECTION}_t0p5_vibradit_hidden_seed${SEED}"
  SOURCE_CSV="${DS}_test_${SOURCE_MODE}_processed.csv"
  TARGET_CSV="${DS}_test_${TARGET_MODE}_processed.csv"
  SMILES_PATH="data/processed/${DS}_test_${SOURCE_MODE}_smiles.txt"

  echo | tee -a "$RESULT_TXT"
  echo "----- extract ${DS} -----" | tee -a "$RESULT_TXT"
  echo "FEATURE_DIR: ${FEATURE_DIR}" | tee -a "$RESULT_TXT"

  if needs_extract "$FEATURE_DIR"; then
    echo "[extract] ${DS}: extracting VibraDiT hidden flow embedding..." | tee -a "$RESULT_TXT"
    python src/extract_flow_embeddings_and_properties.py \
      --data_dir data/processed \
      --checkpoint_dir "$CHECKPOINT_DIR" \
      --source_mode "$SOURCE_MODE" \
      --target_mode "$TARGET_MODE" \
      --backbone vibradit \
      --seed "$SEED" \
      --dit_hidden_dim 384 \
      --dit_depth 8 \
      --dit_num_heads 6 \
      --dit_patch_size 20 \
      --t_for_embedding 0.5 \
      --embedding_type hidden \
      --embedding_num_steps 8 \
      --source_csv "$SOURCE_CSV" \
      --target_csv "$TARGET_CSV" \
      --smiles_path "$SMILES_PATH" \
      --output_dir "$FEATURE_DIR" \
      --batch_size "$BATCH_SIZE" 2>&1 | tee -a "$RESULT_TXT"
  else
    echo "[extract] ${DS}: existing features already complete, skipping." | tee -a "$RESULT_TXT"
  fi
done

echo | tee -a "$RESULT_TXT"
echo "================================================================" | tee -a "$RESULT_TXT"
echo "PHASE 2: Train downstream regressors" | tee -a "$RESULT_TXT"
echo "================================================================" | tee -a "$RESULT_TXT"

for DS in $DATASETS; do
  FEATURE_DIR="downstream/${DS}_ood_${DIRECTION}_t0p5_vibradit_hidden_seed${SEED}"

  echo | tee -a "$RESULT_TXT"
  echo "================================================================" | tee -a "$RESULT_TXT"
  echo "TRAIN DATASET: ${DS}" | tee -a "$RESULT_TXT"
  echo "FEATURE_DIR: ${FEATURE_DIR}" | tee -a "$RESULT_TXT"
  echo "================================================================" | tee -a "$RESULT_TXT"

  for PROP in $PROPERTIES; do
    echo | tee -a "$RESULT_TXT"
    echo "----- ${DS} | ${PROP} -----" | tee -a "$RESULT_TXT"
    python src/train_downstream.py \
      --feature_dir "$FEATURE_DIR" \
      --source_mode "$SOURCE_MODE" \
      --target_mode "$TARGET_MODE" \
      --target_property "$PROP" \
      --model_type ridge \
      --ridge_alpha "$RIDGE_ALPHA" \
      --num_seeds "$NUM_SEEDS" \
      --seed "$TRAIN_SEED" 2>&1 | tee -a "$RESULT_TXT"
  done
done

ARCHIVE="downstream/vibench_ood_${DIRECTION}_t0p5_vibradit_hidden_seed${SEED}_results.tar.gz"
tar -czf "$ARCHIVE" "$RESULT_TXT" downstream/*_ood_${DIRECTION}_t0p5_vibradit_hidden_seed${SEED} 2>/dev/null || true

echo "================================================================" | tee -a "$RESULT_TXT"
echo "DONE" | tee -a "$RESULT_TXT"
echo "Summary txt: ${RESULT_TXT}" | tee -a "$RESULT_TXT"
echo "Archive: ${ARCHIVE}" | tee -a "$RESULT_TXT"
