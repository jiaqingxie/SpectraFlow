# Elucidation: Spectrum -> SMILES

This module trains a spectrum-to-SMILES model with:

- **1D spectrum encoder** (Conv1d + Transformer encoder)
- **Transformer decoder** (BART-like autoregressive decoding)

## Data format

- `ir_csv` / `raman_csv`: same format as existing project processed files  
  (`header=None`, first row is x-axis, first column is index)
- `smiles_txt`: one SMILES per line, or `index<TAB>smiles`, or `index,smiles`

The row order of spectra and SMILES should align.

## Quick start

Train (IR -> SMILES):

```bash
python elucidation/train.py \
  --ir_csv /mnt/shared-storage-user/xiejiaqing/spectrogen_v2/data/processed/qm9_test_ir_processed.csv \
  --raman_csv /mnt/shared-storage-user/xiejiaqing/spectrogen_v2/data/processed/qm9_test_raman_processed.csv \
  --smiles_txt /mnt/shared-storage-user/xiejiaqing/data/qm9s/mapping.txt \
  --input_modality ir \
  --save_dir elucidation/checkpoints_ir \
  --epochs 30 \
  --batch_size 64 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --mask_prob 0.45 \
  --lm_weight 1.0 \
  --mlm_weight 1.0 \
  --seed 2
```

Train (IR+Raman -> SMILES):

```bash
python elucidation/train.py \
  --ir_csv ... \
  --raman_csv ... \
  --smiles_txt ... \
  --input_modality ir_raman \
  --save_dir elucidation/checkpoints_ir_raman
```

Inference:

```bash
python elucidation/infer.py \
  --checkpoint elucidation/checkpoints_ir/best.pt \
  --ir_csv /mnt/shared-storage-user/xiejiaqing/spectrogen_v2/data/processed/qm9_test_ir_processed.csv \
  --raman_csv /mnt/shared-storage-user/xiejiaqing/spectrogen_v2/data/processed/qm9_test_raman_processed.csv \
  --input_modality ir \
  --output_txt elucidation/pred_smiles.txt
```

## Metrics shown in training

- `train_loss` (joint loss)
- `val_loss` (joint loss)
- `lm` / `mlm` sub-losses (vib2mol-like phase2 style)
- `val_token_acc` (next-token top-1 accuracy)
- `val_top1_acc` (exact SMILES match by greedy decoding)
- `test_top1_acc` (printed once after training)
