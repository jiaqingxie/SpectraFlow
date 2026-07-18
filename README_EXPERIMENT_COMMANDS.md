# Experiment Commands

This file records the current runnable commands after the cleanup. Transformer/UNet
scripts were removed, so QM9S commands are written for the retained Flow pipeline
with `--backbone vibradit`.

## QMe14S IR to Raman

Train VibraDiT-Flow:

```bash
python src/train_flow.py \
  --data_dir /mnt/shared-storage-user/xiejiaqing/data/QMe14S/processed \
  --save_dir checkpoints/qme14s \
  --source_mode ir \
  --target_mode raman \
  --heatmap_size 3600 \
  --resize_shape 60 60 \
  --batch_size 64 \
  --epochs 100 \
  --learning_rate 2e-4 \
  --seed 2 \
  --backbone vibradit \
  --dit_patch_size 20 \
  --dit_hidden_dim 384 \
  --dit_depth 8 \
  --dit_num_heads 6 \
  --use_mixed_loss \
  --gen_loss_weight 0.5 \
  --gen_loss_prob 0.1 \
  --train_gen_steps 8 \
  --val_num_steps 8
```

Test VibraDiT-Flow:

```bash
python src/test_flow.py \
  --data_dir /mnt/shared-storage-user/xiejiaqing/data/QMe14S/processed \
  --checkpoint_dir checkpoints/qme14s \
  --source_mode ir \
  --target_mode raman \
  --heatmap_size 3600 \
  --resize_shape 60 60 \
  --batch_size 32 \
  --save_dir results/qme14s_flow_ir2raman_vibradit \
  --seed 2 \
  --backbone vibradit \
  --dit_patch_size 20 \
  --dit_hidden_dim 384 \
  --dit_depth 8 \
  --dit_num_heads 6 \
  --no_rk4
```

Expected checkpoint:

```text
checkpoints/qme14s/flow_ir2raman_vibradit_best_seed2.pt
```

## QM9S IR to Raman

The old Transformer command is converted to the retained Flow/VibraDiT interface.
The original Transformer parameters map as:

```text
hidden_dim 256 -> --dit_hidden_dim 256
depth 6        -> --dit_depth 6
num_heads 4    -> --dit_num_heads 4
patch_size 4   -> --dit_patch_size 4
```

Train VibraDiT-Flow:

```bash
python src/train_flow.py \
  --data_dir /mnt/shared-storage-user/xiejiaqing/spectrogen_v2/data/processed \
  --save_dir checkpoints/qm9s \
  --no_dataset_subdir \
  --source_mode ir \
  --target_mode raman \
  --heatmap_size 3600 \
  --resize_shape 60 60 \
  --batch_size 32 \
  --epochs 100 \
  --learning_rate 2e-4 \
  --seed 1 \
  --backbone vibradit \
  --dit_hidden_dim 256 \
  --dit_depth 6 \
  --dit_num_heads 4 \
  --dit_patch_size 4 \
  --use_mixed_loss \
  --gen_loss_weight 0.5 \
  --gen_loss_prob 0.1 \
  --train_gen_steps 8 \
  --val_num_steps 8
```

Test VibraDiT-Flow:

```bash
python src/test_flow.py \
  --data_dir /mnt/shared-storage-user/xiejiaqing/spectrogen_v2/data/processed \
  --checkpoint_dir checkpoints/qm9s \
  --source_mode ir \
  --target_mode raman \
  --heatmap_size 3600 \
  --resize_shape 60 60 \
  --batch_size 32 \
  --seed 1 \
  --backbone vibradit \
  --dit_hidden_dim 256 \
  --dit_depth 6 \
  --dit_num_heads 4 \
  --dit_patch_size 4 \
  --no_rk4
```

Expected checkpoint:

```text
checkpoints/qm9s/flow_ir2raman_vibradit_best_seed1.pt
```

## VIBench

TODO: add VIBench train/test commands when the final data path, split files, and
checkpoint directory are confirmed.

## Notes

- `--resize_shape` is still required by the current dataset pipeline. For
  `vibradit`, the model flattens the heatmap internally, so the effective 1D
  sequence length is `height * width`.
- Train and validation use Euler sampling by default. RK4 is only used if
  explicitly enabled in training or omitted from test without `--no_rk4`.
- `--lambda_shape 0.05`, `--lambda_ot 0.02`, and `--lambda_pos 0.01` are default
  values in `src/train_flow.py`.
