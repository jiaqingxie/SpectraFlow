# SpectraFlow

**Physics-Guided Diffusion-Transformer Flow Matching for IR-Raman Spectral Interconversion**

SpectraFlow is a spectral translation framework for cross-modal molecular spectra, with a focus on IR and Raman interconversion. The current codebase keeps three main model families:

- **Flow Matching**: `src/train_flow.py`, `src/test_flow.py`, `src/model_flow.py`
- **VibraDiT-Flow**: a 1D wavenumber-aware DiT backbone available through `--backbone vibradit`
- **Seq2Seq / VAE baselines**: retained for comparison and compatibility

## Method Highlights

- Conditional Flow Matching for spectrum-to-spectrum generation.
- Wavenumber-aware 1D Diffusion Transformer backbone for spectral sequences.
- Physics-guided spectral losses, including amplitude, gradient, curvature, local OT, and nonnegative constraints.
- Euler/RK4 ODE sampling support for generation and testing.

## Quick Start

Train VibraDiT-Flow on QMe14S IR to Raman:

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
  --gen_loss_prob 0.25 \
  --train_gen_steps 8 \
  --val_num_steps 8
```

Test the trained checkpoint:

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

More dataset-specific commands are collected in [`README_EXPERIMENT_COMMANDS.md`](README_EXPERIMENT_COMMANDS.md).

## Notes

- `--resize_shape` is still required by the current dataset pipeline. VibraDiT flattens the heatmap internally, so the effective sequence length is `height * width`.
- `--backbone vibradit` automatically enables the VibraDiT spectral loss setup.
- Validation and train-time endpoint sampling use Euler by default; RK4 is mainly intended for final testing when enabled.
