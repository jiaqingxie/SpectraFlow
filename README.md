# SpectraFlow

**Physics-Guided Flow Matching for IR–Raman Spectral Interconversion**

SpectraFlow recasts infrared (IR) ↔ Raman spectral translation as a continuous
transport problem. Instead of a direct point-wise regression, it learns a
conditional velocity field with **flow matching** and integrates a deterministic
probability-flow ODE from the source spectrum to the paired target modality. The
velocity field is a **wavenumber-aware 1D Diffusion Transformer (SpectraDiT /
VibraDiT)**, and training is regularized by physics-guided spectral losses that
emphasize peaks, edges, and sharp features.

Because the prediction is produced by integrating an ODE, every intermediate
state along the trajectory is an inspectable spectrum, and the internal
representation at an intermediate flow time doubles as a transferable molecular
embedding for downstream property prediction.

---

## Table of contents

1. [Method highlights](#method-highlights)
2. [Repository structure](#repository-structure)
3. [Installation](#installation)
4. [Data preparation](#data-preparation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Downstream representation probing](#downstream-representation-probing)
8. [Analysis and figures](#analysis-and-figures)
9. [Reproducing the paper](#reproducing-the-paper)
10. [Default hyperparameters](#default-hyperparameters)
11. [Citation](#citation)

---

## Method highlights

- **Conditional flow matching** for spectrum-to-spectrum generation, trained with
  a straight-line (optimal-transport) interpolation and a constant target
  velocity.
- **Wavenumber-aware 1D DiT backbone** (`--backbone vibradit`): the spectral map
  is flattened into an ordered wavenumber sequence, patchified, and processed
  with self-attention and adaLN-Zero conditioning on flow time, target modality,
  and source spectrum.
- **Physics-guided objective**: weighted elastic (L1/L2) velocity loss with
  amplitude / gradient / curvature importance weighting, plus endpoint
  derivative-shape, local optimal-transport, and non-negativity penalties.
- **Deterministic ODE sampling** with Euler (default) or RK4 solvers; the full
  trajectory can be exported for visualization.
- **Baselines** included for fair comparison: a convolutional conditional VAE and
  an attention-based Seq2Seq (BiGRU + Bahdanau attention) translator.

## Repository structure

```
SpectraFlow/
├── src/
│   ├── model_flow.py       # ConditionalFlowMatching + backbones (unet/dit/vibradit)
│   ├── train_flow.py       # Flow-matching training entry point
│   ├── test_flow.py        # Evaluation (MAE/RMSE/R²/Pearson/PSNR/SSIM/JSD, peaks, DTW)
│   ├── model_seq2seq.py    # Seq2Seq baseline model
│   ├── train_seq2seq.py    # Seq2Seq training / test_seq2seq.py
│   ├── model.py            # CrossModalVAE baseline
│   ├── train.py            # VAE training + shared paired-dataset loaders
│   ├── test.py             # VAE evaluation
│   ├── process.py          # QM9S CSV -> processed spectra (CSV/H5)
│   ├── process_qme14s.py   # QMe14S ingestion
│   ├── merge_vibench_test_h5.py  # Build paired ViBench test sets
│   ├── extract_flow_embeddings_and_properties.py  # Flow embeddings + RDKit labels
│   ├── train_downstream.py # Linear/RF probes on features
│   └── export_test_smiles.py     # Export test-split SMILES aligned to spectra
├── vibradit_flow.py        # Single-file standalone VibraDiT-Flow implementation
├── analyze_scaffold_performance.py   # Per-scaffold R² breakdown
├── build_scaffold_r2_barplots.py     # Scaffold-class R² bar plots (Fig. 2d)
├── build_fig4_embedding_umap*.py     # UMAP embedding figures (Fig. 4)
├── build_vibench_ood_downstream_plots.py  # OOD downstream summary plots
├── visualize_flow_matching.py        # Flow-trajectory illustration
├── requirements.txt
└── README.md
```

> `data/`, `checkpoints/`, and `results/` are created at runtime and are not
> tracked. Point the scripts at your own data locations via `--data_dir`.

## Installation

```bash
git clone https://github.com/jiaqingxie/SpectraFlow.git
cd SpectraFlow

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

RDKit is installed via pip (`rdkit`). If you prefer conda, use
`conda install -c conda-forge rdkit` and install the remaining packages with pip.
A CUDA-enabled PyTorch build is recommended for training; install the matching
wheel from [pytorch.org](https://pytorch.org) for your CUDA version.

## Data preparation

SpectraFlow trains on paired IR/Raman spectra from three benchmarks:

| Dataset  | Molecules | Spectral length `L` | Map size | Role |
|----------|-----------|---------------------|----------|------|
| QM9S     | 129,218   | 3600                | 60×60    | In-distribution |
| QMe14S   | 186,102   | 3600                | 60×60    | Larger / more diverse in-distribution |
| ViBench  | 129,218   | 1024                | 32×32    | Compressed QM9S; OOD generalization |

The data pipeline expects each spectrum resampled to a fixed length (`3600` for
QM9S/QMe14S, `1024` for ViBench) and reshaped into a square single-channel map.
Every spectrum is min–max normalized per sample.

**QM9S** — from raw broadened CSVs (rows = spectra, first row = wavenumber axis):

```bash
python src/process.py \
  --data_dir /path/to/qm9s_raw \
  --output_dir data/processed \
  --h5_only
```

This produces `data/processed/{ir,uv,raman}_broaden_processed.h5`.

**QMe14S** — use `src/process_qme14s.py` with the QMe14S source files.

**ViBench** — merge the released per-domain test H5 files into paired
IR/Raman sets with `src/merge_vibench_test_h5.py`.

To obtain the SMILES aligned to the exact test-split row order (needed for
scaffold and downstream analysis), use `src/export_test_smiles.py`.

## Training

### SpectraFlow (VibraDiT-Flow)

QM9S, IR → Raman (matches the paper configuration):

```bash
python src/train_flow.py \
  --data_dir data/processed \
  --save_dir checkpoints \
  --source_mode ir --target_mode raman \
  --heatmap_size 3600 --resize_shape 60 60 \
  --batch_size 64 --epochs 100 --learning_rate 2e-4 --seed 1 \
  --backbone vibradit \
  --dit_patch_size 20 --dit_hidden_dim 384 --dit_depth 8 --dit_num_heads 6 \
  --use_mixed_loss --gen_loss_weight 0.5 --gen_loss_prob 0.5 \
  --train_gen_steps 8 --val_num_steps 8
```

Notes:
- `--backbone vibradit` automatically enables the physics-guided spectral losses
  (`--lambda_shape 0.05`, `--lambda_ot 0.02`, `--lambda_pos 0.01` by default).
- `--gen_loss_prob` is the endpoint-consistency probability `p` (paper ablates
  `0.1` vs `0.5`; `0.5` is the default best setting).
- If `--data_dir` contains `qm9s`, checkpoints are saved under
  `save_dir/qm9s/` unless `--no_dataset_subdir` is passed.
- The best checkpoint (lowest validation MAE) is written as
  `flow_{src}2{tgt}_vibradit_best_seed{seed}.pt`.
- Reverse direction: swap `--source_mode raman --target_mode ir`.

### Baselines

VAE:

```bash
python src/train.py --data_dir data/processed \
  --source_mode ir --target_mode raman \
  --latent_dim 128 --beta_max 1e-3 --batch_size 32 --learning_rate 4e-4
```

Seq2Seq:

```bash
python src/train_seq2seq.py --data_dir data/processed \
  --source_mode ir --target_mode raman \
  --hidden_dim 256 --depth 6 --patch_size 4 \
  --batch_size 32 --learning_rate 4e-4 --epochs 50
```

## Evaluation

```bash
python src/test_flow.py \
  --data_dir data/processed \
  --checkpoint_dir checkpoints \
  --source_mode ir --target_mode raman \
  --heatmap_size 3600 --resize_shape 60 60 \
  --batch_size 32 --seed 1 \
  --backbone vibradit \
  --dit_patch_size 20 --dit_hidden_dim 384 --dit_depth 8 --dit_num_heads 6 \
  --save_dir results/qm9s_flow_ir2raman \
  --no_rk4
```

`test_flow.py` reports MAE, RMSE, R², and Pearson `r`, and (with extended
metrics) PSNR, SSIM, Jensen–Shannon distance, band-constrained DTW, and
peak-matching errors. Use `--num_steps` to change the number of ODE steps
(default `32` for evaluation) and drop `--no_rk4` to switch to the RK4 solver.
`test_seq2seq.py` and `test.py` evaluate the baselines with the same metrics.

## Downstream representation probing

Extract Flow embeddings and RDKit descriptor/fingerprint labels aligned to the
test SMILES:

```bash
python src/extract_flow_embeddings_and_properties.py \
  --data_dir data/processed \
  --checkpoint_dir checkpoints \
  --source_mode ir --target_mode raman \
  --backbone vibradit \
  --dit_patch_size 20 --dit_hidden_dim 384 --dit_depth 8 --dit_num_heads 6 \
  --smiles_path data/processed/test_smiles.txt \
  --t_for_embedding 0.5 --embedding_type hidden --embedding_num_steps 8 \
  --output_dir downstream/qm9s_ir2raman
```

Then probe each feature set (IR raw / Raman raw / Flow embedding / Morgan
fingerprint / Flow+FP) with a ridge or random-forest regressor:

```bash
python src/train_downstream.py \
  --feature_dir downstream/qm9s_ir2raman \
  --source_mode ir --target_mode raman \
  --target_property LogP \
  --model_type ridge --num_seeds 5
```

## Analysis and figures

- **Scaffold-class R² (Fig. 2d):** `analyze_scaffold_performance.py` then
  `build_scaffold_r2_barplots.py`.
- **Embedding UMAPs (Fig. 4):** `build_fig4_embedding_umap.py` and the
  `build_fig4_*` companions.
- **OOD downstream summaries:** `build_vibench_ood_downstream_plots.py`.
- **Flow trajectory illustration:** `visualize_flow_matching.py`.

## Reproducing the paper

- All compared models share a fixed-seed random split
  (train/val/test = 0.70/0.15/0.15) via `get_paired_loaders(..., seed=...)`.
- Main-table numbers are the mean ± s.d. over **4 seeds**; downstream probes use
  **5 seeds**.
- For the OOD study, train only on the ViBench-QM9 source domain and evaluate on
  the held-out domains without per-dataset tuning (same backbone width/depth/heads,
  only the sequence length changes with map size).
- Expected checkpoint names: `flow_{src}2{tgt}_vibradit_best_seed{seed}.pt`.
- The standalone `vibradit_flow.py` reproduces the core training/inference loop
  in a single file if you prefer to run without the `src/` package layout.

## Default hyperparameters

| Group | Hyperparameter | Value |
|-------|----------------|-------|
| Architecture | hidden width `d` | 384 |
| | DiT blocks (depth) | 8 |
| | attention heads | 6 |
| | patch size `P` | 20 |
| | conditioning | adaLN-Zero |
| Optimization | optimizer | AdamW |
| | learning rate | 2e-4 (cosine) |
| | weight decay | 1e-4 |
| | batch size | 64 |
| | gradient clipping | 1.0 |
| | epochs | 100 |
| Objective | amplitude / grad. / curv. weights | 1.0 / 0.5 / 0.5 |
| | endpoint / shape / OT / nonneg. weights | 0.5 / 0.05 / 0.02 / 0.01 |
| | endpoint probability `p` | 0.5 |
| Sampling | train-time endpoint ODE steps | 8 |
| | inference ODE steps | 32 |
| | solver | Euler / RK4 |

Values above are the defaults for QM9S; the QMe14S/ViBench runs keep the same
optimization and objective settings and only change map size and patch size.

## Citation

If you use SpectraFlow, please cite the accompanying manuscript. A BibTeX entry
will be added here upon publication.
