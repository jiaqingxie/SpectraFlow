# SpectraFlow

**Spectroscopy-informed flow matching for bidirectional IR–Raman translation**

![SpectraFlow overview](docs/figure1_overview.png)

This release includes the training/evaluation code, the manuscript LaTeX source,
audited Results figures, numerical figure inputs and reproducibility scripts.
**Checkpoints and full datasets are not included.** Supply your own weights using
the filenames and task configuration below, or train new models.

The model learns a conditional velocity field between paired, normalized spectra
and predicts the target using deterministic ODE integration. Intermediate states
are numerical model states; they are not measured vibrational dynamics. The
spectral losses are inductive biases, not quantum-mechanical selection rules.

## Reproduction status

The audit is tied to Overleaf commit `b42f38bdc48aa32b04a250409a0ef2df00a43508`.
The manuscript in [`latex/`](latex/) is that draft, preserved for traceability.
Read [`REPRODUCIBILITY_AUDIT.md`](REPRODUCIBILITY_AUDIT.md) for the evidence and
known differences between the manuscript and executable protocols.

| Evidence | Verified scope |
|---|---|
| Principal SpectraDiT checkpoints | Six weights strictly loaded; 32 samples per model; 48 archived spectral anchors matched within 1e-4 normalized absolute tolerance |
| Saved prediction metrics | Full numerical recalculation of 260,386 principal/OOD target–prediction pairs; this is distinct from fresh inference |
| RRUFF experimental checkpoint | All 147 external pairs freshly inferred; physical-order restoration and AsLS baseline correction checked |
| Corrected NIST subset | 33 available identity-correct rows freshly inferred; most molecular identities are in the QM9S training partition |
| Four training seeds / full principal inference | Not established by the current archived weights and completed audit |
| New reviewer experiments | Preparation and matched-training code provided; training/robustness/calibration results are not claimed complete |

The original 84-pair NIST identity matching is invalid and is excluded from the
new figures. The old principal preprocessing permutes local wavenumber adjacency;
existing weights must retain that preprocessing. Ordered retraining is a separate
experiment. The archived QM9 OOD output evaluates 4,004 rows, whereas the supplied
test file contains 26,687 rows; the evaluator exposes both protocols explicitly.

## Repository contents

```text
src/                           models, training, evaluation, data preparation
src/reproduce_paper_checkpoints.py
src/audit_paper_reproducibility.py
src/audit_experimental_pairing.py
src/redraw_audited_results.py
src/export_reproduction_figure_bundle.py
src/reviewer_experiments/       new fixed-group-split / matched-training framework
configs/                       explicit task paths and training profiles
figures/reproduction/          four audited Results figures: PDF, SVG, 300 dpi PNG
figures/reproduction/source_bundle/  compact numeric inputs, without model weights
reproduction_audit/             completed numerical evidence and provenance
latex/                         full manuscript source, bibliography and figure assets
elucidation/                   optional spectrum-to-SMILES research module
run_reproduction_audit_gpu.sh   chemagent_gpu_pool worker launcher
```

Existing peak-resolved analysis, scaffold plots, UMAP, downstream summaries and
experimental figure scripts remain in `src/` and at the repository root.
Scaffold-class plots summarize subgroups; they do not establish scaffold-disjoint
generalization. The optional elucidation module is not part of the verified paper
benchmark and its weights are excluded.

## Installation

Use Python 3.10 or later. The checkpoint audit used Python 3.10 and
PyTorch 2.6.0 with CUDA 12.4; the recorded numerical versions are in
[`requirements-reproduction.txt`](requirements-reproduction.txt).

```bash
git clone https://github.com/jiaqingxie/SpectraFlow.git
cd SpectraFlow
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the recorded numerical stack, use `pip install -r requirements-reproduction.txt`.
Install a CUDA-enabled PyTorch 2.6.0 build appropriate for the worker. Raw LMDB
or Parquet ingestion additionally uses `requirements-raw-data.txt`. UMAP is an
optional dependency for the original representation figures. R-based OpenSpecy
preparation requires R and the packages used by `src/prepare_openspecy_benchmark.R`.

## Redraw the new Results figures without weights or full datasets

```bash
python src/redraw_audited_results.py \
  --source-bundle figures/reproduction/source_bundle \
  --output-dir figures/redrawn
```

This produces all four figure families in PDF, SVG and 300 dpi PNG, plus source
tables and captions. [`figure_manifest.json`](figures/reproduction/figure_manifest.json)
states the provenance and example-selection rules. The compact bundle contains
four quantile-selected QM9S curves, the restored RRUFF evaluation arrays and
aggregate metric/probe tables. It contains no neural-network weights.

| Figure | Numerical content |
|---|---|
| [Results 1](figures/reproduction/result_1_translation.pdf) | Principal saved seed-2 means and individual archived OOD comparisons; no unverified seed error bars |
| [Results 2](figures/reproduction/result_2_quantile_examples.pdf) | Examples nearest the 50th/90th nMAE percentiles of the full saved QM9S test results |
| [Results 3](figures/reproduction/result_3_downstream.pdf) | Five-split ridge summaries and signed Flow+fingerprint gains, including decreases |
| [Results 4](figures/reproduction/result_4_rruff_external.pdf) | All 147 RRUFF pairs and explicitly selected high-correlation examples |

To regenerate the compact bundle from the original local numerical artifacts:

```bash
python src/export_reproduction_figure_bundle.py
```

That export requires the archived `results/`, processed QM9S HDF5 and audit tables.
The redraw command above only requires the published compact bundle. Figure
styling follows Chen Liu's
[scientific-figure-making skill](https://github.com/ChenLiu-1996/figures4papers/tree/main/scientific-figure-making).

## Data layout and identity metadata

Obtain complete datasets from their original sources. Paths in
[`configs/checkpoint_tasks.json`](configs/checkpoint_tasks.json) are relative to
the repository root; edit them to use another storage layout.

| Input | Expected path / local processed size |
|---|---|
| QM9S | `data/processed/{ir,raman}_broaden_processed.h5`; 129,817 rows × 3600 points |
| QMe14S | `../data/QMe14S/processed/{ir,raman}_broaden_processed.h5`; 186,102 rows × 3600 points |
| ViBench-Full | `data/processed/vibench_test_full_{ir,raman}_processed.h5`; locally merged 91,949 rows × 1024 points |
| ViBench domains | `data/processed/{domain}_test_{ir,raman}_processed.h5`, including source-domain `qm9` |
| RRUFF | `datasets/openspecy_rruff_576/{train,test}_{ir,raman}.h5` and aligned `*_pairs.csv` |
| QM9S molecular identity | Original `../data/qm9s/mapping.txt`, IDs 1…129817 in the HDF5 row order |
| QMe14S identity | `data/processed/qme14s_id_smiles.csv`, with sequential `row_index` |
| ViBench identity | Aligned `{domain}_test_{ir,raman}_smiles.txt` lists |

HDF5 files contain `spectra` (N × L) and `x_axis` (L). Source and target rows must
represent the same identity; shape equality alone does not validate identity.
The audit found 256 all-zero QMe14S test rows excluded by the old evaluator; all
compared methods must apply the same declared quality rule and denominator.

The raw-data parent used by identity/audit scripts can be changed without editing
code:

```bash
export SPECTRAFLOW_RAW_DATA_ROOT=/path/to/raw-data
```

Its default is `../data` relative to the checkout. For raw preparation, use
`src/process.py`, `src/process_qme14s.py`, `src/process_all_lmdb.py` and
`src/merge_vibench_test_h5.py`. Do not substitute QMe14S `IR_broaden.zip` metadata
for QM9S molecular identities.

## Checkpoint configuration and fresh inference

No `.pt`, `.pth`, `.ckpt` or safetensors files are tracked. Expected names:

| Model | Weight path |
|---|---|
| Principal Flow | `checkpoints/{qm9s,qme14s,vibench}/flow_{ir2raman,raman2ir}_vibradit_best_seed2.pt` |
| OOD Flow | `checkpoints/vibench_ood/flow_{direction}_vibradit_best_seed{0,1}.pt` |
| Existing Direct | `checkpoints/qm9s_direct/direct_{direction}_vibradit_best_seed2.pt` |
| Experimental RRUFF | `checkpoints/openspecy_rruff_576_ordered_p1/flow_ir2raman_vibradit_best_seed42.pt` |

These are naming/path conventions, not download links. Original pretrained
weights are not distributed by this commit. Width, depth and patch size are
inferred from checkpoint tensors and strictly checked on load. Attention-head
count cannot be inferred from tensor shapes: pass the original run's value using
`--heads`; the documented profile uses 6, and original OOD configuration provenance
still needs confirmation.

```bash
python src/reproduce_paper_checkpoints.py \
  --task-config configs/checkpoint_tasks.json --list-tasks

python src/reproduce_paper_checkpoints.py \
  --task-config configs/checkpoint_tasks.json \
  --suite id --device cuda --heads 6 --steps 8 \
  --output-dir results/checkpoint_reproduction/id

python src/reproduce_paper_checkpoints.py \
  --task-config configs/checkpoint_tasks.json \
  --suite ood --protocol paper --device cuda --heads 6 \
  --output-dir results/checkpoint_reproduction/ood_full
```

`--protocol archived` reproduces the old QM9 OOD re-split; `paper` evaluates the
supplied OOD files in full. `--limit 32` is a labeled subset check, not a full-test
estimate. Evaluation saves every selected HDF5 row ID, per-molecule metrics,
checkpoint SHA256, solver settings and a first-batch prediction archive. It does
not save all predicted spectra. Principal archived anchors are compared when the
original saved predictions are available.

CUDA is the default and fails explicitly if unavailable. A CPU fallback must be
requested with `--device cpu --threads 32`; do not compare CPU timing with GPU
timing. Native-intensity errors use reference target extrema for evaluation.
With an absent target modality, the model predicts normalized shape unless an
independent intensity calibration is available.

The main checkpoint tensor shapes observed in the audit are:

| Dataset | Width / blocks | Patch size, IR→Raman / Raman→IR | Input order |
|---|---|---|---|
| QM9S | 384 / 8 | 20 / 20 | Legacy patch permutation |
| QMe14S | 384 / 8 | 20 / 20 | Legacy patch permutation |
| ViBench-Full | 384 / 8 | 20 / 16 | Legacy patch permutation |
| Experimental RRUFF | 384 / 8 | 4 / — | Physical wavenumber order |

These observed shapes take precedence over assuming one patch size for every
archived run. The full original training configuration is not stored in most
principal checkpoints.

## GPU pool launcher

Run from a terminal authenticated for the workload pool:

```bash
REPRO_PYTHON=/path/to/cuda-env/bin/python \
  bash run_reproduction_audit_gpu.sh \
  --task-config configs/checkpoint_tasks.json --suite id
```

The launcher requests 8 CPU cores, 1 GPU, 80000 MiB memory and
`--charged-group=chemagent_gpu_pool`, with the shared `xiejiaqing` GPFS mount.
`REPRO_CPUS`, `REPRO_CHARGED_GROUP` and `REPRO_PYTHON` override the defaults.
Authentication remains the responsibility of the invoking terminal.

## Explicit training profiles

[`configs/training_defaults.json`](configs/training_defaults.json) records the
profiles below. They are explicit run settings, not a claim that every archived
checkpoint was trained with all the same settings.

| Parameter | Principal/new matched-training profile |
|---|---|
| SpectraDiT width / blocks / heads | 384 / 8 / 6 |
| Optimizer / learning rate | Adam / 2e-4, cosine decay |
| Batch size / gradient clipping | 32 / 1.0 |
| New QM9S/QMe14S/ViBench training budget | 100 epochs; also compare a separately declared equal-training-time budget |
| Velocity L1/L2 mixture | 0.6 / 0.4 |
| Amplitude / gradient / curvature weighting | 1.0 / 0.5 / 0.5 |
| Endpoint / shape / local OT coefficients | 0.5 / 0.05 / 0.02 |
| Endpoint sampling probability / OT window | 0.1 / 64 points |
| Endpoint training / inference solver | Euler, 8 network evaluations |
| New data split / initialization seeds | Fixed split seed 2; initialization seeds 0, 1, 2, 3 |

The old nonnegative loss is zero after output projection; no independent benefit
is attributed to it. Increasing endpoint probability to 0.5 changes sampling
frequency, while inverse-probability weighting keeps the expected coefficient
fixed. A deterministic paired straight-line path is used; `sigma_min` is retained
as a compatibility argument and does not add noise in this implementation.

For an explicitly configured legacy-order QM9S training run:

```bash
python src/train_flow.py \
  --data_dir data/processed --save_dir checkpoints/qm9s \
  --no_dataset_subdir --source_mode ir --target_mode raman \
  --heatmap_size 3600 --resize_shape 60 60 \
  --batch_size 32 --epochs 100 --learning_rate 2e-4 --seed 2 \
  --backbone vibradit --dit_patch_size 20 --dit_hidden_dim 384 \
  --dit_depth 8 --dit_num_heads 6 \
  --use_mixed_loss --gen_loss_weight 0.5 --gen_loss_prob 0.1 \
  --train_gen_steps 8 --val_num_steps 8
```

`--preserve_spectral_order` changes preprocessing and must be used consistently
for newly trained models and their evaluation. Do not apply it to existing
legacy-order weights. The original loader's `--seed` controls both splitting and
model initialization; use the new framework below for fixed-split seed studies.
Checkpoint selection minimizes validation `0.6*MAE + 0.4*MSE`.

## New independent splits and matched Flow/Direct training

The framework is preparatory code; the proposed experiment matrix has not been
completed. It checks aligned metadata, declares constant/nonfinite-spectrum
exclusions, and partitions entire identities/scaffolds/RRUFF IDs/mineral names.
Its new protocol is separate from the archived paper's random split.

```bash
python src/complete_reviewer_experiments.py prepare
python src/complete_reviewer_experiments.py train \
  --run qm9s_identity_ir2raman_flow_full_seed0 --device cuda
python src/complete_reviewer_experiments.py train \
  --run qm9s_identity_ir2raman_direct_full_seed0 --device cuda
python src/complete_reviewer_experiments.py train \
  --run qm9s_identity_ir2raman_flow_full_seed0 --device cuda --resume
```

Preparation requires all datasets and identity files in the layout above,
including OpenSpecy material metadata at `datasets/openspecy_cov100/metadata.csv`.
Splits, configs and generated weights go to the ignored root
`reviewer_experiments/` directory. Group assignment uses deterministic SHA256
thresholds with expected 70% train, 10% validation, 5% calibration, 15% test;
large scaffold groups can change the achieved fractions. Acyclic compounds are
grouped by their element-agnostic full graph topology rather than one empty
Murcko scaffold.

Flow and Direct share architecture, initialization, batch order, projected
endpoint losses and checkpoint-selection metric. Runs save split/code hashes,
optimizer/RNG states, update counts, examples seen and synchronized training time.
Use `--budget-seconds VALUE` for a separate equal-training-time comparison;
equal update counts are not equal compute. Available Flow variants include
`no_peak`, `no_shape`, `no_ot`, `basic` and `legacy_order`. Robustness, NFE curves,
downstream-use experiments and uncertainty calibration remain outstanding.

## Other analyses and manuscript build

- Baselines: `src/train.py`, `src/train_seq2seq.py`, `src/train_direct.py`;
  the existing shorter-budget Direct result does not isolate integration alone.
- Peak fidelity: `src/evaluate_peak_resolved_spectra.py` and
  `src/plot_bidirectional_peak_resolved_summary.py`.
- Properties: `src/extract_flow_embeddings_and_properties.py`,
  `src/train_downstream.py`, `run_vibench_ood_downstream.sh` and UMAP scripts.
  Probes are fitted separately in each domain; this is not zero-shot property prediction.
- Experimental data: OpenSpecy pairing/repeatability scripts and
  `src/audit_experimental_pairing.py`.
- Corrected NIST matching: `src/build_nist_qm9s_overlap.py` requires
  `--qm9s-mapping-txt`; the invalid ZIP-based matching interface was removed.
- Manuscript: follow [`latex/README.md`](latex/README.md) to build the complete
  draft from the included source and figures. The compiled
  [archived draft PDF](latex/manuscript_snapshot.pdf) is also included.

To audit existing saved numerical artifacts without neural inference:

```bash
python src/audit_paper_reproducibility.py \
  --output-dir results/audit \
  --recompute results/verify_fig1_seed2/qm9s_flow_ir2raman_vibradit_seed2
python src/audit_experimental_pairing.py
```

The first command requires the archived predictions for recalculation; the
second requires the original identity mapping, raw metadata and saved experimental
artifacts. Published audit files already document completed checks.

## Dataset attribution

Full spectra and pretrained models must be obtained separately. The compact
figure bundle contains selected numerical figure inputs with provenance, not the
complete third-party datasets. Cite and follow each resource's distribution terms:

- [QM9S](https://figshare.com/articles/dataset/QM9S_dataset/24235333).
- QMe14S: Yuan et al., QMe14S dataset release (2025); see the manuscript bibliography.
- [ViBench / Vib2Mol](https://arxiv.org/abs/2503.07014).
- [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/), SRD 69.
- [OpenSpecy](https://doi.org/10.1021/acs.analchem.1c00123).
- [RRUFF](https://rruff.info/), mineral spectral records.

The accompanying manuscript is an unpublished draft. A publication citation will
be added when available; this repository does not establish an acceptance status.
