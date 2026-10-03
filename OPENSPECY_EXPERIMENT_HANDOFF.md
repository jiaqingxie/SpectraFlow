# OpenSpecy experiment handoff

## Current conclusion

- The source library was loaded through the OpenSpecy R package, but strict
  cross-modal pairing by `rruffid` reduces it to the RRUFF mineral subset.
- Raw OpenSpecy contains 44,401 rows (22,585 FTIR; 20,925 Raman).
- Strict shared `rruffid`: 743 identities and 5,555 records, all from the
  `rruff` organization and `minerals` library type.
- The R audit confirmed every generated pair has the correct modality,
  `rruffid`, and material name.
- Numerical R-to-HDF5 audit is exact: correlation 1.0 and maximum absolute
  error around 1e-6 to 4e-6. Pairing and HDF5 row alignment are not the cause.

## Model results

- Identity-disjoint SpectraFlow IR->Raman:
  - R2: -1.0413
  - Pearson: 0.1926
- Identity-disjoint VAE:
  - R2: -1.2523
  - Pearson: 0.2580
- Within-identity VAE (deterministic latent mean):
  - Test R2: -1.0226; Pearson: 0.2878
  - Training-set R2: -0.8961; Pearson: 0.3347
- VAE inference was fixed to use latent `mu` instead of random sampling.

## Repeatability diagnosis

The first within-identity split leaked duplicate spectral content:

- nearest-replicate oracle median R2 = 1.0

The builder was then changed to group duplicate Raman spectra before splitting.
Deduplicated result:

- 737 train identities
- 73 test identities
- 1,928 unique training pairs
- 73 test representatives
- 20 duplicate test records excluded

Deduplicated Raman repeatability upper bounds:

- global train mean: mean R2 -1.5045; Pearson 0.0597
- same-identity mean: mean R2 -4.0813; Pearson -0.1530
- nearest same-identity oracle: mean R2 -4.0640; Pearson -0.1452

Therefore, different Raman measurements under the same `rruffid` are not a
stable pointwise target after current preprocessing. OpenSpecy/RRUFF is not
suitable for deterministic sample-wise IR->Raman regression without additional
condition metadata and stronger Raman preprocessing. Prefer material retrieval
(Top-1/Top-5/MRR), classification, or same-modality robustness.

## High-R2 VAE examples

From the original identity-disjoint VAE result:

- index 25: Rutile, R050031, R2 0.817832
- index 29: Calcite, R050048, R2 0.766943
- index 9: Humite, R040071, R2 0.750828

Plot command:

```powershell
python -u src\plot_selected_openspecy_predictions.py --result_dir results\openspecy_rruff_576_vae_ordered --data_dir datasets\openspecy_rruff_576 --prefix vae_ir2raman --indices 25 29 9 --output results\openspecy_rruff_576_vae_ordered\vae_top3_raman_predictions.png
```

## Experimental paired-data candidates

- Thermo Nicolet matched FTIR/Raman collection: 3,119 commercial pairs.
- AIST SDBS: about 3,500 experimental Raman spectra plus FTIR under shared
  compound IDs; bulk access requires AIST permission (50/day public limit).
- VibraCLIP: 320 NIST-IR/OMNIC-Raman molecule pairs; Raman source is commercial.
- Academic datasets: 185 pairs available by author request, ETH 14-molecule
  public set, and a public WHO-antibiotics IR/Raman set.

## Added or changed files

- `src/audit_openspecy_pairing.R`
- `src/build_openspecy_identity_pairs.py`
- `src/eval_openspecy_repeatability.py`
- `src/eval_openspecy_retrieval.py`
- `src/plot_selected_openspecy_predictions.py`
- `src/model.py`
- `src/train.py`
- `src/train_flow.py`
- `src/test.py`

## Recommended next step

1. View the top-three VAE example plot.
2. Decide whether to:
   - preprocess Raman with baseline correction/outlier removal and repeat the
     oracle analysis, or
   - stop pointwise OpenSpecy generation and implement full-OpenSpecy
     material-level retrieval.
3. For a genuine experimental paired benchmark, pursue AIST SDBS permission or
   institutional access to the Thermo Nicolet matched collection.
