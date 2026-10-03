"""Audit experimental identities and recalculate saved predictions, without inference.

RDKit is used only for molecular-identity checks. The corrected NIST manifest
contains matches, not new predictions or a validated wavenumber convention.
"""
import argparse
import csv
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy import sparse
from scipy.sparse.linalg import spsolve

from audit_paper_reproducibility import legacy_order, metric_arrays

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = Path(os.environ.get('SPECTRAFLOW_RAW_DATA_ROOT', str(ROOT.parent/'data'))).expanduser().resolve()


def canonical(smiles, stereo=True):
    if not isinstance(smiles, str) or not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=stereo) if mol else ""


def asls_correct(y, smoothness=1e5, asymmetry=.01, iterations=10):
    n = len(y)
    difference = sparse.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)],
                              [0, 1, 2], shape=(n-2, n), format="csc")
    penalty = smoothness * (difference.T @ difference)
    weights = np.ones(n)
    for _ in range(iterations):
        baseline = spsolve(sparse.diags(weights, format="csc") + penalty, weights*y)
        weights = asymmetry*(y > baseline) + (1-asymmetry)*(y <= baseline)
    return y-baseline


def audit_nist_identities(output):
    mapping = [row.split("\t", 1) for row in (RAW_DATA/"qm9s/mapping.txt").read_text().splitlines()]
    with h5py.File(ROOT/"data/processed/ir_broaden_processed.h5") as f:
        if len(mapping) != len(f["spectra"]):
            raise ValueError("QM9S mapping does not have the same row count as HDF5")
    if [int(row[0]) for row in mapping] != list(range(1, len(mapping)+1)):
        raise ValueError("Verify original QM9S identifiers before using positional mapping")
    nist = ROOT/"results/nist_ftir_qm9s_overlap"
    matches = pd.read_csv(nist/"all_matches_matches.csv")
    matches["actual_qm9s_id"] = [mapping[int(i)][0] for i in matches.qm9s_index]
    matches["actual_qm9s_smiles"] = [mapping[int(i)][1] for i in matches.qm9s_index]
    verified = []
    for identifier, smiles in zip(matches.actual_qm9s_id, matches.actual_qm9s_smiles):
        with (RAW_DATA/"qm9s/qm9s_csv"/(identifier+".csv")).open() as f:
            reader = csv.reader(f)
            next(reader)
            verified.append(canonical(next(reader)[3]) == canonical(smiles))
    matches["mapping_agrees_with_original_molecule_file"] = verified
    if not all(verified):
        raise ValueError("QM9S mapping and original molecular files disagree")
    matches["source_canonical"] = [canonical(s) for s in matches.ftir_smiles]
    matches["claimed_target_canonical"] = [canonical(s) for s in matches.qm9s_smiles]
    matches["actual_target_canonical"] = [canonical(s) for s in matches.actual_qm9s_smiles]
    matches["actual_identity_match_stereo"] = matches.source_canonical == matches.actual_target_canonical
    matches["actual_identity_match_connectivity"] = [
        canonical(a, False) == canonical(b, False)
        for a, b in zip(matches.ftir_smiles, matches.actual_qm9s_smiles)
    ]
    matches.to_csv(output/"nist_identity_audit.csv", index=False)
    with zipfile.ZipFile(RAW_DATA/"IR_broaden.zip") as z:
        members = [name for name in z.namelist() if name.endswith(".csv")]
        lines = z.read("IR_broaden/IR_000000.csv").decode().strip().splitlines()
        raw = np.array([float(x) for x in lines[-1].split(",")], dtype=np.float32)
    interpolated = np.interp(np.linspace(0, 1, 3600), np.linspace(0, 1, len(raw)), raw).astype(np.float32)
    with h5py.File(RAW_DATA/"QMe14S/processed/ir_broaden_processed.h5") as q, \
            h5py.File(ROOT/"data/processed/ir_broaden_processed.h5") as s:
        qme_error = float(np.max(np.abs(interpolated-q["spectra"][0])))
        qm9_error = float(np.max(np.abs(interpolated-s["spectra"][0])))
    return mapping, dict(n=len(matches), actual_stereo_identity_matches=int(matches.actual_identity_match_stereo.sum()),
                         actual_connectivity_identity_matches=int(matches.actual_identity_match_connectivity.sum()),
                         raw_qm9s_molecule_files_verified=int(sum(verified)),
                         wrong_smiles_archive_n=len(members), wrong_smiles_archive_first_smiles=lines[0],
                         archive_first_spectrum_max_abs_difference_qme14s=qme_error,
                         archive_first_spectrum_max_abs_difference_qm9s=qm9_error,
                         conclusion="Archived NIST identity matching uses QMe14S SMILES indices for QM9S spectra")


def corrected_nist_manifest(mapping, output):
    import torch
    permutation = torch.randperm(len(mapping), generator=torch.Generator().manual_seed(2)).numpy()
    labels = np.empty(len(mapping), dtype=object)
    a, b = int(.7*len(mapping)), int(.15*len(mapping))
    labels[permutation[:a]], labels[permutation[a:a+b]], labels[permutation[a+b:]] = "train", "valid", "test"
    exact, connected = defaultdict(list), defaultdict(list)
    for i, (identifier, smiles) in enumerate(mapping):
        a, b = canonical(smiles), canonical(smiles, False)
        if a:
            exact[a].append(i)
            connected[b].append(i)
    metadata = pd.read_csv(ROOT/"datasets/ftir_pubchem_metadata.csv", dtype={"cid": str})
    metadata["split_priority"] = metadata["split"].map({"test": 0, "valid": 1, "train": 2}).fillna(9)
    metadata = metadata.sort_values(["split_priority", "cid"]).drop_duplicates("cid")
    records = []
    for record in metadata.to_dict("records"):
        smi = record.get("isomeric_smiles")
        if not isinstance(smi, str) or not smi:
            smi = record.get("canonical_smiles")
        a, b = canonical(smi), canonical(smi, False)
        if not a:
            continue
        candidates = exact.get(a, [])
        match_type = "exact_isomeric"
        if not candidates:
            candidates = connected.get(b, [])
            match_type = "connectivity"
        for index in candidates:
            records.append(dict(cid=record["cid"], ftir_split=record["split"], qm9s_index=index,
                                qm9s_original_id=mapping[index][0], qm9s_smiles=mapping[index][1],
                                qm9s_split_seed2=labels[index],
                                ftir_smiles=smi, match_type=match_type, match_multiplicity=len(candidates),
                                title=record.get("title", ""), evidence_level="corrected_identity_only_no_inference"))
    corrected = pd.DataFrame(records)
    raw_dir = ROOT/"results/nist_ftir_qm9s_overlap/ftir_files"
    corrected["raw_spectrum_available_locally"] = corrected.cid.map(lambda cid: (raw_dir/(cid+".npy")).exists())
    corrected.to_csv(output/"corrected_nist_qm9s_identity_matches.csv", index=False)
    prepare_available_corrected_nist(corrected, output)
    return dict(n_match_rows=len(corrected), n_unique_cids=int(corrected.cid.nunique()) if len(corrected) else 0,
                n_unique_qm9s_rows=int(corrected.qm9s_index.nunique()) if len(corrected) else 0,
                locally_available_rows=int(corrected.raw_spectrum_available_locally.sum()),
                qm9s_split_counts_seed2=corrected.qm9s_split_seed2.value_counts().to_dict(),
                note="Only available raw files are materialized; ascending 399-4000 source grid is an explicit protocol assumption")


def prepare_available_corrected_nist(corrected, output):
    """Rebuild identity-correct input/target pairs with the paper's axis assumption."""
    selected = corrected[corrected.raw_spectrum_available_locally].reset_index(drop=True)
    raw_dir = ROOT/"results/nist_ftir_qm9s_overlap/ftir_files"
    ir, raman, axis_records = [], [], []
    with h5py.File(ROOT/"data/processed/ir_broaden_processed.h5") as sf, \
            h5py.File(ROOT/"data/processed/raman_broaden_processed.h5") as tf:
        wave = tf["x_axis"][:]
        for record in selected.to_dict("records"):
            raw = np.load(raw_dir/(record["cid"]+".npy")).ravel()
            raw_wave = np.linspace(399., 4000., len(raw))
            ascending = np.interp(wave, raw_wave, raw, left=0., right=0.).astype(np.float32)
            descending = np.interp(wave, raw_wave, raw[::-1], left=0., right=0.).astype(np.float32)
            computed_ir = sf["spectra"][record["qm9s_index"]]
            axis_records.append(dict(cid=record["cid"], qm9s_index=record["qm9s_index"],
                                     ascending_ir_pearson=float(np.corrcoef(ascending, computed_ir)[0, 1]),
                                     descending_ir_pearson=float(np.corrcoef(descending, computed_ir)[0, 1])))
            ir.append(ascending)
            raman.append(tf["spectra"][record["qm9s_index"]])
    for mode, arrays in [("ir", ir), ("raman", raman)]:
        with h5py.File(output/f"corrected_nist_available_{mode}.h5", "w") as f:
            f.create_dataset("spectra", data=np.stack(arrays), compression="gzip")
            f.create_dataset("x_axis", data=wave)
            f.attrs["source_axis_assumption"] = "NIST array ascending 399-4000 cm^-1, as in manuscript"
            f.attrs["identity_manifest"] = "corrected_nist_available_pairs.csv"
    selected.to_csv(output/"corrected_nist_available_pairs.csv", index=False)
    pd.DataFrame(axis_records).to_csv(output/"corrected_nist_source_axis_sensitivity.csv", index=False)


def saved_arrays(directory):
    p = pd.read_csv(directory/"flow_ir2raman_preds.csv")
    y = pd.read_csv(directory/"flow_ir2raman_targets.csv")
    if not np.array_equal(p["index"], y["index"]) or list(p.columns) != list(y.columns):
        raise ValueError("Prediction/reference indices or columns differ")
    return y["index"].to_numpy(), y.iloc[:, 1:].to_numpy(), p.iloc[:, 1:].to_numpy()


def audit_saved_experiments(output):
    directory = ROOT/"results/openspecy_rruff_576_ordered_p1/flow_ir2raman"
    indices, y, p = saved_arrays(directory)
    with h5py.File(ROOT/"datasets/openspecy_rruff_576/test_raman.h5") as f:
        actual = f["spectra"][:][indices]
        wave = f["x_axis"][:]
    # The ordered-training evaluator always applies legacy inverse heatmap
    # conversion. Undo that erroneous conversion before local baseline fitting.
    order = legacy_order(24)
    restored = y[:, order]
    restore_error = float(np.max(np.abs(restored-actual)))
    if restore_error > 1e-5:
        raise ValueError("RRUFF saved targets cannot be aligned to physical wavenumbers")
    y, p = restored, p[:, order]
    metrics = pd.DataFrame(metric_arrays(y, p))
    metrics.insert(0, "index", indices)
    yc = np.stack([asls_correct(row) for row in y])
    pc = np.stack([asls_correct(row) for row in p])
    metrics["pearson_baseline_corrected"] = metric_arrays(yc, pc)["pearson"]
    pairs = pd.read_csv(ROOT/"datasets/openspecy_rruff_576/test_pairs.csv").iloc[indices].reset_index(drop=True)
    metadata = pd.read_csv(ROOT/"datasets/openspecy_cov100/metadata.csv")
    material_by_id = metadata.dropna(subset=["rruffid"]).assign(
        identity=lambda frame: frame.rruffid.str.lower()).drop_duplicates("identity").set_index("identity").spectrum_identity
    metrics["rruff_id"] = pairs.identity_key
    metrics["mineral_name"] = pairs.identity_key.map(material_by_id)
    metrics.to_csv(output/"rruff_recomputed_per_pair.csv", index=False)
    np.savez_compressed(output/"rruff_physical_order_predictions.npz", target=y, prediction=p, x_axis=wave, indices=indices)
    train = pd.read_csv(ROOT/"datasets/openspecy_rruff_576/train_pairs.csv")
    train_ids, test_ids = set(train.identity_key), set(pairs.identity_key)
    train_names = set(train.identity_key.map(material_by_id).dropna())
    test_names = set(metrics.mineral_name.dropna())
    summary = dict(n=len(metrics), pearson_median=float(metrics.pearson.median()), pearson_mean=float(metrics.pearson.mean()),
                   pearson_baseline_corrected_median=float(metrics.pearson_baseline_corrected.median()),
                   r2_mean=float(metrics.r2.mean()), target_restore_max_abs_error=restore_error,
                   development_pair_n=len(train), development_identity_n=len(train_ids),
                   test_identity_n=len(test_ids), identity_overlap_n=len(train_ids & test_ids),
                   development_mineral_names_n=len(train_names), test_mineral_names_n=len(test_names),
                   mineral_name_overlap_n=len(train_names & test_names),
                   asls=dict(smoothness=1e5, asymmetry=.01, iterations=10), evidence_level="saved_predictions_recomputed")
    idx, ny, npred = saved_arrays(ROOT/"results/nist_ftir_qm9s_overlap/eval_all84_aligned")
    nm = pd.DataFrame(metric_arrays(ny, npred))
    nm.insert(0, "index", idx)
    nm["valid_identity_matched_evaluation"] = False
    nm.to_csv(output/"nist_archived_recomputed_invalid_identity.csv", index=False)
    return dict(rruff=summary, nist_archived_invalid_identity=dict(n=len(nm), pearson_median=float(nm.pearson.median()),
                pearson_mean=float(nm.pearson.mean()), r2_mean=float(nm.r2.mean()),
                evidence_level="saved_predictions_recomputed_identity_invalid"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"reproduction_audit/experimental")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    mapping, identities = audit_nist_identities(args.output_dir)
    print(json.dumps(identities), flush=True)
    corrected = corrected_nist_manifest(mapping, args.output_dir)
    print(json.dumps(corrected), flush=True)
    summary = dict(nist_identity_audit=identities, corrected_nist_manifest=corrected,
                   **audit_saved_experiments(args.output_dir))
    (args.output_dir/"experimental_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
