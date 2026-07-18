"""Build identity-matched experimental NIST FTIR / computed QM9S Raman subsets."""
from __future__ import annotations

import argparse
import csv
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from rdkit import Chem


SPLIT_PRIORITY = {"test": 0, "valid": 1, "train": 2}


def canonical_smiles(smiles: str, isomeric: bool) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def qm9s_split_labels(n: int, seed: int) -> np.ndarray:
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    permutation = torch.randperm(
        n, generator=torch.Generator().manual_seed(seed)
    ).numpy()
    labels = np.empty(n, dtype=object)
    labels[permutation[:train_size]] = "train"
    labels[permutation[train_size : train_size + val_size]] = "valid"
    labels[permutation[train_size + val_size :]] = "test"
    return labels


def read_qm9s_smiles(zip_path: Path, count: int, prefix: str) -> list[str]:
    smiles = []
    with zipfile.ZipFile(zip_path) as archive:
        for index in range(count):
            member = f"{prefix}{index:06d}.csv"
            with archive.open(member) as handle:
                smiles.append(handle.readline().decode("utf-8").strip())
    return smiles


def build_matches(
    metadata_path: Path,
    qm9s_smiles: list[str],
    split_labels: np.ndarray,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, dtype={"cid": str})
    metadata["split_priority"] = metadata["split"].map(SPLIT_PRIORITY).fillna(9)
    metadata = (
        metadata.sort_values(["split_priority", "cid"])
        .drop_duplicates("cid", keep="first")
        .copy()
    )

    exact_map: dict[str, list[int]] = defaultdict(list)
    connectivity_map: dict[str, list[int]] = defaultdict(list)
    for index, smiles in enumerate(qm9s_smiles):
        exact_map[canonical_smiles(smiles, isomeric=True)].append(index)
        connectivity_map[canonical_smiles(smiles, isomeric=False)].append(index)

    rows = []
    invalid_ftir = 0
    for record in metadata.to_dict("records"):
        source_smiles = record.get("isomeric_smiles") or record.get(
            "canonical_smiles", ""
        )
        exact_key = canonical_smiles(source_smiles, isomeric=True)
        connectivity_key = canonical_smiles(source_smiles, isomeric=False)
        if not connectivity_key:
            invalid_ftir += 1
            continue

        candidates = exact_map.get(exact_key, [])
        match_type = "exact_isomeric"
        if not candidates:
            candidates = connectivity_map.get(connectivity_key, [])
            match_type = "connectivity"
        if not candidates:
            continue

        for qm9s_index in candidates:
            rows.append(
                {
                    "cid": str(record["cid"]),
                    "ftir_split": record["split"],
                    "qm9s_index": qm9s_index,
                    "qm9s_split": split_labels[qm9s_index],
                    "match_type": match_type,
                    "match_multiplicity": len(candidates),
                    "canonical_key": connectivity_key,
                    "ftir_smiles": source_smiles,
                    "qm9s_smiles": qm9s_smiles[qm9s_index],
                    "title": record.get("title", ""),
                }
            )

    matches = pd.DataFrame(rows)
    if matches.empty:
        raise RuntimeError("No NIST FTIR / QM9S SMILES overlap was found.")
    matches = matches.sort_values(
        ["qm9s_split", "qm9s_index", "cid"]
    ).reset_index(drop=True)
    print(f"Unique NIST CIDs checked: {len(metadata)}")
    print(f"Invalid NIST SMILES: {invalid_ftir}")
    print(f"Matched rows: {len(matches)}")
    print(f"Unique matched NIST CIDs: {matches['cid'].nunique()}")
    print(f"Unique matched QM9S rows: {matches['qm9s_index'].nunique()}")
    print(f"QM9S split counts: {matches['qm9s_split'].value_counts().to_dict()}")
    print(f"Match types: {matches['match_type'].value_counts().to_dict()}")
    print(
        "Ambiguous rows:",
        int((matches["match_multiplicity"] > 1).sum()),
    )
    return matches


def resample_ftir(
    spectrum: np.ndarray,
    target_axis: np.ndarray,
    high_wavenumber: float,
    low_wavenumber: float,
) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=np.float64).ravel()
    # Identity-matched QM9S IR confirms that the Fcg-Former/NIST arrays run
    # from low to high wavenumber (399 -> 4000 cm^-1).
    source_axis = np.linspace(
        low_wavenumber, high_wavenumber, spectrum.size
    )
    target_axis = np.asarray(target_axis, dtype=np.float64).ravel()
    values = np.interp(
        target_axis,
        source_axis,
        spectrum,
        left=0.0,
        right=0.0,
    )
    return values.astype(np.float32)


def write_placeholder_csv(path: Path, axis: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", *np.asarray(axis).tolist()])


def write_h5(
    path: Path,
    spectra: np.ndarray,
    axis: np.ndarray,
    ids: list[str],
    smiles: list[str],
) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("spectra", data=spectra, compression="gzip")
        handle.create_dataset("x_axis", data=np.asarray(axis, dtype=np.float32))
        handle.create_dataset(
            "ids", data=np.asarray(ids, dtype=object), dtype=string_dtype
        )
        handle.create_dataset(
            "smiles",
            data=np.asarray(smiles, dtype=object),
            dtype=string_dtype,
        )


def materialize_subset(
    matches: pd.DataFrame,
    ftir_dir: Path,
    raman_h5_path: Path,
    output_dir: Path,
    subset_name: str,
    high_wavenumber: float,
    low_wavenumber: float,
) -> None:
    selected = matches.copy()
    if subset_name == "qm9s_test":
        selected = selected[selected["qm9s_split"] == "test"].copy()
    elif subset_name.startswith("nist_"):
        nist_split = subset_name.removeprefix("nist_")
        selected = selected[selected["ftir_split"] == nist_split].copy()
    if selected.empty:
        raise RuntimeError(f"No rows available for subset {subset_name!r}.")

    # Avoid weighting duplicate CIDs or duplicate QM9S rows more than once.
    selected = selected.sort_values(
        ["match_multiplicity", "qm9s_index", "cid"]
    ).drop_duplicates(["cid", "qm9s_index"])

    with h5py.File(raman_h5_path, "r") as raman_h5:
        target_axis = raman_h5["x_axis"][:]
        qm9s_raman = raman_h5["spectra"]
        sources = []
        targets = []
        kept_rows = []
        for record in selected.to_dict("records"):
            split_path = ftir_dir / record["ftir_split"] / f"{record['cid']}.npy"
            flat_path = ftir_dir / f"{record['cid']}.npy"
            spectrum_path = split_path if split_path.exists() else flat_path
            if not spectrum_path.exists():
                print(f"Missing FTIR file, skipping: {spectrum_path}")
                continue
            sources.append(
                resample_ftir(
                    np.load(spectrum_path),
                    target_axis,
                    high_wavenumber,
                    low_wavenumber,
                )
            )
            targets.append(
                np.asarray(qm9s_raman[int(record["qm9s_index"])], dtype=np.float32)
            )
            kept_rows.append(record)

    if not kept_rows:
        raise RuntimeError("No matched FTIR files were found for materialization.")
    source_array = np.stack(sources)
    target_array = np.stack(targets)
    if source_array.shape != target_array.shape:
        raise ValueError(
            f"Aligned arrays differ: FTIR {source_array.shape}, "
            f"Raman {target_array.shape}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / subset_name
    ids = [str(row["cid"]) for row in kept_rows]
    smiles = [str(row["qm9s_smiles"]) for row in kept_rows]
    write_h5(
        prefix.with_name(prefix.name + "_ir.h5"),
        source_array,
        target_axis,
        ids,
        smiles,
    )
    write_h5(
        prefix.with_name(prefix.name + "_raman.h5"),
        target_array,
        target_axis,
        ids,
        smiles,
    )
    write_placeholder_csv(
        prefix.with_name(prefix.name + "_ir.csv"), target_axis
    )
    write_placeholder_csv(
        prefix.with_name(prefix.name + "_raman.csv"), target_axis
    )
    pd.DataFrame(kept_rows).to_csv(
        prefix.with_name(prefix.name + "_matches.csv"), index=False
    )
    print(
        f"Materialized {len(kept_rows)} rows for {subset_name}: "
        f"{source_array.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--qm9s-ir-zip", type=Path, required=True)
    parser.add_argument("--qm9s-ir-h5", type=Path, required=True)
    parser.add_argument("--qm9s-raman-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ftir-dir", type=Path)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--inner-prefix", default="IR_broaden/IR_")
    parser.add_argument("--ftir-high-wavenumber", type=float, default=4000.0)
    parser.add_argument("--ftir-low-wavenumber", type=float, default=399.0)
    parser.add_argument("--match-only", action="store_true")
    args = parser.parse_args()

    with h5py.File(args.qm9s_ir_h5, "r") as handle:
        count = int(handle["spectra"].shape[0])
        print(
            "QM9S IR:",
            handle["spectra"].shape,
            "axis:",
            float(handle["x_axis"][0]),
            "to",
            float(handle["x_axis"][-1]),
        )
    with h5py.File(args.qm9s_raman_h5, "r") as handle:
        print(
            "QM9S Raman:",
            handle["spectra"].shape,
            "axis:",
            float(handle["x_axis"][0]),
            "to",
            float(handle["x_axis"][-1]),
        )

    qm9s_smiles = read_qm9s_smiles(
        args.qm9s_ir_zip, count, args.inner_prefix
    )
    matches = build_matches(
        args.metadata,
        qm9s_smiles,
        qm9s_split_labels(count, args.seed),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    match_path = args.output_dir / "nist_qm9s_smiles_matches.csv"
    matches.to_csv(match_path, index=False)
    print(f"Saved matches: {match_path}")

    if args.match_only:
        return
    if args.ftir_dir is None:
        parser.error("--ftir-dir is required unless --match-only is used.")

    materialize_subset(
        matches,
        args.ftir_dir,
        args.qm9s_raman_h5,
        args.output_dir,
        "qm9s_test",
        args.ftir_high_wavenumber,
        args.ftir_low_wavenumber,
    )
    for nist_split in ("train", "valid", "test"):
        materialize_subset(
            matches,
            args.ftir_dir,
            args.qm9s_raman_h5,
            args.output_dir,
            f"nist_{nist_split}",
            args.ftir_high_wavenumber,
            args.ftir_low_wavenumber,
        )
    materialize_subset(
        matches,
        args.ftir_dir,
        args.qm9s_raman_h5,
        args.output_dir,
        "all_matches",
        args.ftir_high_wavenumber,
        args.ftir_low_wavenumber,
    )


if __name__ == "__main__":
    main()
