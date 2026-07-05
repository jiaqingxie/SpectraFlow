"""
Export SMILES/IDs for the exact samples used by test.py/test_flow.py.

This script does not run model inference. It only reproduces the dataset order
and optional random_split test subset, then writes the corresponding metadata.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import h5py
import pandas as pd
import torch

from utils import paired_dataset_use_random_split_by_default


def decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_h5_metadata(h5_path: Path):
    if not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as f:
        n = int(f["spectra"].shape[0])
        ids = [str(i) for i in range(n)]
        smiles = [""] * n
        if "ids" in f:
            ids = [decode_value(v) for v in f["ids"][:]]
        if "smiles" in f:
            smiles = [decode_value(v) for v in f["smiles"][:]]
    return ids, smiles


def read_csv_ids(csv_path: Path):
    df = pd.read_csv(csv_path, header=None, usecols=[0])
    ids = [str(v) for v in df.iloc[1:, 0].tolist()]
    return ids


def read_dataset_length(source_csv: Path):
    source_h5 = Path(str(source_csv).replace(".csv", ".h5"))
    if source_h5.exists():
        with h5py.File(source_h5, "r") as f:
            return int(f["spectra"].shape[0])
    return len(read_csv_ids(source_csv))


def read_mapping_txt(path: Path):
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" in line:
                key, value = line.split("\t", 1)
            else:
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue
                key, value = parts
            mapping[str(key).strip()] = value.strip()
    return mapping


def read_metadata_csv(path: Path):
    ids = []
    smiles = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        lower_to_name = {name.lower(): name for name in (reader.fieldnames or [])}
        id_col = lower_to_name.get("id") or lower_to_name.get("index") or lower_to_name.get("sample_id")
        smiles_col = lower_to_name.get("smiles")
        if smiles_col is None:
            raise ValueError(f"metadata_csv must contain a smiles column: {path}")
        for i, row in enumerate(reader):
            ids.append(str(row[id_col]) if id_col else str(i))
            smiles.append(str(row[smiles_col]))
    return ids, smiles


def read_smiles_file(path: Path, n: int):
    with path.open("r", encoding="utf-8") as f:
        smiles = [line.rstrip("\n") for line in f]
    if len(smiles) != n:
        raise ValueError(f"smiles_file length {len(smiles)} != dataset length {n}: {path}")
    ids = [str(i) for i in range(n)]
    return ids, smiles


def build_metadata(args, source_csv: Path):
    n = read_dataset_length(source_csv)

    if args.metadata_csv:
        ids, smiles = read_metadata_csv(Path(args.metadata_csv))
    elif args.smiles_file:
        ids, smiles = read_smiles_file(Path(args.smiles_file), n)
    else:
        source_h5 = Path(str(source_csv).replace(".csv", ".h5"))
        meta = read_h5_metadata(source_h5)
        if meta is not None:
            ids, smiles = meta
        else:
            ids = read_csv_ids(source_csv)
            smiles = [""] * len(ids)

    if args.mapping_txt:
        mapping = read_mapping_txt(Path(args.mapping_txt))
        if not ids or len(ids) != n:
            ids = read_csv_ids(source_csv)
        smiles = [mapping.get(str(sample_id), "") for sample_id in ids]

    if len(ids) != n or len(smiles) != n:
        raise ValueError(
            f"Metadata length mismatch: ids={len(ids)}, smiles={len(smiles)}, dataset={n}. "
            "Use --metadata_csv/--smiles_file matching the processed dataset order."
        )
    return ids, smiles


def split_indices(n: int, seed: int, use_full_as_test: bool):
    if use_full_as_test:
        return list(range(n))
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    return perm[train_size + val_size:train_size + val_size + test_size]


def main():
    parser = argparse.ArgumentParser(description="Export test-set IDs/SMILES without inference")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--source_mode", type=str, required=True, choices=["ir", "uv", "raman"])
    parser.add_argument("--target_mode", type=str, required=True, choices=["ir", "uv", "raman"])
    parser.add_argument("--source_csv", type=str, default=None)
    parser.add_argument("--target_csv", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_split", action="store_true",
                        help="Use full dataset as test, matching test.py/test_flow.py --no_split.")
    parser.add_argument("--use_split_test", action="store_true",
                        help="Force random_split test subset, matching test.py/test_flow.py --use_split_test.")
    parser.add_argument("--metadata_csv", type=str, default=None,
                        help="Optional CSV with columns id,smiles in processed dataset order.")
    parser.add_argument("--smiles_file", type=str, default=None,
                        help="Optional one-SMILES-per-line file in processed dataset order.")
    parser.add_argument("--mapping_txt", type=str, default=None,
                        help="Optional id-to-SMILES mapping file: '<id>\\t<smiles>'. IDs come from source CSV first column or H5 ids.")
    parser.add_argument("--smiles_only", action="store_true",
                        help="Write only one SMILES per line for the selected test samples, without a CSV header.")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.use_split_test and args.no_split:
        raise ValueError("--use_split_test and --no_split cannot be used together.")

    data_dir = Path(args.data_dir)
    source_csv = (
        Path(args.source_csv) if args.source_csv and os.path.isabs(args.source_csv)
        else data_dir / (args.source_csv if args.source_csv else f"{args.source_mode}_broaden_processed.csv")
    )
    target_csv = (
        Path(args.target_csv) if args.target_csv and os.path.isabs(args.target_csv)
        else data_dir / (args.target_csv if args.target_csv else f"{args.target_mode}_broaden_processed.csv")
    )

    source_h5 = Path(str(source_csv).replace(".csv", ".h5"))
    target_h5 = Path(str(target_csv).replace(".csv", ".h5"))
    if not ((source_csv.exists() or source_h5.exists()) and (target_csv.exists() or target_h5.exists())):
        raise FileNotFoundError(f"Data files not found: source={source_csv}, target={target_csv}")

    ids, smiles = build_metadata(args, source_csv)
    n = len(ids)

    if args.use_split_test:
        use_full_as_test = False
    elif paired_dataset_use_random_split_by_default(args.data_dir, args.source_csv, args.target_csv):
        use_full_as_test = args.no_split
    else:
        use_full_as_test = args.no_split or (args.source_csv is not None) or (args.target_csv is not None)

    test_indices = split_indices(n, args.seed, use_full_as_test)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.smiles_only:
        with output_path.open("w", encoding="utf-8", newline="\n") as f:
            for dataset_i in test_indices:
                f.write(f"{smiles[dataset_i]}\n")
    else:
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["test_index", "dataset_index", "id", "smiles"])
            for test_i, dataset_i in enumerate(test_indices):
                writer.writerow([test_i, dataset_i, ids[dataset_i], smiles[dataset_i]])

    missing = sum(1 for i in test_indices if not smiles[i])
    split_name = "FULL" if use_full_as_test else "random_split test subset"
    print(f"Dataset size: {n}")
    print(f"Exported: {len(test_indices)} samples ({split_name})")
    print(f"Missing SMILES: {missing}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
