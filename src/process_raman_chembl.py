"""
Convert Raman-ChEMBL SQLite databases into fixed-length HDF5 files.

This script reads the `molecule` table from one or more SQLite `.db` files.
Each row stores lightweight metadata in table columns and vibrational data in a
zlib-compressed JSON blob. We keep only the fields needed for paired training:

- IR intensities
- Raman activities
- original sequence length
- atom count
- SMILES
- a unique record id

The variable-length mode lists are filtered by minimum length, then
interpolated onto a shared frequency grid with fixed target size
(default: 324 = 18x18) so they can be loaded by the existing HDF5-based
dataset pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import zlib
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def decode_blob(blob: bytes) -> dict:
    """Decode one zlib-compressed JSON blob from the SQLite table."""
    return json.loads(zlib.decompress(blob).decode("utf-8"))


def make_record_id(db_path: Path, row_id: int) -> str:
    """Create a globally unique id when concatenating multiple DB parts."""
    stem = db_path.stem.replace("Raman-ChEMBL-", "").replace(".db", "")
    return f"{stem}:{row_id}"


def interpolate_sequence(
    freq: list[float],
    values: list[float],
    target_grid: np.ndarray,
) -> np.ndarray:
    """
    Interpolate one intensity sequence onto a shared frequency grid.

    Points outside the sample's native frequency range are filled with zeros.
    """
    x = np.asarray(freq, dtype=np.float32)
    y = np.asarray(values, dtype=np.float32)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"Expected 1D inputs, got freq={x.shape}, values={y.shape}")
    if x.size != y.size:
        raise ValueError(f"freq/value length mismatch: {x.size} vs {y.size}")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Remove duplicate frequencies to keep np.interp stable.
    unique_x, unique_idx = np.unique(x, return_index=True)
    unique_y = y[unique_idx]
    if unique_x.size == 1:
        out = np.zeros_like(target_grid, dtype=np.float32)
        out[np.argmin(np.abs(target_grid - unique_x[0]))] = unique_y[0]
        return out

    interp = np.interp(
        target_grid,
        unique_x,
        unique_y,
        left=0.0,
        right=0.0,
    )
    return interp.astype(np.float32)


def collect_records(
    db_paths: list[Path],
    min_len: int,
    target_size: int,
    freq_min: float,
    freq_max: float,
):
    """
    Read all DB rows, keep only records with mode length >= min_len,
    and interpolate IR/Raman sequences to target_size.
    """
    ir_spectra = []
    raman_spectra = []
    lengths = []
    atoms = []
    smiles = []
    ids = []
    target_grid = np.linspace(freq_min, freq_max, target_size, dtype=np.float32)

    total_rows = 0
    kept_rows = 0

    for db_path in db_paths:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        n_rows = cur.execute("SELECT COUNT(*) FROM molecule").fetchone()[0]

        print(f"\nReading DB: {db_path}")
        print(f"Rows in molecule table: {n_rows}")

        kept_this_db = 0
        skipped_short = 0

        for row_id, smiles_text, blob in tqdm(
            cur.execute("SELECT id, SMILES, blob_data FROM molecule"),
            total=n_rows,
            desc=f"Decode {db_path.stem}",
            unit="row",
        ):
            total_rows += 1
            obj = decode_blob(blob)

            freq = obj["freq"]
            ir = obj["IR Inten"]
            raman = obj["Raman Activ"]
            atom_list = obj["atoms"]

            seq_len = len(freq)
            if not (seq_len == len(ir) == len(raman)):
                raise ValueError(
                    f"Length mismatch in {db_path} id={row_id}: "
                    f"freq={len(freq)}, ir={len(ir)}, raman={len(raman)}"
                )
            if seq_len < min_len:
                skipped_short += 1
                continue

            ids.append(make_record_id(db_path, int(row_id)))
            smiles.append(smiles_text or "")
            atoms.append(len(atom_list))
            lengths.append(seq_len)
            ir_spectra.append(interpolate_sequence(freq, ir, target_grid))
            raman_spectra.append(interpolate_sequence(freq, raman, target_grid))

            kept_rows += 1
            kept_this_db += 1

        conn.close()

        print(
            f"Kept {kept_this_db}/{n_rows} rows from {db_path.name} "
            f"(skipped short: {skipped_short})"
        )

    if kept_rows == 0:
        raise ValueError(
            f"No rows kept after filtering with min_len={min_len} and target_size={target_size}"
        )

    print(f"\nTotal rows scanned: {total_rows}")
    print(f"Total rows kept:    {kept_rows}")
    print(f"Length range kept:  {min(lengths)} - {max(lengths)}")
    print(f"Interpolation grid: [{freq_min}, {freq_max}] -> {target_size} points")

    return {
        "ids": ids,
        "smiles": smiles,
        "atoms": np.asarray(atoms, dtype=np.int32),
        "lengths": np.asarray(lengths, dtype=np.int32),
        "ir": np.stack(ir_spectra, axis=0).astype(np.float32),
        "raman": np.stack(raman_spectra, axis=0).astype(np.float32),
        "x_axis": target_grid,
    }


def save_modality_h5(
    output_path: Path,
    spectra: np.ndarray,
    x_axis: np.ndarray,
    ids: list[str],
    smiles: list[str],
    atoms: np.ndarray,
    lengths: np.ndarray,
):
    """Save one modality to HDF5 in a format compatible with current loaders."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("spectra", data=spectra, compression="gzip", compression_opts=4)
        f.create_dataset("x_axis", data=x_axis)
        f.create_dataset("ids", data=np.asarray(ids, dtype=object), dtype=str_dtype)
        f.create_dataset("smiles", data=np.asarray(smiles, dtype=object), dtype=str_dtype)
        f.create_dataset("atoms", data=atoms, compression="gzip", compression_opts=4)
        f.create_dataset("lengths", data=lengths, compression="gzip", compression_opts=4)
        f.attrs["n_samples"] = int(spectra.shape[0])
        f.attrs["spectrum_length"] = int(spectra.shape[1])
        f.attrs["preprocess"] = "interpolate_on_shared_frequency_grid"
        f.attrs["source"] = "Raman-ChEMBL SQLite"

    print(f"Saved HDF5: {output_path}")
    print(f"  spectra shape: {spectra.shape}")


def save_loader_compatible_csv(
    output_path: Path,
    ids: list[str],
    spectra: np.ndarray,
    x_axis: np.ndarray,
):
    """Save CSV in the format expected by the existing CSV fallback loader."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", *x_axis.tolist()])
        for sample_id, spec in zip(ids, spectra):
            writer.writerow([sample_id, *spec.tolist()])

    print(f"Saved CSV:  {output_path}")
    print(f"  spectra shape: {spectra.shape}")


def save_metadata_csv(
    output_path: Path,
    ids: list[str],
    smiles: list[str],
    atoms: np.ndarray,
    lengths: np.ndarray,
):
    """Save lightweight metadata in a human-readable sidecar CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "smiles", "atoms", "length"])
        for sample_id, smiles_text, atom_count, seq_len in zip(ids, smiles, atoms, lengths):
            writer.writerow([sample_id, smiles_text, int(atom_count), int(seq_len)])

    print(f"Saved metadata CSV: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Raman-ChEMBL SQLite DBs to fixed-length HDF5 files"
    )
    parser.add_argument(
        "--db_paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more Raman-ChEMBL SQLite .db files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save ir_broaden_processed.h5 and raman_broaden_processed.h5",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=100,
        help="Keep only rows whose mode length is at least this value",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=324,
        help="Interpolate sequences to this fixed length (default: 324 = 18x18)",
    )
    parser.add_argument(
        "--freq_min",
        type=float,
        default=0.0,
        help="Minimum frequency of the shared interpolation grid",
    )
    parser.add_argument(
        "--freq_max",
        type=float,
        default=4000.0,
        help="Maximum frequency of the shared interpolation grid",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Also save CSV files compatible with the existing CSV fallback loader",
    )
    args = parser.parse_args()

    if args.target_size <= 0:
        raise ValueError("--target_size must be positive")
    if args.min_len <= 0:
        raise ValueError("--min_len must be positive")
    if args.freq_min >= args.freq_max:
        raise ValueError("--freq_min must be smaller than --freq_max")

    db_paths = [Path(p) for p in args.db_paths]
    for db_path in db_paths:
        if not db_path.exists():
            raise FileNotFoundError(f"DB file not found: {db_path}")

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data = collect_records(
        db_paths,
        min_len=args.min_len,
        target_size=args.target_size,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
    )

    save_modality_h5(
        output_dir / "ir_broaden_processed.h5",
        data["ir"],
        data["x_axis"],
        data["ids"],
        data["smiles"],
        data["atoms"],
        data["lengths"],
    )
    save_modality_h5(
        output_dir / "raman_broaden_processed.h5",
        data["raman"],
        data["x_axis"],
        data["ids"],
        data["smiles"],
        data["atoms"],
        data["lengths"],
    )

    if args.save_csv:
        save_loader_compatible_csv(
            output_dir / "ir_broaden_processed.csv",
            data["ids"],
            data["ir"],
            data["x_axis"],
        )
        save_loader_compatible_csv(
            output_dir / "raman_broaden_processed.csv",
            data["ids"],
            data["raman"],
            data["x_axis"],
        )
        save_metadata_csv(
            output_dir / "metadata.csv",
            data["ids"],
            data["smiles"],
            data["atoms"],
            data["lengths"],
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
