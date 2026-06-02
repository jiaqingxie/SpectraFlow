"""
Merge ViBench processed TEST H5 files into one paired IR/Raman dataset.

Default input patterns:
- *_test_ir_processed.h5
- *_test_raman_processed.h5

This script pairs files by dataset prefix before "_test_",
concatenates spectra in dataset order, and writes merged H5 outputs.
Optionally, it can also export CSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def list_mode_files(input_dir: Path, mode: str) -> dict[str, Path]:
    suffix = f"_test_{mode}_processed.h5"
    mapping: dict[str, Path] = {}
    for path in sorted(input_dir.glob(f"*{suffix}")):
        name = path.name
        if not name.endswith(suffix):
            continue
        prefix = name[: -len(suffix)]
        mapping[prefix] = path
    return mapping


def read_h5(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with h5py.File(path, "r") as f:
        spectra = f["spectra"][:]
        x_axis = f["x_axis"][:]
        physical = f["physical_params"][:] if "physical_params" in f else None
    return spectra, x_axis, physical


def write_h5(path: Path, spectra: np.ndarray, x_axis: np.ndarray, physical: np.ndarray | None, datasets: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("spectra", data=spectra, compression="gzip", compression_opts=4)
        f.create_dataset("x_axis", data=x_axis)
        if physical is not None:
            f.create_dataset("physical_params", data=physical, compression="gzip", compression_opts=4)
        f.attrs["n_samples"] = len(spectra)
        f.attrs["spectrum_length"] = spectra.shape[1]
        f.attrs["datasets"] = ",".join(datasets)


def save_loader_compatible_csv(path: Path, spectra: np.ndarray, x_axis: np.ndarray) -> None:
    """
    Save CSV compatible with PairedModalDataset CSV fallback in train.py:
    - row 0: [dummy, x_axis...]
    - row i: [i-1, spectrum...]
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = spectra.shape[0]
    first_col = np.arange(n + 1, dtype=np.float32).reshape(-1, 1)
    body = np.vstack([x_axis.reshape(1, -1), spectra]).astype(np.float32)
    matrix = np.hstack([first_col, body])
    np.savetxt(path, matrix, delimiter=",", fmt="%.6e")


def concat_optional(arrays: list[np.ndarray | None]) -> np.ndarray | None:
    valid = [a for a in arrays if a is not None]
    if not valid:
        return None
    if len(valid) != len(arrays):
        raise ValueError("Inconsistent physical_params presence across files.")
    return np.concatenate(valid, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ViBench *_test_{ir,raman}_processed.h5 into one big paired dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing processed H5 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save merged files")
    parser.add_argument("--output_prefix", type=str, default="vibench_test_full", help="Output file prefix")
    parser.add_argument("--save_csv", action="store_true", help="Also export merged CSV files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ir_files = list_mode_files(input_dir, "ir")
    raman_files = list_mode_files(input_dir, "raman")
    common = sorted(set(ir_files) & set(raman_files))

    if not common:
        raise ValueError(
            "No matched dataset prefixes found for '*_test_ir_processed.h5' and '*_test_raman_processed.h5'."
        )

    print(f"Matched datasets: {len(common)}")
    for name in common:
        print(f"  - {name}")

    ir_specs_all: list[np.ndarray] = []
    raman_specs_all: list[np.ndarray] = []
    ir_phys_all: list[np.ndarray | None] = []
    raman_phys_all: list[np.ndarray | None] = []
    ir_x_ref = None
    raman_x_ref = None

    for name in common:
        ir_path = ir_files[name]
        raman_path = raman_files[name]

        ir_spec, ir_x, ir_phys = read_h5(ir_path)
        raman_spec, raman_x, raman_phys = read_h5(raman_path)

        if len(ir_spec) != len(raman_spec):
            raise ValueError(
                f"Sample count mismatch in '{name}': ir={len(ir_spec)} vs raman={len(raman_spec)}"
            )

        if ir_x_ref is None:
            ir_x_ref = ir_x
        elif ir_x_ref.shape != ir_x.shape or not np.allclose(ir_x_ref, ir_x):
            raise ValueError(f"IR x_axis mismatch: {ir_path}")

        if raman_x_ref is None:
            raman_x_ref = raman_x
        elif raman_x_ref.shape != raman_x.shape or not np.allclose(raman_x_ref, raman_x):
            raise ValueError(f"Raman x_axis mismatch: {raman_path}")

        ir_specs_all.append(ir_spec)
        raman_specs_all.append(raman_spec)
        ir_phys_all.append(ir_phys)
        raman_phys_all.append(raman_phys)
        print(f"[ok] {name}: {len(ir_spec)} paired samples")

    ir_full = np.concatenate(ir_specs_all, axis=0)
    raman_full = np.concatenate(raman_specs_all, axis=0)
    ir_phys_full = concat_optional(ir_phys_all)
    raman_phys_full = concat_optional(raman_phys_all)

    ir_h5_out = output_dir / f"{args.output_prefix}_ir_processed.h5"
    raman_h5_out = output_dir / f"{args.output_prefix}_raman_processed.h5"
    write_h5(ir_h5_out, ir_full, ir_x_ref, ir_phys_full, common)
    write_h5(raman_h5_out, raman_full, raman_x_ref, raman_phys_full, common)

    print(f"[done] IR merged: {ir_h5_out} | shape={ir_full.shape}")
    print(f"[done] Raman merged: {raman_h5_out} | shape={raman_full.shape}")

    if args.save_csv:
        ir_csv_out = output_dir / f"{args.output_prefix}_ir_processed.csv"
        raman_csv_out = output_dir / f"{args.output_prefix}_raman_processed.csv"
        save_loader_compatible_csv(ir_csv_out, ir_full, ir_x_ref)
        save_loader_compatible_csv(raman_csv_out, raman_full, raman_x_ref)
        print(f"[done] IR csv: {ir_csv_out}")
        print(f"[done] Raman csv: {raman_csv_out}")


if __name__ == "__main__":
    main()

