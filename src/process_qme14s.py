"""
Convert QMe14S per-file broaden spectra into the aggregated processed format
used by the existing VAE/Flow/Transformer loaders.

Expected raw layout from QMe14S:
- one CSV per molecule in each modality directory
- filename contains a shared numeric id, e.g. IR_000001.csv / Raman_000001.csv
- file body contains optional metadata/atom lines, then one final spectrum line

Output layout:
- ir_broaden_processed.csv / raman_broaden_processed.csv
- ir_broaden_processed.h5 / raman_broaden_processed.h5
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from process import process_spectrum_to_3600, save_processed_data_h5
from utils import compute_params


def extract_sample_id(path: Path) -> str:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot extract numeric sample id from filename: {path.name}")
    return match.group(1)


def parse_qme14s_spectrum(csv_path: Path) -> np.ndarray:
    """
    Parse one QMe14S spectrum file.

    The README says the broadened spectrum is on the last content line.
    Some files may have a SMILES/header line before the atom block, some may not.
    We only need the final spectrum line for the current model pipeline.
    """
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty file: {csv_path}")

    spectrum_line = lines[-1]
    try:
        spectrum = np.array([float(x) for x in spectrum_line.split(",")], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"Failed to parse spectrum line in {csv_path}") from exc

    if spectrum.ndim != 1 or spectrum.size < 32:
        raise ValueError(f"Suspicious spectrum length {spectrum.size} in {csv_path}")

    return spectrum


def build_index(dir_path: Path) -> dict[str, Path]:
    files = sorted(dir_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {dir_path}")
    return {extract_sample_id(path): path for path in files}


def compute_physical_params(processed_spectra: np.ndarray, desc: str) -> np.ndarray:
    physical_list = []
    for spec in tqdm(processed_spectra, desc=desc, unit="spec"):
        params = compute_params(spec)
        physical_vec = np.array([
            params.get("mean", 0),
            params.get("std", 0),
            params.get("bandwidth", 0),
            len(params.get("peak_positions", [])),
            params.get("max_intensity", 0),
            params.get("energy_range", (0, 0))[0],
            params.get("energy_range", (0, 0))[1],
        ], dtype=np.float32)
        physical_vec = np.nan_to_num(physical_vec, nan=0.0, posinf=1e6, neginf=-1e6)
        physical_vec = np.clip(physical_vec, -1e3, 1e3)
        physical_list.append(physical_vec)
    return np.stack(physical_list, axis=0)


def save_loader_compatible_csv(sample_ids: list[str], spectra: np.ndarray, x_axis: np.ndarray, output_path: Path) -> None:
    """
    Save CSV in the shape expected by PairedModalDataset CSV fallback:
    row 0: [dummy, x_axis...]
    row i: [sample_id, spectrum...]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [["id", *x_axis.tolist()]]
    for sample_id, spec in zip(sample_ids, spectra):
        data.append([sample_id, *spec.tolist()])

    pd.DataFrame(data).to_csv(output_path, index=False, header=False)


def process_modality(sample_ids: list[str], path_map: dict[str, Path], target_size: int, desc: str) -> tuple[np.ndarray, np.ndarray]:
    raw_spectra = []
    raw_lengths = set()

    for sample_id in tqdm(sample_ids, desc=f"Read {desc}", unit="file"):
        spec = parse_qme14s_spectrum(path_map[sample_id])
        raw_spectra.append(spec)
        raw_lengths.add(spec.size)

    if len(raw_lengths) != 1:
        raise ValueError(f"Found inconsistent raw spectrum lengths: {sorted(raw_lengths)}")

    processed = np.stack([
        process_spectrum_to_3600(spec, target_size=target_size)
        for spec in tqdm(raw_spectra, desc=f"Interpolate {desc}", unit="spec")
    ], axis=0).astype(np.float32)
    return np.stack(raw_spectra, axis=0), processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert QMe14S per-file spectra to SpectroGen processed format")
    parser.add_argument("--ir_dir", type=str, required=True, help="Directory containing IR_*.csv files")
    parser.add_argument("--raman_dir", type=str, required=True, help="Directory containing Raman_*.csv files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed outputs")
    parser.add_argument("--target_size", type=int, default=3600, help="Interpolated spectrum length")
    parser.add_argument("--wavenumber_min", type=float, default=500.0, help="Minimum wave number from dataset README")
    parser.add_argument("--wavenumber_max", type=float, default=4000.0, help="Maximum wave number from dataset README")
    parser.add_argument("--csv_only", action="store_true", help="Only save aggregated CSV files")
    parser.add_argument("--h5_only", action="store_true", help="Only save HDF5 files")
    parser.add_argument("--no_physical_params", action="store_true",
                        help="Skip computing physical_params to speed up preprocessing")
    args = parser.parse_args()

    if args.csv_only and args.h5_only:
        raise ValueError("--csv_only and --h5_only cannot be used together")

    ir_dir = Path(args.ir_dir)
    raman_dir = Path(args.raman_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ir_map = build_index(ir_dir)
    raman_map = build_index(raman_dir)

    common_ids = sorted(set(ir_map) & set(raman_map), key=lambda x: int(x))
    if not common_ids:
        raise ValueError("No overlapping sample ids found between IR and Raman directories")

    print(f"IR files: {len(ir_map)}")
    print(f"Raman files: {len(raman_map)}")
    print(f"Paired samples: {len(common_ids)}")

    _, ir_processed = process_modality(common_ids, ir_map, target_size=args.target_size, desc="IR")
    _, raman_processed = process_modality(common_ids, raman_map, target_size=args.target_size, desc="Raman")

    processed_x_axis = np.linspace(args.wavenumber_min, args.wavenumber_max, args.target_size, dtype=np.float32)
    if args.no_physical_params:
        ir_physical = None
        raman_physical = None
        print("Skipping physical_params computation.")
    else:
        ir_physical = compute_physical_params(ir_processed, desc="Compute IR physical params")
        raman_physical = compute_physical_params(raman_processed, desc="Compute Raman physical params")

    ir_csv = output_dir / "ir_broaden_processed.csv"
    raman_csv = output_dir / "raman_broaden_processed.csv"
    ir_h5 = output_dir / "ir_broaden_processed.h5"
    raman_h5 = output_dir / "raman_broaden_processed.h5"

    if not args.h5_only:
        save_loader_compatible_csv(common_ids, ir_processed, processed_x_axis, ir_csv)
        save_loader_compatible_csv(common_ids, raman_processed, processed_x_axis, raman_csv)
        print(f"Saved CSV: {ir_csv}")
        print(f"Saved CSV: {raman_csv}")

    if not args.csv_only:
        save_processed_data_h5(ir_processed, processed_x_axis, str(ir_h5), physical_params=ir_physical)
        save_processed_data_h5(raman_processed, processed_x_axis, str(raman_h5), physical_params=raman_physical)
        print(f"Saved H5:  {ir_h5}")
        print(f"Saved H5:  {raman_h5}")


if __name__ == "__main__":
    main()
