"""Compare both possible NIST FTIR axis directions against matched QM9S IR."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.min()) / (np.ptp(values) + 1e-12)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(minmax(left), minmax(right))[0, 1])


def resample(
    spectrum: np.ndarray,
    target_axis: np.ndarray,
    source_axis: np.ndarray,
) -> np.ndarray:
    order = np.argsort(source_axis)
    return np.interp(
        target_axis,
        source_axis[order],
        np.asarray(spectrum)[order],
        left=0.0,
        right=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--ftir-dir", type=Path, required=True)
    parser.add_argument("--qm9s-ir-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qm9s-split", default="test")
    parser.add_argument("--low-wavenumber", type=float, default=399.0)
    parser.add_argument("--high-wavenumber", type=float, default=4000.0)
    args = parser.parse_args()

    matches = pd.read_csv(args.matches, dtype={"cid": str})
    matches = matches[matches["qm9s_split"] == args.qm9s_split].copy()
    rows = []
    with h5py.File(args.qm9s_ir_h5, "r") as handle:
        target_axis = handle["x_axis"][:]
        for record in matches.to_dict("records"):
            path = args.ftir_dir / f"{record['cid']}.npy"
            spectrum = np.load(path).ravel()
            ascending_axis = np.linspace(
                args.low_wavenumber,
                args.high_wavenumber,
                spectrum.size,
            )
            descending_axis = ascending_axis[::-1]
            computed_ir = handle["spectra"][int(record["qm9s_index"])]
            rows.append(
                {
                    "cid": record["cid"],
                    "qm9s_index": record["qm9s_index"],
                    "corr_assume_ascending": correlation(
                        resample(spectrum, target_axis, ascending_axis),
                        computed_ir,
                    ),
                    "corr_assume_descending": correlation(
                        resample(spectrum, target_axis, descending_axis),
                        computed_ir,
                    ),
                }
            )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print("Means:", result.mean(numeric_only=True).to_dict())
    print("Medians:", result.median(numeric_only=True).to_dict())
    print(
        "Descending wins:",
        int(
            (
                result["corr_assume_descending"]
                > result["corr_assume_ascending"]
            ).sum()
        ),
        "/",
        len(result),
    )


if __name__ == "__main__":
    main()
