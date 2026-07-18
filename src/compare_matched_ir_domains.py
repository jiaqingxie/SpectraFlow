"""Compare identity-matched experimental and computed IR spectra."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import h5py
import numpy as np
import pandas as pd


def minmax_rows(values: np.ndarray) -> np.ndarray:
    minimum = values.min(axis=1, keepdims=True)
    maximum = values.max(axis=1, keepdims=True)
    return (values - minimum) / np.maximum(maximum - minimum, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--experimental-ir-h5", type=Path, required=True)
    parser.add_argument("--computed-ir-h5", type=Path, required=True)
    parser.add_argument("--computed-ir-zip", type=Path)
    parser.add_argument("--zip-prefix", default="IR_broaden/IR_")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matches = pd.read_csv(args.matches)
    with h5py.File(args.experimental_ir_h5, "r") as handle:
        experimental = np.asarray(handle["spectra"], dtype=np.float64)
        axis = np.asarray(handle["x_axis"], dtype=np.float64)
    if args.computed_ir_zip:
        computed_rows = []
        target_grid = np.linspace(0.0, 1.0, experimental.shape[1])
        with zipfile.ZipFile(args.computed_ir_zip) as archive:
            for index in matches["qm9s_index"].to_numpy(dtype=int):
                member = f"{args.zip_prefix}{index:06d}.csv"
                lines = archive.read(member).decode("utf-8").strip().splitlines()
                spectrum = np.fromstring(lines[-1], sep=",")
                computed_rows.append(
                    np.interp(
                        target_grid,
                        np.linspace(0.0, 1.0, spectrum.size),
                        spectrum,
                    )
                )
        computed = np.stack(computed_rows)
    else:
        with h5py.File(args.computed_ir_h5, "r") as handle:
            computed = np.asarray(
                handle["spectra"][matches["qm9s_index"].to_numpy(dtype=int)],
                dtype=np.float64,
            )

    if experimental.shape != computed.shape:
        raise ValueError(
            f"Shape mismatch: experimental={experimental.shape}, "
            f"computed={computed.shape}"
        )

    experimental_norm = minmax_rows(experimental)
    computed_norm = minmax_rows(computed)
    exp_centered = experimental_norm - experimental_norm.mean(
        axis=1, keepdims=True
    )
    comp_centered = computed_norm - computed_norm.mean(
        axis=1, keepdims=True
    )
    covariance = (exp_centered * comp_centered).sum(axis=1)
    denominator = np.sqrt(
        np.square(exp_centered).sum(axis=1)
        * np.square(comp_centered).sum(axis=1)
    )
    pearson = np.divide(
        covariance,
        denominator,
        out=np.full(len(matches), np.nan),
        where=denominator > 0,
    )
    residual_ss = np.square(experimental_norm - computed_norm).sum(axis=1)
    total_ss = np.square(
        computed_norm - computed_norm.mean(axis=1, keepdims=True)
    ).sum(axis=1)
    r2 = 1.0 - residual_ss / total_ss
    mae = np.abs(experimental_norm - computed_norm).mean(axis=1)
    cosine = (experimental_norm * computed_norm).sum(axis=1) / np.sqrt(
        np.square(experimental_norm).sum(axis=1)
        * np.square(computed_norm).sum(axis=1)
    )
    peak_exp = axis[np.argmax(experimental_norm, axis=1)]
    peak_comp = axis[np.argmax(computed_norm, axis=1)]

    result = matches.copy()
    result["ir_pearson"] = pearson
    result["ir_r2_minmax"] = r2
    result["ir_mae_minmax"] = mae
    result["ir_cosine"] = cosine
    result["strongest_peak_delta_cm1"] = np.abs(peak_exp - peak_comp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    columns = [
        "ir_pearson",
        "ir_r2_minmax",
        "ir_mae_minmax",
        "ir_cosine",
        "strongest_peak_delta_cm1",
    ]
    print(f"Compared rows: {len(result)}")
    print(result[columns].agg(["mean", "median", "std"]).to_string())
    if "qm9s_split" in result:
        print("\nBy QM9S split:")
        print(result.groupby("qm9s_split")[columns].mean().to_string())


if __name__ == "__main__":
    main()
