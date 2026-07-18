"""Measure the attainable agreement between OpenSpecy Raman replicates."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def load_spectra(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return handle["spectra"][:].astype(np.float64)


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    residual = np.sum((target - prediction) ** 2)
    total = np.sum((target - target.mean()) ** 2)
    return float(1.0 - residual / max(total, 1e-12))


def pearson(target: np.ndarray, prediction: np.ndarray) -> float:
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    denominator = np.linalg.norm(target_centered) * np.linalg.norm(prediction_centered)
    return float(target_centered @ prediction_centered / max(denominator, 1e-12))


def summarize(name: str, targets: list[np.ndarray], predictions: list[np.ndarray]) -> dict:
    r2_values = np.array([r2(y, p) for y, p in zip(targets, predictions)])
    pearson_values = np.array(
        [pearson(y, p) for y, p in zip(targets, predictions)]
    )
    mse_values = np.array(
        [np.mean((y - p) ** 2) for y, p in zip(targets, predictions)]
    )
    return {
        "baseline": name,
        "mean_r2": r2_values.mean(),
        "median_r2": np.median(r2_values),
        "mean_pearson": pearson_values.mean(),
        "mean_mse": mse_values.mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    args = parser.parse_args()

    train_pairs = pd.read_csv(args.data_dir / "train_pairs.csv")
    test_pairs = pd.read_csv(args.data_dir / "test_pairs.csv")
    train_raman = load_spectra(args.data_dir / "train_raman.h5")
    test_raman = load_spectra(args.data_dir / "test_raman.h5")

    if len(train_pairs) != len(train_raman) or len(test_pairs) != len(test_raman):
        raise ValueError("Pair metadata and HDF5 row counts do not match")

    identity_to_train: dict[str, np.ndarray] = {}
    for identity, rows in train_pairs.groupby("identity_key").groups.items():
        # The pair builder may reuse a target with different IR measurements.
        identity_to_train[str(identity)] = np.unique(train_raman[list(rows)], axis=0)

    targets: list[np.ndarray] = []
    identity_mean: list[np.ndarray] = []
    nearest_mse_oracle: list[np.ndarray] = []
    nearest_pearson_oracle: list[np.ndarray] = []

    for row_index, pair in test_pairs.iterrows():
        identity = str(pair["identity_key"])
        references = identity_to_train.get(identity)
        if references is None or len(references) == 0:
            continue
        target = test_raman[row_index]
        mean_reference = references.mean(axis=0)
        mse_index = int(np.argmin(np.mean((references - target) ** 2, axis=1)))
        pearson_scores = np.array([pearson(target, ref) for ref in references])
        pearson_index = int(np.argmax(pearson_scores))

        targets.append(target)
        identity_mean.append(mean_reference)
        nearest_mse_oracle.append(references[mse_index])
        nearest_pearson_oracle.append(references[pearson_index])

    global_mean = train_raman.mean(axis=0)
    results = pd.DataFrame(
        [
            summarize(
                "global_train_mean",
                targets,
                [global_mean] * len(targets),
            ),
            summarize("same_identity_mean", targets, identity_mean),
            summarize("nearest_same_identity_mse_oracle", targets, nearest_mse_oracle),
            summarize(
                "nearest_same_identity_pearson_oracle",
                targets,
                nearest_pearson_oracle,
            ),
        ]
    )

    output_path = args.data_dir / "raman_repeatability_upper_bounds.csv"
    results.to_csv(output_path, index=False)
    print(f"Evaluated held-out Raman spectra: {len(targets)}")
    print(results.to_string(index=False))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
