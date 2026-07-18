"""Plot selected local RRUFF IR-to-Raman predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def restore_preserved_spectral_order(values: np.ndarray) -> np.ndarray:
    """Undo the legacy inverse-patch permutation applied during evaluation."""
    side = int(np.sqrt(values.shape[1]))
    if side * side != values.shape[1]:
        raise ValueError("Spectrum length must be a perfect square")
    patch_size = max(
        size for size in (10, 8, 5, 4, 2, 1) if side % size == 0
    )
    patches_per_row = side // patch_size
    restored = []
    for spectrum in values:
        patches = spectrum.reshape(-1, patch_size, patch_size)
        rows = [
            np.concatenate(
                patches[
                    row * patches_per_row : (row + 1) * patches_per_row
                ],
                axis=1,
            )
            for row in range(patches_per_row)
        ]
        restored.append(np.concatenate(rows, axis=0).reshape(-1))
    return np.stack(restored)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--raman-h5", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(
        args.result_dir / "flow_ir2raman_preds.csv"
    ).iloc[:, 1:].to_numpy()
    targets = pd.read_csv(
        args.result_dir / "flow_ir2raman_targets.csv"
    ).iloc[:, 1:].to_numpy()
    predictions = restore_preserved_spectral_order(predictions)
    targets = restore_preserved_spectral_order(targets)
    pairs = pd.read_csv(args.pairs)
    metadata = pd.read_csv(args.metadata).set_index("sample_name")
    with h5py.File(args.raman_h5, "r") as handle:
        axis = np.asarray(handle["x_axis"])

    if predictions.shape != targets.shape:
        raise ValueError("Prediction and target shapes do not match")
    if predictions.shape[1] != axis.size or len(pairs) != len(predictions):
        raise ValueError("Results, pair metadata, and Raman axis are not aligned")

    figure, axes = plt.subplots(
        len(args.indices),
        1,
        figsize=(12, 3.5 * len(args.indices)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    colors = ["#1677ff", "#e24a33", "#2ca02c"]

    for plot_index, (sample_index, current_axis) in enumerate(
        zip(args.indices, axes)
    ):
        pair = pairs.iloc[sample_index]
        material = metadata.loc[
            pair["raman_sample_name"], "spectrum_identity"
        ]
        pearson = np.corrcoef(
            predictions[sample_index], targets[sample_index]
        )[0, 1]
        current_axis.plot(
            axis,
            targets[sample_index],
            color="black",
            linewidth=1.25,
            label="Measured Raman (target)",
        )
        current_axis.plot(
            axis,
            predictions[sample_index],
            color=colors[plot_index % len(colors)],
            linewidth=1.1,
            alpha=0.9,
            label="Predicted Raman",
        )
        current_axis.set_title(
            f"{material.title()} | {pair['identity_key']} | "
            f"Pearson = {pearson:.3f}"
        )
        current_axis.set_ylabel("Intensity")
        current_axis.legend(frameon=False, loc="upper right")
        current_axis.grid(alpha=0.18)

    axes[-1].set_xlabel("Wavenumber (cm⁻¹)")
    figure.suptitle("Flow Model: Experimental IR → Experimental Raman", fontsize=14)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
