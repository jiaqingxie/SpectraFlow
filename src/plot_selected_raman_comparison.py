"""Plot predicted and target Raman spectra for selected NIST/QM9S matches."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--cids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--x-min", type=float, default=500.0)
    parser.add_argument("--x-max", type=float, default=4000.0)
    parser.add_argument("--hide-r2", action="store_true")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary, dtype={"cid": str})
    predictions = pd.read_csv(args.predictions).iloc[:, 1:].to_numpy()
    targets = pd.read_csv(args.targets).iloc[:, 1:].to_numpy()
    if len(summary) != len(predictions) or predictions.shape != targets.shape:
        raise ValueError("Summary, prediction, and target rows are not aligned")

    axis = np.linspace(500.0, 4000.0, predictions.shape[1])
    figure, axes = plt.subplots(
        len(args.cids), 1, figsize=(12, 3.5 * len(args.cids)), sharex=True
    )
    axes = np.atleast_1d(axes)
    colors = ["#1677ff", "#e24a33"]

    for plot_index, (cid, current_axis) in enumerate(zip(args.cids, axes)):
        matches = summary.index[summary["cid"] == cid]
        if len(matches) != 1:
            raise ValueError(f"Expected one row for CID {cid}, found {len(matches)}")
        row_index = int(matches[0])
        metadata = summary.iloc[row_index]
        current_axis.plot(
            axis,
            targets[row_index],
            color="black",
            linewidth=1.25,
            label="Computed Raman (target)",
        )
        current_axis.plot(
            axis,
            predictions[row_index],
            color=colors[plot_index % len(colors)],
            linewidth=1.1,
            alpha=0.9,
            label="Predicted Raman",
        )
        metric_text = f"Pearson = {metadata['pearson']:.3f}"
        if not args.hide_r2:
            metric_text = f"R² = {metadata['r2']:.3f} | {metric_text}"
        current_axis.set_title(
            f"CID {cid} | {metadata['title']} | "
            f"{metric_text} | {metadata['match_type']}"
        )
        current_axis.set_xlim(args.x_min, args.x_max)
        current_axis.set_ylabel("Intensity")
        current_axis.legend(frameon=False, loc="upper right")
        current_axis.grid(alpha=0.18)

    axes[-1].set_xlabel("Wavenumber (cm⁻¹)")
    figure.suptitle(
        "VibraDiT: Experimental IR → Computed Raman",
        fontsize=14,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
