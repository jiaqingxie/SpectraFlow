"""Plot a bidirectional summary of peak-resolved QM9S translation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def finite(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column].to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values)
    return values, np.arange(1, len(values) + 1) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir2raman", type=Path, required=True)
    parser.add_argument("--raman2ir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames = {
        r"IR$\rightarrow$Raman": pd.read_csv(args.ir2raman),
        r"Raman$\rightarrow$IR": pd.read_csv(args.raman2ir),
    }
    colors = {
        r"IR$\rightarrow$Raman": "#345E85",
        r"Raman$\rightarrow$IR": "#A45A3F",
    }
    tolerances = np.array([5.0, 10.0, 20.0, 30.0])

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))

    for label, frame in frames.items():
        recall = [
            frame[f"recall_at_{tolerance:g}_cm-1"].mean()
            for tolerance in tolerances
        ]
        axes[0, 0].plot(
            tolerances,
            recall,
            marker="o",
            lw=1.5,
            ms=4,
            color=colors[label],
            label=label,
        )
    axes[0, 0].set(
        xlabel=r"Peak-matching tolerance (cm$^{-1}$)",
        ylabel="Mean target-peak recall",
        ylim=(0.5, 0.9),
        title="a  Peak recovery",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)

    grid = np.linspace(-0.2, 1.0, 300)
    for label, frame in frames.items():
        values = finite(frame, "intensity_pearson")
        density = gaussian_kde(values)(grid)
        axes[0, 1].plot(grid, density, color=colors[label], lw=1.5, label=label)
        axes[0, 1].axvline(
            np.median(values), color=colors[label], ls="--", lw=0.9
        )
    axes[0, 1].set(
        xlabel="Per-spectrum matched-peak intensity Pearson",
        ylabel="Density",
        xlim=(-0.2, 1.0),
        title="b  Peak-intensity agreement",
    )

    for label, frame in frames.items():
        x, y = ecdf(finite(frame, "position_mae_cm-1"))
        axes[1, 0].plot(x, y, color=colors[label], lw=1.5, label=label)
    axes[1, 0].set(
        xlabel=r"Per-spectrum peak-position MAE (cm$^{-1}$)",
        ylabel="Cumulative fraction",
        xlim=(0, 10),
        ylim=(0, 1.02),
        title="c  Peak-position error",
    )

    metrics = [
        ("peak_precision", "Peak\nprecision"),
        ("peak_recall", "Peak recall\n(20 cm$^{-1}$)"),
        ("top5_recall", "Top-5\nrecall"),
    ]
    x_positions = np.arange(len(metrics))
    width = 0.34
    for offset, (label, frame) in zip((-width / 2, width / 2), frames.items()):
        values = [frame[column].mean() for column, _ in metrics]
        axes[1, 1].bar(
            x_positions + offset,
            values,
            width=width,
            color=colors[label],
            label=label,
        )
    axes[1, 1].set(
        ylabel="Mean score",
        ylim=(0, 1.0),
        title="d  Peak-level summary",
    )
    axes[1, 1].set_xticks(x_positions)
    axes[1, 1].set_xticklabels([display for _, display in metrics])

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout(h_pad=1.7, w_pad=1.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
