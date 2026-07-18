"""Peak-resolved evaluation for saved predicted and target spectra.

This analysis is a mode-level proxy for broadened spectra. It detects reference
peaks, performs one-to-one peak matching, and reports peak recall, position
error, and intensity agreement. It must not be described as a direct Raman
activity evaluation unless raw normal-mode activities are supplied separately.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr


def normalize_spectrum(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - np.nanmin(values)
    maximum = np.nanmax(values)
    if not np.isfinite(maximum) or maximum <= 1e-12:
        return np.zeros_like(values)
    return np.clip(values / maximum, 0.0, None)


def match_peaks(
    target_indices: np.ndarray,
    pred_indices: np.ndarray,
    axis: np.ndarray,
    tolerance: float,
) -> list[tuple[int, int]]:
    if len(target_indices) == 0 or len(pred_indices) == 0:
        return []
    distances = np.abs(
        axis[target_indices, None] - axis[pred_indices[None, :]]
    )
    target_rows, pred_cols = linear_sum_assignment(distances)
    return [
        (int(target_indices[row]), int(pred_indices[col]))
        for row, col in zip(target_rows, pred_cols)
        if distances[row, col] <= tolerance
    ]


def safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    result = pearsonr(x, y) if method == "pearson" else spearmanr(x, y)
    if hasattr(result, "statistic"):
        return float(result.statistic)
    if hasattr(result, "correlation"):
        return float(result.correlation)
    return float(result[0])


def bootstrap_ci(
    values: np.ndarray, statistic, rng: np.random.Generator, draws: int
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sample = rng.choice(values, size=len(values), replace=True)
        estimates[draw] = statistic(sample)
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def evaluate_pair(
    target: np.ndarray,
    prediction: np.ndarray,
    axis: np.ndarray,
    prominence: float,
    min_distance_points: int,
    match_tolerance: float,
    top_k: int,
    recall_tolerances: list[float],
) -> tuple[dict[str, float], list[tuple[float, float]]]:
    target = normalize_spectrum(target)
    prediction = normalize_spectrum(prediction)
    target_peaks, _ = find_peaks(
        target, prominence=prominence, distance=min_distance_points
    )
    pred_peaks, _ = find_peaks(
        prediction, prominence=prominence, distance=min_distance_points
    )

    matches = match_peaks(
        target_peaks, pred_peaks, axis=axis, tolerance=match_tolerance
    )
    target_intensities = np.asarray([target[i] for i, _ in matches])
    pred_intensities = np.asarray([prediction[j] for _, j in matches])
    position_errors = np.asarray(
        [abs(axis[i] - axis[j]) for i, j in matches], dtype=np.float64
    )

    target_top = target_peaks[
        np.argsort(target[target_peaks])[-min(top_k, len(target_peaks)) :]
    ]
    pred_top = pred_peaks[
        np.argsort(prediction[pred_peaks])[-min(top_k, len(pred_peaks)) :]
    ]
    top_matches = match_peaks(
        target_top, pred_top, axis=axis, tolerance=match_tolerance
    )

    row = {
        "n_target_peaks": float(len(target_peaks)),
        "n_pred_peaks": float(len(pred_peaks)),
        "n_matched_peaks": float(len(matches)),
        "peak_recall": len(matches) / len(target_peaks) if len(target_peaks) else np.nan,
        "peak_precision": len(matches) / len(pred_peaks) if len(pred_peaks) else np.nan,
        "position_mae_cm-1": (
            float(np.mean(position_errors)) if len(position_errors) else np.nan
        ),
        "intensity_mae": (
            float(np.mean(np.abs(target_intensities - pred_intensities)))
            if len(matches)
            else np.nan
        ),
        "intensity_pearson": safe_correlation(
            target_intensities, pred_intensities, "pearson"
        ),
        "intensity_spearman": safe_correlation(
            target_intensities, pred_intensities, "spearman"
        ),
        f"top{top_k}_recall": (
            len(top_matches) / len(target_top) if len(target_top) else np.nan
        ),
    }
    for tolerance in recall_tolerances:
        tolerance_matches = match_peaks(
            target_peaks, pred_peaks, axis=axis, tolerance=tolerance
        )
        row[f"recall_at_{tolerance:g}_cm-1"] = (
            len(tolerance_matches) / len(target_peaks)
            if len(target_peaks)
            else np.nan
        )
    scatter = list(zip(target_intensities.tolist(), pred_intensities.tolist()))
    return row, scatter


def plot_results(
    metrics: pd.DataFrame,
    scatter_points: np.ndarray,
    recall_tolerances: list[float],
    output_prefix: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    color = "#345E85"
    accent = "#9E5A3A"
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))

    recall_means = [
        metrics[f"recall_at_{tolerance:g}_cm-1"].mean()
        for tolerance in recall_tolerances
    ]
    axes[0].plot(recall_tolerances, recall_means, marker="o", color=color, lw=1.5)
    axes[0].set(
        xlabel=r"Peak-matching tolerance (cm$^{-1}$)",
        ylabel="Mean target-peak recall",
        ylim=(0, 1.02),
        title="a  Peak recovery",
    )

    pearson_values = metrics["intensity_pearson"].dropna().to_numpy()
    axes[1].hist(
        pearson_values,
        bins=np.linspace(-1, 1, 31),
        color=color,
        edgecolor="white",
        linewidth=0.35,
    )
    median = float(np.median(pearson_values))
    axes[1].axvline(median, color=accent, ls="--", lw=1.2)
    axes[1].set(
        xlabel="Per-spectrum peak-intensity Pearson",
        ylabel="Number of spectra",
        title=f"b  Intensity agreement\nMedian = {median:.3f}",
    )

    if len(scatter_points) > 100_000:
        rng = np.random.default_rng(42)
        scatter_points = scatter_points[
            rng.choice(len(scatter_points), 100_000, replace=False)
        ]
    axes[2].hexbin(
        scatter_points[:, 0],
        scatter_points[:, 1],
        gridsize=45,
        mincnt=1,
        bins="log",
        cmap="Blues",
    )
    axes[2].plot([0, 1], [0, 1], color=accent, ls="--", lw=1.0)
    axes[2].set(
        xlabel="Reference normalized peak intensity",
        ylabel="Predicted normalized peak intensity",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
        title="c  Matched peak intensities",
    )

    for axis_obj in axes:
        axis_obj.spines["top"].set_visible(False)
        axis_obj.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=128)
    parser.add_argument("--wavenumber_min", type=float, default=400.0)
    parser.add_argument("--wavenumber_max", type=float, default=4000.0)
    parser.add_argument("--prominence", type=float, default=0.05)
    parser.add_argument("--min_peak_distance_cm", type=float, default=10.0)
    parser.add_argument("--match_tolerance_cm", type=float, default=20.0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--recall_tolerances",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0, 30.0],
    )
    parser.add_argument("--bootstrap_draws", type=int, default=2000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_reader = pd.read_csv(args.preds, chunksize=args.chunksize)
    target_reader = pd.read_csv(args.targets, chunksize=args.chunksize)
    rows: list[dict[str, float]] = []
    scatter_points: list[tuple[float, float]] = []
    spectrum_length = None
    axis = None

    for pred_chunk, target_chunk in itertools.zip_longest(pred_reader, target_reader):
        if pred_chunk is None or target_chunk is None:
            raise ValueError("Prediction and target files contain different row counts")
        pred_indices = pred_chunk.iloc[:, 0].to_numpy()
        target_indices = target_chunk.iloc[:, 0].to_numpy()
        if not np.array_equal(pred_indices, target_indices):
            raise ValueError("Prediction and target sample indices are not aligned")
        predictions = pred_chunk.iloc[:, 1:].to_numpy(dtype=np.float64)
        targets = target_chunk.iloc[:, 1:].to_numpy(dtype=np.float64)
        if spectrum_length is None:
            spectrum_length = predictions.shape[1]
            axis = np.linspace(
                args.wavenumber_min, args.wavenumber_max, spectrum_length
            )
        spacing = abs(float(axis[1] - axis[0]))
        min_distance_points = max(
            1, int(round(args.min_peak_distance_cm / spacing))
        )

        for sample_index, prediction, target in zip(
            pred_indices, predictions, targets
        ):
            row, scatter = evaluate_pair(
                target=target,
                prediction=prediction,
                axis=axis,
                prominence=args.prominence,
                min_distance_points=min_distance_points,
                match_tolerance=args.match_tolerance_cm,
                top_k=args.top_k,
                recall_tolerances=args.recall_tolerances,
            )
            row["index"] = int(sample_index)
            rows.append(row)
            scatter_points.extend(scatter)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "peak_resolved_metrics_per_sample.csv", index=False)

    rng = np.random.default_rng(42)
    summary_rows = []
    for column in metrics.columns:
        if column == "index":
            continue
        values = metrics[column].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        mean_ci = bootstrap_ci(finite, np.mean, rng, args.bootstrap_draws)
        median_ci = bootstrap_ci(finite, np.median, rng, args.bootstrap_draws)
        summary_rows.append(
            {
                "metric": column,
                "n": len(finite),
                "mean": float(np.mean(finite)),
                "mean_ci_low": mean_ci[0],
                "mean_ci_high": mean_ci[1],
                "median": float(np.median(finite)),
                "median_ci_low": median_ci[0],
                "median_ci_high": median_ci[1],
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "peak_resolved_summary.csv", index=False
    )
    plot_results(
        metrics,
        np.asarray(scatter_points, dtype=np.float64),
        args.recall_tolerances,
        args.output_dir / "qm9s_peak_resolved_evaluation",
    )
    print(f"Evaluated {len(metrics)} spectra")
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
