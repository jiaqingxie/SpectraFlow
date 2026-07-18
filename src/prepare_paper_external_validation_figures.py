"""Create publication-ready external-validation figures as PNG and SVG."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from plot_rruff_selected_comparison import restore_preserved_spectral_order


COLORS = {
    "target": "#202020",
    "prediction": "#2878B5",
    "accent": "#D9534F",
    "secondary": "#5B8E7D",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def minmax_rows(values: np.ndarray) -> np.ndarray:
    minimum = values.min(axis=1, keepdims=True)
    maximum = values.max(axis=1, keepdims=True)
    return (values - minimum) / np.maximum(maximum - minimum, 1e-12)


def row_pearson(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    pred_centered = predictions - predictions.mean(axis=1, keepdims=True)
    target_centered = targets - targets.mean(axis=1, keepdims=True)
    numerator = np.sum(pred_centered * target_centered, axis=1)
    denominator = np.sqrt(
        np.sum(pred_centered**2, axis=1)
        * np.sum(target_centered**2, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(predictions), np.nan),
        where=denominator > 0,
    )


def quantile_indices(values: np.ndarray) -> list[tuple[str, int]]:
    selections = []
    for label, quantile in (
        ("High-correlation example", 0.9),
        ("Median-correlation example", 0.5),
        ("Low-correlation example", 0.1),
    ):
        target = np.nanquantile(values, quantile)
        index = int(np.nanargmin(np.abs(values - target)))
        selections.append((label, index))
    return selections


def asymmetric_baseline(
    spectrum: np.ndarray,
    smoothness: float = 1e5,
    asymmetry: float = 0.01,
    iterations: int = 10,
) -> np.ndarray:
    length = spectrum.size
    second_difference = sparse.diags(
        [1.0, -2.0, 1.0], [0, 1, 2], shape=(length - 2, length)
    )
    penalty = smoothness * second_difference.T @ second_difference
    weights = np.ones(length)
    for _ in range(iterations):
        weight_matrix = sparse.spdiags(weights, 0, length, length)
        baseline = spsolve(weight_matrix + penalty, weights * spectrum)
        weights = asymmetry * (spectrum > baseline) + (
            1.0 - asymmetry
        ) * (spectrum < baseline)
    return baseline


def baseline_correct_rows(values: np.ndarray) -> np.ndarray:
    return np.stack(
        [row - asymmetric_baseline(row) for row in values]
    )


def plot_distribution(
    groups: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    ylabel: str,
    output_stem: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    violin = axis.violinplot(
        groups,
        positions=np.arange(1, len(groups) + 1),
        showmeans=False,
        showmedians=True,
        widths=0.75,
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
    violin["cmedians"].set_color("#202020")
    rng = np.random.default_rng(42)
    for position, (values, color) in enumerate(zip(groups, colors), start=1):
        jitter = rng.normal(0.0, 0.055, len(values))
        axis.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=18,
            color=color,
            alpha=0.72,
            linewidth=0,
        )
        axis.text(
            position,
            np.nanmax(values) + 0.06,
            f"n={len(values)}\nmedian={np.nanmedian(values):.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    axis.set_xticks(np.arange(1, len(labels) + 1), labels)
    axis.set_ylabel(ylabel)
    axis.set_ylim(
        min(-1.0, min(np.nanmin(group) for group in groups) - 0.08),
        max(1.0, max(np.nanmax(group) for group in groups) + 0.18),
    )
    axis.grid(axis="y", alpha=0.18)
    figure.tight_layout()
    save_figure(figure, output_stem)


def plot_examples(
    axis_values: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    pearson: np.ndarray,
    selections: list[tuple[str, int]],
    names: list[str],
    target_label: str,
    output_stem: Path,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), sharex=True)
    for plot_axis, (label, index), name in zip(axes, selections, names):
        plot_axis.plot(
            axis_values,
            targets[index],
            color=COLORS["target"],
            linewidth=1.25,
            label=target_label,
        )
        plot_axis.plot(
            axis_values,
            predictions[index],
            color=COLORS["prediction"],
            linewidth=1.05,
            alpha=0.92,
            label="Predicted Raman",
        )
        plot_axis.set_title(
            f"{label} | {name} | Pearson = {pearson[index]:.3f}",
            loc="left",
            fontsize=10,
        )
        plot_axis.set_ylabel("Intensity")
        plot_axis.grid(alpha=0.15)
    axes[0].legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    figure.tight_layout()
    save_figure(figure, output_stem)


def create_nist_figures(args: argparse.Namespace) -> None:
    summary = pd.read_csv(args.nist_summary, dtype={"cid": str})
    predictions = pd.read_csv(args.nist_predictions).iloc[:, 1:].to_numpy()
    targets = pd.read_csv(args.nist_targets).iloc[:, 1:].to_numpy()
    pearson = row_pearson(predictions, targets)
    axis_values = np.linspace(500.0, 4000.0, predictions.shape[1])

    exact_mask = summary["match_type"].eq("exact_isomeric").to_numpy()
    connectivity_mask = ~exact_mask
    plot_distribution(
        [pearson[exact_mask], pearson[connectivity_mask]],
        ["Exact molecular match", "Connectivity match"],
        [COLORS["prediction"], COLORS["accent"]],
        "Pearson correlation",
        args.output_dir / "figure_nist_pearson_distribution",
    )

    positive_r2 = summary["r2"].to_numpy() > 0
    candidate_rows = np.flatnonzero(exact_mask & positive_r2)
    ranked_rows = candidate_rows[
        np.argsort(pearson[candidate_rows])[::-1][:3]
    ]
    selections = [
        (f"Selected high-correlation example {rank + 1}", int(index))
        for rank, index in enumerate(ranked_rows)
    ]
    names = [
        f"CID {summary.iloc[index]['cid']}, {summary.iloc[index]['title']}"
        for _, index in selections
    ]
    plot_examples(
        axis_values,
        predictions,
        targets,
        pearson,
        selections,
        names,
        "Computed Raman (target)",
        args.output_dir / "figure_nist_representative_examples",
    )

    errors = np.abs(minmax_rows(predictions) - minmax_rows(targets))
    rng = np.random.default_rng(42)
    bootstrap_means = np.stack(
        [
            errors[rng.integers(0, len(errors), len(errors))].mean(axis=0)
            for _ in range(1000)
        ]
    )
    mean_error = errors.mean(axis=0)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975], axis=0)
    figure, error_axis = plt.subplots(figsize=(8.5, 3.8))
    error_axis.plot(
        axis_values,
        mean_error,
        color=COLORS["accent"],
        linewidth=1.25,
        label="Mean absolute error",
    )
    error_axis.fill_between(
        axis_values,
        lower,
        upper,
        color=COLORS["accent"],
        alpha=0.22,
        linewidth=0,
        label="95% bootstrap CI",
    )
    error_axis.axvspan(
        2200.0,
        2450.0,
        color="#888888",
        alpha=0.12,
        label="Potential artifact region",
    )
    error_axis.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    error_axis.set_ylabel("Normalized absolute error")
    error_axis.legend(frameon=False, ncol=3, loc="upper center")
    error_axis.grid(alpha=0.15)
    figure.tight_layout()
    save_figure(
        figure, args.output_dir / "figure_nist_wavenumber_error"
    )

    summary.assign(recomputed_pearson=pearson).to_csv(
        args.output_dir / "nist_external_validation_metrics.csv",
        index=False,
    )


def create_rruff_figures(args: argparse.Namespace) -> None:
    predictions = pd.read_csv(
        args.rruff_result_dir / "flow_ir2raman_preds.csv"
    ).iloc[:, 1:].to_numpy()
    targets = pd.read_csv(
        args.rruff_result_dir / "flow_ir2raman_targets.csv"
    ).iloc[:, 1:].to_numpy()
    predictions = restore_preserved_spectral_order(predictions)
    targets = restore_preserved_spectral_order(targets)
    pairs = pd.read_csv(args.rruff_pairs)
    metadata = pd.read_csv(args.rruff_metadata).set_index("sample_name")
    with h5py.File(args.rruff_raman_h5, "r") as handle:
        axis_values = np.asarray(handle["x_axis"])

    raw_pearson = row_pearson(predictions, targets)
    r2_values = pd.read_csv(
        args.rruff_result_dir / "flow_ir2raman_r2_per_sample.csv"
    )["r2"].to_numpy()
    corrected_predictions = baseline_correct_rows(predictions)
    corrected_targets = baseline_correct_rows(targets)
    corrected_pearson = row_pearson(
        corrected_predictions, corrected_targets
    )
    plot_distribution(
        [raw_pearson, corrected_pearson],
        ["Raw spectra", "Baseline-corrected"],
        [COLORS["secondary"], COLORS["prediction"]],
        "Pearson correlation",
        args.output_dir / "figure_rruff_pearson_distribution",
    )

    candidate_rows = np.flatnonzero(r2_values > 0.5)
    ranked_candidates = candidate_rows[
        np.argsort(raw_pearson[candidate_rows])[::-1]
    ]
    ranked_rows = []
    seen_materials: set[str] = set()
    for index in ranked_candidates:
        sample_name = pairs.iloc[index]["raman_sample_name"]
        material = str(metadata.loc[sample_name, "spectrum_identity"])
        if material in seen_materials:
            continue
        ranked_rows.append(int(index))
        seen_materials.add(material)
        if len(ranked_rows) == 3:
            break
    ranked_rows = np.asarray(ranked_rows, dtype=int)
    selections = [
        (f"Selected high-correlation example {rank + 1}", int(index))
        for rank, index in enumerate(ranked_rows)
    ]
    names = []
    for _, index in selections:
        sample_name = pairs.iloc[index]["raman_sample_name"]
        material = metadata.loc[sample_name, "spectrum_identity"]
        names.append(
            f"{str(material).title()}, {pairs.iloc[index]['identity_key']}"
        )
    plot_examples(
        axis_values,
        predictions,
        targets,
        raw_pearson,
        selections,
        names,
        "Measured Raman (target)",
        args.output_dir / "figure_rruff_representative_examples",
    )

    output = pairs.copy()
    output["raw_pearson"] = raw_pearson
    output["baseline_corrected_pearson"] = corrected_pearson
    output.to_csv(
        args.output_dir / "rruff_external_validation_metrics.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nist-summary", type=Path, required=True)
    parser.add_argument("--nist-predictions", type=Path, required=True)
    parser.add_argument("--nist-targets", type=Path, required=True)
    parser.add_argument("--rruff-result-dir", type=Path, required=True)
    parser.add_argument("--rruff-pairs", type=Path, required=True)
    parser.add_argument("--rruff-metadata", type=Path, required=True)
    parser.add_argument("--rruff-raman-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    create_nist_figures(args)
    create_rruff_figures(args)
    print(args.output_dir)


if __name__ == "__main__":
    main()
