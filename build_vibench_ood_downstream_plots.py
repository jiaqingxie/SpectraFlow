#!/usr/bin/env python
"""
Plot Vibench OOD downstream results from train_downstream.py summary logs.

The script parses "Seed Summary" blocks and writes:
- a tidy metrics CSV
- overview R2 heatmaps by feature
- a Flow-vs-raw R2 gain heatmap
- per-dataset grouped barplots for property/model comparisons
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from matplotlib.colors import ListedColormap
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_ORDER = ["pahs", "nist_ir", "peptide_mod", "peptide", "geom", "zinc15", "qm9", "mols"]
PROPERTY_ORDER = [
    "LogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
    "NumAromaticRings",
    "MolMR",
    "LabuteASA",
]
EXCLUDED_PROPERTIES = {"MolWt", "HeavyAtomCount"}
FEATURE_ORDER = ["IR raw", "Raman raw", "Flow embedding", "Morgan fingerprint", "Flow + Morgan FP"]
FEATURE_COLORS = {
    "IR raw": "#7f7f7f",
    "Raman raw": "#4c78a8",
    "Flow embedding": "#f58518",
    "Morgan fingerprint": "#54a24b",
    "Flow + Morgan FP": "#b279a2",
}
FEATURE_LINESTYLES = {
    "IR raw": "-",
    "Raman raw": "-",
    "Flow embedding": "-",
    "Morgan fingerprint": "-",
    "Flow + Morgan FP": "--",
}
FEATURE_MARKERS = {
    "IR raw": None,
    "Raman raw": None,
    "Flow embedding": None,
    "Morgan fingerprint": "o",
    "Flow + Morgan FP": "s",
}
FEATURE_LABELS = {
    "IR raw": "IR",
    "Raman raw": "Ra",
    "Flow embedding": "Flow",
    "Morgan fingerprint": "FP",
    "Flow + Morgan FP": "F+FP",
}
PROPERTY_LABELS = {
    "LogP": "LogP",
    "TPSA": "TPSA",
    "NumHDonors": "H donors",
    "NumHAcceptors": "H acceptors",
    "NumRotatableBonds": "Rot. bonds",
    "RingCount": "Rings",
    "FractionCSP3": "Frac. CSP3",
    "NumAromaticRings": "Arom. rings",
    "MolMR": "MolMR",
    "LabuteASA": "LabuteASA",
}


def configure_plot_style():
    font_candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\Arial.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            break

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

SUMMARY_RE = re.compile(
    r"^=== Seed Summary: (?P<property>.+?) \| Features: (?P<feature>.+?) "
    r"\| Model: (?P<model>.+?) \| Seeds: (?P<seeds>\d+) ===$"
)
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
METRIC_RE = re.compile(rf"^(?P<name>R\^2|RMSE)\s*:\s*(?P<mean>{FLOAT_RE})\s+\S+\s+(?P<std>{FLOAT_RE})$")


def parse_summary(path: Path) -> pd.DataFrame:
    records = []
    lines = path.read_text(errors="replace").splitlines()
    dataset = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("TRAIN DATASET:"):
            dataset = line.split(":", 1)[1].strip()
            i += 1
            continue

        match = SUMMARY_RE.match(line)
        if not match:
            i += 1
            continue

        if dataset is None:
            raise ValueError(f"Found Seed Summary before dataset marker near line {i + 1}")

        values = {}
        for metric_line in lines[i + 1 : i + 3]:
            metric_match = METRIC_RE.match(metric_line.strip())
            if metric_match:
                metric_name = "r2" if metric_match.group("name") == "R^2" else "rmse"
                values[f"{metric_name}_mean"] = float(metric_match.group("mean"))
                values[f"{metric_name}_std"] = float(metric_match.group("std"))

        if {"r2_mean", "r2_std", "rmse_mean", "rmse_std"}.issubset(values):
            records.append(
                {
                    "dataset": dataset,
                    "property": match.group("property"),
                    "feature": match.group("feature"),
                    "model": match.group("model"),
                    "seeds": int(match.group("seeds")),
                    **values,
                }
            )
        i += 3

    if not records:
        raise ValueError(f"No Seed Summary records parsed from {path}")

    df = pd.DataFrame.from_records(records)
    df = df[~df["property"].isin(EXCLUDED_PROPERTIES)]
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    df["property"] = pd.Categorical(df["property"], PROPERTY_ORDER, ordered=True)
    df["feature"] = pd.Categorical(df["feature"], FEATURE_ORDER, ordered=True)
    return df.sort_values(["dataset", "property", "feature"]).reset_index(drop=True)


def _ordered_values(values, preferred_order):
    present = [value for value in preferred_order if value in set(values)]
    extras = sorted(set(values) - set(present))
    return present + extras


def _savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        fig.savefig(tmp_path, dpi=240, bbox_inches="tight")
        if tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Generated empty plot file: {tmp_path}")
        tmp_path.replace(path)
    finally:
        plt.close(fig)
        if tmp_path.exists():
            tmp_path.unlink()


def plot_feature_heatmaps(df: pd.DataFrame, out_dir: Path):
    datasets = _ordered_values(df["dataset"].astype(str).unique(), DATASET_ORDER)
    properties = _ordered_values(df["property"].astype(str).unique(), PROPERTY_ORDER)
    features = _ordered_values(df["feature"].astype(str).unique(), FEATURE_ORDER)

    fig, axes = plt.subplots(
        len(features),
        1,
        figsize=(max(12, len(properties) * 0.85), max(9, len(features) * 1.8)),
        sharex=True,
    )
    if len(features) == 1:
        axes = [axes]

    vmin, vmax = -1.0, 1.0
    last_im = None
    for ax, feature in zip(axes, features):
        pivot = (
            df[df["feature"].astype(str) == feature]
            .pivot(index="dataset", columns="property", values="r2_mean")
            .reindex(index=datasets, columns=properties)
        )
        matrix = pivot.to_numpy(dtype=float)
        last_im = ax.imshow(np.clip(matrix, vmin, vmax), aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax.set_title(f"{feature} R2", loc="left", fontsize=11, weight="bold")
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets)
        ax.grid(False)

    axes[-1].set_xticks(range(len(properties)))
    axes[-1].set_xticklabels(properties, rotation=35, ha="right")
    fig.suptitle("Vibench OOD downstream R2", fontsize=14, weight="bold")
    fig.subplots_adjust(right=0.90, hspace=0.32)
    cbar_ax = fig.add_axes([0.92, 0.16, 0.015, 0.68])
    fig.colorbar(last_im, cax=cbar_ax, label="Mean R2")
    _savefig(fig, out_dir / "vibench_ood_r2_feature_heatmaps.png")


def plot_flow_gain_heatmap(df: pd.DataFrame, out_dir: Path):
    datasets = _ordered_values(df["dataset"].astype(str).unique(), DATASET_ORDER)
    properties = _ordered_values(df["property"].astype(str).unique(), PROPERTY_ORDER)

    flow = df[df["feature"].astype(str) == "Flow embedding"].pivot(
        index="dataset", columns="property", values="r2_mean"
    )
    raw = df[df["feature"].astype(str).isin(["IR raw", "Raman raw"])]
    best_raw = raw.pivot_table(
        index="dataset",
        columns="property",
        values="r2_mean",
        aggfunc="max",
        observed=True,
    )
    gain = flow.subtract(best_raw).reindex(index=datasets, columns=properties)
    matrix = gain.to_numpy(dtype=float)

    finite = matrix[np.isfinite(matrix)]
    limit = max(0.25, float(np.nanpercentile(np.abs(finite), 95))) if finite.size else 1.0
    limit = min(limit, 2.0)

    fig, ax = plt.subplots(figsize=(max(12, len(properties) * 0.9), max(5, len(datasets) * 0.55)))
    im = ax.imshow(np.clip(matrix, -limit, limit), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_title("Flow gain over best raw feature", loc="left", fontsize=14, weight="bold")
    ax.set_xlabel("Property")
    ax.set_ylabel("Dataset")
    ax.set_xticks(range(len(properties)))
    ax.set_xticklabels(properties, rotation=35, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)

    for row in range(len(datasets)):
        for col in range(len(properties)):
            value = matrix[row, col]
            if math.isfinite(value):
                ax.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, label="Delta mean R2")
    _savefig(fig, out_dir / "vibench_ood_flow_vs_raw_gain_heatmap.png")


def plot_mean_feature_bars(df: pd.DataFrame, out_dir: Path):
    datasets = _ordered_values(df["dataset"].astype(str).unique(), DATASET_ORDER)
    features = _ordered_values(df["feature"].astype(str).unique(), FEATURE_ORDER)
    summary = (
        df.assign(r2_clipped=df["r2_mean"].clip(-1, 1))
        .groupby(["dataset", "feature"], observed=True)["r2_clipped"]
        .median()
        .unstack("feature")
        .reindex(index=datasets, columns=features)
    )

    x = np.arange(len(datasets))
    width = 0.8 / max(1, len(features))
    fig, ax = plt.subplots(figsize=(max(11, len(datasets) * 0.9), 5.2))
    for idx, feature in enumerate(features):
        offset = (idx - (len(features) - 1) / 2) * width
        ax.bar(
            x + offset,
            summary[feature].to_numpy(dtype=float),
            width=width,
            label=feature,
            color=FEATURE_COLORS.get(feature),
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Median R2 by dataset", loc="left", fontsize=14, weight="bold")
    ax.set_ylabel("Median mean R2")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(ncol=min(3, len(features)), frameon=False, loc="upper left", bbox_to_anchor=(0, 1.18))
    ax.grid(axis="y", alpha=0.25)
    _savefig(fig, out_dir / "vibench_ood_feature_median_r2_by_dataset.png")


def plot_dataset_bars(df: pd.DataFrame, out_dir: Path):
    datasets = _ordered_values(df["dataset"].astype(str).unique(), DATASET_ORDER)
    properties = _ordered_values(df["property"].astype(str).unique(), PROPERTY_ORDER)
    features = _ordered_values(df["feature"].astype(str).unique(), FEATURE_ORDER)
    dataset_dir = out_dir / "by_dataset"

    for dataset in datasets:
        subset = df[df["dataset"].astype(str) == dataset]
        pivot = (
            subset.pivot(index="property", columns="feature", values="r2_mean")
            .reindex(index=properties, columns=features)
            .clip(-1, 1)
        )

        x = np.arange(len(properties))
        width = 0.82 / max(1, len(features))
        fig, ax = plt.subplots(figsize=(max(13, len(properties) * 0.9), 5.8))
        for idx, feature in enumerate(features):
            offset = (idx - (len(features) - 1) / 2) * width
            ax.bar(
                x + offset,
                pivot[feature].to_numpy(dtype=float),
                width=width,
                label=feature,
                color=FEATURE_COLORS.get(feature),
            )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{dataset}: R2 by property", loc="left", fontsize=14, weight="bold")
        ax.set_ylabel("Mean R2")
        ax.set_xlabel("Property")
        ax.set_xticks(x)
        ax.set_xticklabels(properties, rotation=35, ha="right")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(ncol=min(3, len(features)), frameon=False, loc="upper left", bbox_to_anchor=(0, 1.18))
        ax.grid(axis="y", alpha=0.25)
        _savefig(fig, dataset_dir / f"{dataset}_r2_by_property_and_feature.png")


def plot_best_feature_counts(df: pd.DataFrame, out_dir: Path):
    idx = df.groupby(["dataset", "property"], observed=True)["r2_mean"].idxmax()
    best = df.loc[idx]
    counts = (
        best.groupby(["dataset", "feature"], observed=True)
        .size()
        .unstack("feature", fill_value=0)
        .reindex(index=_ordered_values(df["dataset"].astype(str).unique(), DATASET_ORDER), columns=FEATURE_ORDER)
        .fillna(0)
    )

    datasets = list(counts.index.astype(str))
    features = [f for f in FEATURE_ORDER if f in counts.columns]
    x = np.arange(len(datasets))
    bottom = np.zeros(len(datasets))

    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 0.8), 5))
    for feature in features:
        values = counts[feature].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=feature, color=FEATURE_COLORS.get(feature))
        bottom += values

    ax.set_title("Best feature count", loc="left", fontsize=14, weight="bold")
    ax.set_ylabel("Number of winning properties")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.legend(ncol=min(3, len(features)), frameon=False, loc="upper left", bbox_to_anchor=(0, 1.18))
    ax.grid(axis="y", alpha=0.25)
    _savefig(fig, out_dir / "vibench_ood_best_feature_counts_by_dataset.png")


def plot_best_feature_tilemap(df: pd.DataFrame, out_dir: Path):
    plot_dir = out_dir / "non_bar"
    datasets = _ordered_values(df["dataset"].astype(str).unique(), DATASET_ORDER)
    properties = _ordered_values(df["property"].astype(str).unique(), PROPERTY_ORDER)
    features = _ordered_values(df["feature"].astype(str).unique(), FEATURE_ORDER)
    feature_to_idx = {feature: idx for idx, feature in enumerate(features)}

    idx = df.groupby(["dataset", "property"], observed=True)["r2_mean"].idxmax()
    best = df.loc[idx].copy()
    best["feature_idx"] = best["feature"].astype(str).map(feature_to_idx)

    feature_matrix = (
        best.pivot(index="dataset", columns="property", values="feature_idx")
        .reindex(index=datasets, columns=properties)
        .to_numpy(dtype=float)
    )
    label_matrix = (
        best.assign(label=best["feature"].astype(str).map(FEATURE_LABELS))
        .pivot(index="dataset", columns="property", values="label")
        .reindex(index=datasets, columns=properties)
        .to_numpy()
    )

    cmap = ListedColormap([FEATURE_COLORS.get(feature) for feature in features])
    fig, ax = plt.subplots(figsize=(max(12, len(properties) * 0.9), max(5, len(datasets) * 0.55)))
    ax.imshow(feature_matrix, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(features) - 0.5)
    ax.set_title("Winning feature by property", loc="left", fontsize=14, weight="bold")
    ax.set_xlabel("Property")
    ax.set_ylabel("Dataset")
    ax.set_xticks(range(len(properties)))
    ax.set_xticklabels(properties, rotation=35, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)

    for row in range(len(datasets)):
        for col in range(len(properties)):
            if math.isfinite(feature_matrix[row, col]):
                ax.text(col, row, str(label_matrix[row, col]), ha="center", va="center", fontsize=8, color="black")

    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=FEATURE_COLORS.get(feature), markersize=9)
        for feature in features
    ]
    ax.legend(handles, features, ncol=min(3, len(features)), frameon=False, loc="upper left", bbox_to_anchor=(0, 1.18))
    ax.set_xticks(np.arange(-0.5, len(properties), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    _savefig(fig, plot_dir / "vibench_ood_winning_feature_tilemap.png")


def plot_pairwise_scatter(df: pd.DataFrame, out_dir: Path):
    plot_dir = out_dir / "non_bar"
    datasets = _ordered_values(df["dataset"].astype(str).unique(), DATASET_ORDER)
    dataset_colors = dict(zip(datasets, plt.get_cmap("tab10").colors))

    wide = df.pivot_table(
        index=["dataset", "property"],
        columns="feature",
        values="r2_mean",
        observed=True,
    ).reset_index()
    wide["Best raw"] = wide[["IR raw", "Raman raw"]].max(axis=1)

    comparisons = [
        ("Best raw", "Flow embedding", "Flow vs raw"),
        ("Morgan fingerprint", "Flow + Morgan FP", "F+FP vs FP"),
        ("Flow embedding", "Flow + Morgan FP", "F+FP vs Flow"),
    ]

    fig, axes = plt.subplots(1, len(comparisons), figsize=(15, 4.6), sharex=True, sharey=True)
    for ax, (x_col, y_col, title) in zip(axes, comparisons):
        for dataset in datasets:
            subset = wide[wide["dataset"].astype(str) == dataset]
            ax.scatter(
                subset[x_col].clip(-1, 1),
                subset[y_col].clip(-1, 1),
                s=34,
                alpha=0.78,
                color=dataset_colors[dataset],
                label=dataset,
                edgecolors="white",
                linewidth=0.35,
            )
        ax.plot([-1, 1], [-1, 1], color="black", linewidth=0.9)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel(f"{x_col} R2")
        ax.grid(alpha=0.25)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)

    axes[0].set_ylabel("Compared feature R2")
    axes[-1].legend(ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.02))
    fig.suptitle("Pairwise feature comparison", fontsize=14, weight="bold")
    _savefig(fig, plot_dir / "vibench_ood_pairwise_feature_scatter.png")


def plot_dataset_dotplots(df: pd.DataFrame, out_dir: Path):
    plot_dir = out_dir / "non_bar" / "by_dataset_dotplot"
    datasets = _ordered_values(df["dataset"].astype(str).unique(), DATASET_ORDER)
    properties = _ordered_values(df["property"].astype(str).unique(), PROPERTY_ORDER)
    features = _ordered_values(df["feature"].astype(str).unique(), FEATURE_ORDER)
    offsets = np.linspace(-0.24, 0.24, len(features))

    for dataset in datasets:
        subset = df[df["dataset"].astype(str) == dataset]
        pivot = (
            subset.pivot(index="property", columns="feature", values="r2_mean")
            .reindex(index=properties, columns=features)
            .clip(-1, 1)
        )

        y = np.arange(len(properties))
        fig, ax = plt.subplots(figsize=(8.8, max(5.6, len(properties) * 0.38)))
        for offset, feature in zip(offsets, features):
            values = pivot[feature].to_numpy(dtype=float)
            ax.scatter(
                values,
                y + offset,
                s=42,
                color=FEATURE_COLORS.get(feature),
                label=feature,
                alpha=0.9,
                edgecolors="white",
                linewidth=0.35,
            )

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{dataset}: R2 dot plot", loc="left", fontsize=14, weight="bold")
        ax.set_xlabel("Mean R2")
        ax.set_ylabel("Property")
        ax.set_xlim(-1.05, 1.05)
        ax.set_yticks(y)
        ax.set_yticklabels(properties)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        ax.legend(ncol=min(3, len(features)), frameon=False, loc="upper left", bbox_to_anchor=(0, 1.16))
        _savefig(fig, plot_dir / f"{dataset}_r2_dotplot.png")


def _r2_to_radar_radius(values: np.ndarray) -> np.ndarray:
    return (np.clip(values, -1, 1) + 1.0) / 2.0


def plot_dataset_starplots(df: pd.DataFrame, out_dir: Path):
    plot_dir = out_dir / "starplots"
    datasets = _ordered_values(df["dataset"].astype(str).unique(), DATASET_ORDER)
    properties = _ordered_values(df["property"].astype(str).unique(), PROPERTY_ORDER)
    features = _ordered_values(df["feature"].astype(str).unique(), FEATURE_ORDER)
    labels = [PROPERTY_LABELS.get(prop, prop) for prop in properties]
    angles = np.linspace(0, 2 * np.pi, len(properties), endpoint=False)
    closed_angles = np.concatenate([angles, [angles[0]]])

    for dataset in datasets:
        subset = df[df["dataset"].astype(str) == dataset]
        pivot = (
            subset.pivot(index="feature", columns="property", values="r2_mean")
            .reindex(index=features, columns=properties)
        )

        fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw={"projection": "polar"})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["-1", "-0.5", "0", "0.5", "1"], fontsize=8)
        ax.set_rlabel_position(92)
        ax.grid(alpha=0.32)

        for feature in features:
            values = pivot.loc[feature].to_numpy(dtype=float)
            radii = _r2_to_radar_radius(values)
            closed_radii = np.concatenate([radii, [radii[0]]])
            ax.plot(
                closed_angles,
                closed_radii,
                linewidth=1.8,
                linestyle=FEATURE_LINESTYLES.get(feature, "-"),
                marker=FEATURE_MARKERS.get(feature),
                markersize=3.5 if FEATURE_MARKERS.get(feature) else 0,
                markerfacecolor="white" if FEATURE_MARKERS.get(feature) else FEATURE_COLORS.get(feature),
                markeredgewidth=0.8,
                color=FEATURE_COLORS.get(feature),
                label=feature,
            )
            fill_alpha = 0.035 if feature in {"Morgan fingerprint", "Flow + Morgan FP"} else 0.06
            ax.fill(closed_angles, closed_radii, color=FEATURE_COLORS.get(feature), alpha=fill_alpha)

        ax.set_title(dataset, fontsize=14, weight="bold", pad=24)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
        _savefig(fig, plot_dir / f"{dataset}_r2_starplot.png")


def plot_starplot_small_multiples(df: pd.DataFrame, out_dir: Path):
    plot_dir = out_dir / "starplots"
    datasets = _ordered_values(df["dataset"].astype(str).unique(), DATASET_ORDER)
    properties = _ordered_values(df["property"].astype(str).unique(), PROPERTY_ORDER)
    features = ["Raman raw", "Flow embedding", "Morgan fingerprint", "Flow + Morgan FP"]
    features = [feature for feature in features if feature in set(df["feature"].astype(str))]
    labels = [PROPERTY_LABELS.get(prop, prop) for prop in properties]
    angles = np.linspace(0, 2 * np.pi, len(properties), endpoint=False)
    closed_angles = np.concatenate([angles, [angles[0]]])

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8.8),
        subplot_kw={"projection": "polar"},
    )
    axes = axes.ravel()
    handles = []

    for ax, dataset in zip(axes, datasets):
        subset = df[df["dataset"].astype(str) == dataset]
        pivot = (
            subset.pivot(index="feature", columns="property", values="r2_mean")
            .reindex(index=features, columns=properties)
        )

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["-1", "0", "1"], fontsize=7)
        ax.grid(alpha=0.25)
        ax.set_title(dataset, fontsize=12, weight="bold", pad=12)

        for feature in features:
            values = pivot.loc[feature].to_numpy(dtype=float)
            radii = _r2_to_radar_radius(values)
            closed_radii = np.concatenate([radii, [radii[0]]])
            line = ax.plot(
                closed_angles,
                closed_radii,
                linewidth=1.35,
                linestyle=FEATURE_LINESTYLES.get(feature, "-"),
                marker=FEATURE_MARKERS.get(feature),
                markersize=2.6 if FEATURE_MARKERS.get(feature) else 0,
                markerfacecolor="white" if FEATURE_MARKERS.get(feature) else FEATURE_COLORS.get(feature),
                markeredgewidth=0.65,
                color=FEATURE_COLORS.get(feature),
                label=feature,
            )[0]
            fill_alpha = 0.025 if feature in {"Morgan fingerprint", "Flow + Morgan FP"} else 0.04
            ax.fill(closed_angles, closed_radii, color=FEATURE_COLORS.get(feature), alpha=fill_alpha)
            if len(handles) < len(features):
                handles.append(line)

    for ax in axes[len(datasets) :]:
        ax.set_visible(False)

    fig.suptitle("Vibench OOD R2 starplots", fontsize=16, weight="bold", y=0.98)
    fig.legend(handles, features, ncol=len(features), frameon=False, loc="lower center")
    fig.subplots_adjust(top=0.90, bottom=0.10, hspace=0.36, wspace=0.28)
    _savefig(fig, plot_dir / "vibench_ood_r2_starplot_small_multiples.png")


def main():
    parser = argparse.ArgumentParser(description="Plot Vibench OOD downstream summary results.")
    parser.add_argument(
        "--summary_txt",
        type=Path,
        default=Path("downstream/vibench_ood_ir2raman_t0p5_vibradit_hidden_seed0_all_results.txt"),
        help="Summary txt produced by run_vibench_ood_downstream.sh.",
    )
    parser.add_argument(
        "--metrics_csv",
        type=Path,
        default=None,
        help="Optional parsed metrics CSV. If provided, skip parsing --summary_txt.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/vibench_ood_downstream_plots"),
        help="Directory for CSV and PNG outputs.",
    )
    args = parser.parse_args()

    configure_plot_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "vibench_ood_downstream_metrics.csv"
    if args.metrics_csv is not None:
        df = pd.read_csv(args.metrics_csv)
        df = df[~df["property"].isin(EXCLUDED_PROPERTIES)]
        df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
        df["property"] = pd.Categorical(df["property"], PROPERTY_ORDER, ordered=True)
        df["feature"] = pd.Categorical(df["feature"], FEATURE_ORDER, ordered=True)
        df = df.sort_values(["dataset", "property", "feature"]).reset_index(drop=True)
    else:
        df = parse_summary(args.summary_txt)
        df.to_csv(csv_path, index=False)

    plot_feature_heatmaps(df, args.out_dir)
    plot_flow_gain_heatmap(df, args.out_dir)
    plot_mean_feature_bars(df, args.out_dir)
    plot_dataset_bars(df, args.out_dir)
    plot_best_feature_counts(df, args.out_dir)
    plot_best_feature_tilemap(df, args.out_dir)
    plot_pairwise_scatter(df, args.out_dir)
    plot_dataset_dotplots(df, args.out_dir)
    plot_dataset_starplots(df, args.out_dir)
    plot_starplot_small_multiples(df, args.out_dir)

    print(f"Parsed rows: {len(df)}")
    print(f"Metrics CSV: {args.metrics_csv or csv_path}")
    print(f"Wrote plots under: {args.out_dir}")


if __name__ == "__main__":
    main()
