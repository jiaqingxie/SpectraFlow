#!/usr/bin/env python
"""Draw UMAP panels for one continuous property across multiple datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_fig4_embedding_umap import extract_vae_mu, project_umap

FEATURES = {
    "Raw IR": "ir_raw_ir2raman.npy",
    "Raw Raman": "raman_raw_ir2raman.npy",
    "Flow embedding": "flow_embedding_ir2raman.npy",
}
FEATURE_ORDER = ["Raw IR", "Raw Raman", "VAE latent", "Flow embedding"]


def robust_limits(values: np.ndarray):
    finite = values[np.isfinite(values)]
    lo, hi = np.percentile(finite, [2, 98])
    if np.isclose(lo, hi):
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if np.isclose(lo, hi):
        hi = lo + 1.0
    return float(lo), float(hi)


def load_dataset(feature_root: Path, dataset: str, property_name: str, max_samples: int, seed: int):
    feature_dir = feature_root / dataset
    ir = np.load(feature_dir / FEATURES["Raw IR"])
    raman = np.load(feature_dir / FEATURES["Raw Raman"])
    flow = np.load(feature_dir / FEATURES["Flow embedding"])
    props = pd.read_csv(feature_dir / "molecular_properties_ir2raman.csv")
    if property_name not in props.columns:
        raise ValueError(f"{property_name} not found in {feature_dir / 'molecular_properties_ir2raman.csv'}")

    n = min(len(ir), len(raman), len(flow), len(props))
    if max_samples and n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        idx.sort()
    else:
        idx = np.arange(n)

    return {
        "ir": ir[idx],
        "raman": raman[idx],
        "flow": flow[idx],
        "property": props.iloc[idx][property_name].to_numpy(dtype=float),
        "n": len(idx),
    }


def main():
    parser = argparse.ArgumentParser(description="UMAP for one property across datasets.")
    parser.add_argument("--feature_root", type=Path, required=True)
    parser.add_argument("--vae_checkpoint", type=Path, required=True)
    parser.add_argument("--out_png", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["geom", "nist_ir", "peptide"])
    parser.add_argument("--property", default="LogP")
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--vae_image_size", type=int, default=32)
    args = parser.parse_args()

    datasets = {
        ds: load_dataset(args.feature_root, ds, args.property, args.max_samples, args.seed)
        for ds in args.datasets
    }
    all_values = np.concatenate([d["property"] for d in datasets.values()])
    vmin, vmax = robust_limits(all_values)

    coords = {}
    for ds, data in datasets.items():
        vae_mu = extract_vae_mu(data["ir"], args.vae_checkpoint, image_size=args.vae_image_size)
        arrays = {
            "Raw IR": data["ir"],
            "Raw Raman": data["raman"],
            "VAE latent": vae_mu,
            "Flow embedding": data["flow"],
        }
        coords[ds] = {
            name: project_umap(x, args.pca_dim, args.seed, args.n_neighbors, args.min_dist)
            for name, x in arrays.items()
        }

    plt.rcParams.update({"font.family": "Arial", "axes.unicode_minus": False})
    n_rows = len(args.datasets)
    n_cols = len(FEATURE_ORDER)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.45 * n_cols + 0.9, 3.8 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    last_scatter = None
    for row, ds in enumerate(args.datasets):
        values = datasets[ds]["property"]
        for col, feature in enumerate(FEATURE_ORDER):
            ax = axes[row, col]
            xy = coords[ds][feature]
            last_scatter = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=values,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                s=8,
                alpha=0.82,
                linewidths=0,
            )
            if row == 0:
                ax.set_title(feature, fontsize=18, fontweight="bold")
            if col == 0:
                ax.set_ylabel(ds, fontsize=18, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("UMAP 1", fontsize=12)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#444444")

    cbar = fig.colorbar(last_scatter, ax=axes[:, :], shrink=0.86, pad=0.012)
    cbar.set_label(args.property, fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=13)
    fig.suptitle(f"UMAP colored by {args.property}", fontsize=24, fontweight="bold")

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
