#!/usr/bin/env python
"""Draw matched-sample UMAP panels colored by continuous molecular properties."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_fig4_embedding_umap import extract_vae_mu, project_umap


FEATURE_FILES = {
    "Raw IR": "ir_raw_ir2raman.npy",
    "Raw Raman": "raman_raw_ir2raman.npy",
    "Flow embedding": "flow_embedding_ir2raman.npy",
}


def robust_limits(values: np.ndarray):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [2, 98])
    if np.isclose(lo, hi):
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if np.isclose(lo, hi):
        hi = lo + 1.0
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(description="Draw continuous-property UMAP panels.")
    parser.add_argument("--feature_dir", type=Path, required=True)
    parser.add_argument("--vae_checkpoint", type=Path, required=True)
    parser.add_argument("--out_svg", type=Path, required=True)
    parser.add_argument("--dataset_name", type=str, default="geom")
    parser.add_argument(
        "--properties",
        nargs="+",
        default=["LogP", "TPSA", "RingCount", "FractionCSP3"],
    )
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--vae_image_size", type=int, default=32)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ir = np.load(args.feature_dir / FEATURE_FILES["Raw IR"])
    raman = np.load(args.feature_dir / FEATURE_FILES["Raw Raman"])
    flow = np.load(args.feature_dir / FEATURE_FILES["Flow embedding"])
    props = pd.read_csv(args.feature_dir / "molecular_properties_ir2raman.csv")

    missing = [name for name in args.properties if name not in props.columns]
    if missing:
        raise ValueError(f"Missing properties in CSV: {missing}")

    n = min(len(ir), len(raman), len(flow), len(props))
    if args.max_samples and n > args.max_samples:
        idx = rng.choice(n, size=args.max_samples, replace=False)
        idx.sort()
    else:
        idx = np.arange(n)

    ir = ir[idx]
    raman = raman[idx]
    flow = flow[idx]
    props = props.iloc[idx].reset_index(drop=True)
    vae_mu = extract_vae_mu(ir, args.vae_checkpoint, image_size=args.vae_image_size)

    feature_arrays = {
        "Raw IR": ir,
        "Raw Raman": raman,
        "VAE latent": vae_mu,
        "Flow embedding": flow,
    }
    coords = {
        name: project_umap(x, args.pca_dim, args.seed, args.n_neighbors, args.min_dist)
        for name, x in feature_arrays.items()
    }

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    n_rows = len(args.properties)
    n_cols = len(feature_arrays)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.45 * n_cols + 0.9, 3.85 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for row, prop in enumerate(args.properties):
        values = props[prop].to_numpy(dtype=float)
        vmin, vmax = robust_limits(values)
        for col, (feature_name, xy) in enumerate(coords.items()):
            ax = axes[row, col]
            sc = ax.scatter(
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
                ax.set_title(feature_name, fontsize=18, fontweight="bold")
            if col == 0:
                ax.set_ylabel(prop, fontsize=18, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("UMAP 1", fontsize=12)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#444444")

        cbar = fig.colorbar(sc, ax=axes[row, :], shrink=0.82, pad=0.012)
        cbar.set_label(prop, fontsize=15, fontweight="bold")
        cbar.ax.tick_params(labelsize=12)

    fig.suptitle(
        f"{args.dataset_name}: UMAP colored by molecular properties",
        fontsize=22,
        fontweight="bold",
    )
    args.out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_svg, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {args.out_svg}")


if __name__ == "__main__":
    main()
