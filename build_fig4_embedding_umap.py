#!/usr/bin/env python
"""
Build Fig. 4-style UMAP panels for spectral/latent/flow embeddings.

For each feature type, the script uses the same samples, StandardScaler,
PCA-to-common-dimension, UMAP random seed, and UMAP parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from model import CrossModalVAE  # noqa: E402


GROUP_PATTERNS = [
    ("carboxyl", "C(=O)[OX2H1]"),
    ("ester", "C(=O)O[#6]"),
    ("amide", "C(=O)N"),
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("ketone", "[#6][CX3](=O)[#6]"),
    ("nitrile", "C#N"),
    ("alkyne", "C#C"),
    ("alkene", "C=C"),
    ("aromatic ring", "a1aaaaa1"),
    ("hydroxyl", "[OX2H]"),
    ("amine", "[NX3;H2,H1,H0;!$(NC=O)]"),
    ("ether", "[OD2]([#6])[#6]"),
    ("alkyl halide", "[CX4][F,Cl,Br,I]"),
    ("sulfide", "[#16X2][#6]"),
    ("thiol", "[SX2H]"),
]

GROUP_COLORS = {
    "ether": "#4C78A8",
    "aromatic ring": "#F58518",
    "amine": "#54A24B",
    "hydroxyl": "#E45756",
    "alkene": "#72B7B2",
    "amide": "#B279A2",
    "ketone": "#FF9DA6",
    "alkyne": "#9D755D",
    "nitrile": "#BAB0AC",
    "aldehyde": "#8CD17D",
    "alkyl halide": "#B6992D",
    "ester": "#499894",
    "sulfide": "#D37295",
    "carboxyl": "#A0CBE8",
    "thiol": "#FABFD2",
    "other": "#BDBDBD",
}


def spectrum_to_heatmap_matrix(spectra: np.ndarray, heatmap_size: int) -> torch.Tensor:
    heatmaps = []
    side = int(np.sqrt(heatmap_size))
    if side * side != heatmap_size:
        raise ValueError(f"heatmap_size must be a square, got {heatmap_size}")
    patch_size = max(p for p in [20, 10, 8, 5, 4, 2, 1] if side % p == 0)
    num_patches_per_row = side // patch_size

    for spec in spectra:
        spec = np.asarray(spec, dtype=np.float32)
        if len(spec) != heatmap_size:
            x_old = np.linspace(0, 1, len(spec))
            x_new = np.linspace(0, 1, heatmap_size)
            spec = np.interp(x_new, x_old, spec).astype(np.float32)
        min_val = float(np.min(spec))
        max_val = float(np.max(spec))
        norm = (spec - min_val) / (max_val - min_val + 1e-8)
        patches = norm.reshape(-1, patch_size, patch_size)
        rows = [
            np.concatenate(
                patches[i * num_patches_per_row : (i + 1) * num_patches_per_row],
                axis=1,
            )
            for i in range(num_patches_per_row)
        ]
        heatmap = np.concatenate(rows, axis=0)
        heatmaps.append(heatmap[None, :, :])
    return torch.tensor(np.stack(heatmaps), dtype=torch.float32)


def extract_vae_mu(
    spectra: np.ndarray,
    checkpoint_path: Path,
    batch_size: int = 256,
    hidden_channels: int = 128,
    latent_dim: int = 128,
    image_size: int = 32,
):
    device = torch.device("cpu")
    model = CrossModalVAE(
        in_channels=1,
        hidden_channels=hidden_channels,
        latent_dim=latent_dim,
        image_size=(image_size, image_size),
        num_modes=3,
        use_physical_prior=False,
        physical_dim=7,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    mus = []
    with torch.no_grad():
        for start in range(0, len(spectra), batch_size):
            batch = spectrum_to_heatmap_matrix(
                spectra[start : start + batch_size],
                heatmap_size=image_size * image_size,
            ).to(device)
            mu, _ = model.encode(batch)
            mus.append(mu.cpu().numpy())
    return np.vstack(mus)


def assign_functional_group(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "other"
    for name, smarts in GROUP_PATTERNS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            return name
    return "other"


def project_umap(features: np.ndarray, pca_dim: int, seed: int, n_neighbors: int, min_dist: float):
    x = StandardScaler().fit_transform(features)
    n_comp = min(pca_dim, x.shape[1], x.shape[0] - 1)
    x_pca = PCA(n_components=n_comp, random_state=seed).fit_transform(x)
    return UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
        n_components=2,
    ).fit_transform(x_pca)


def main():
    parser = argparse.ArgumentParser(description="Draw Fig. 4 UMAP panels.")
    parser.add_argument("--feature_dir", type=Path, required=True)
    parser.add_argument("--vae_checkpoint", type=Path, required=True)
    parser.add_argument("--out_svg", type=Path, required=True)
    parser.add_argument("--dataset_name", type=str, default="geom")
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--vae_image_size", type=int, default=32)
    args = parser.parse_args()

    np.random.seed(args.seed)
    ir = np.load(args.feature_dir / "ir_raw_ir2raman.npy")
    raman = np.load(args.feature_dir / "raman_raw_ir2raman.npy")
    flow = np.load(args.feature_dir / "flow_embedding_ir2raman.npy")
    props = pd.read_csv(args.feature_dir / "molecular_properties_ir2raman.csv")

    n = min(len(ir), len(raman), len(flow), len(props))
    if args.max_samples and n > args.max_samples:
        idx = np.random.default_rng(args.seed).choice(n, size=args.max_samples, replace=False)
        idx.sort()
    else:
        idx = np.arange(n)

    ir = ir[idx]
    raman = raman[idx]
    flow = flow[idx]
    props = props.iloc[idx].reset_index(drop=True)
    labels = props["smiles"].map(assign_functional_group).to_numpy()

    vae_mu = extract_vae_mu(ir, args.vae_checkpoint, image_size=args.vae_image_size)
    features = {
        "Raw IR": ir,
        "Raw Raman": raman,
        "VAE latent": vae_mu,
        "Flow embedding": flow,
    }
    coords = {
        name: project_umap(x, args.pca_dim, args.seed, args.n_neighbors, args.min_dist)
        for name, x in features.items()
    }

    counts = pd.Series(labels).value_counts()
    ordered_groups = [g for g in GROUP_COLORS if g in counts.index]

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(15.8, 4.4), constrained_layout=True)
    for ax, (name, xy) in zip(axes, coords.items()):
        for group in ordered_groups:
            mask = labels == group
            if not np.any(mask):
                continue
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=8,
                alpha=0.78,
                linewidths=0,
                color=GROUP_COLORS[group],
                label=group,
            )
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#444444")

    handles, labels_legend = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="lower center",
        ncol=8,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle(
        f"{args.dataset_name}: matched-sample UMAP by functional group",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.06,
        f"StandardScaler -> PCA({args.pca_dim}) -> UMAP(n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, seed={args.seed}); n={len(idx)}",
        fontsize=9,
    )
    args.out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_svg}")


if __name__ == "__main__":
    main()
