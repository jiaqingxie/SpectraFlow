"""Plot selected OpenSpecy target and predicted Raman spectra."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import pandas as pd


def load_values(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.set_index("index")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--prefix", default="vae_ir2raman")
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = load_values(args.result_dir / f"{args.prefix}_preds.csv")
    targets = load_values(args.result_dir / f"{args.prefix}_targets.csv")
    scores = pd.read_csv(
        args.result_dir / f"{args.prefix}_r2_per_sample.csv"
    ).set_index("index")
    pairs = pd.read_csv(args.data_dir / "test_pairs.csv")
    metadata = pd.read_csv(args.data_dir.parent / "openspecy_cov100" / "metadata.csv")
    material_by_rruff = (
        metadata.dropna(subset=["rruffid"])
        .assign(rruffid=lambda frame: frame["rruffid"].str.lower())
        .drop_duplicates("rruffid")
        .set_index("rruffid")["spectrum_identity"]
    )

    with h5py.File(args.data_dir / "test_raman.h5", "r") as handle:
        x_axis = handle["x_axis"][:]

    figure, axes = plt.subplots(
        len(args.indices),
        1,
        figsize=(10, 3.15 * len(args.indices)),
        sharex=True,
        constrained_layout=True,
    )
    if len(args.indices) == 1:
        axes = [axes]

    for axis, index in zip(axes, args.indices):
        identity = str(pairs.iloc[index]["identity_key"]).lower()
        material = material_by_rruff.get(identity, "unknown")
        target = targets.loc[index].to_numpy()
        prediction = predictions.loc[index].to_numpy()
        score = float(scores.loc[index, "r2"])

        axis.plot(x_axis, target, color="#2878B5", linewidth=1.7, label="Experimental Raman")
        axis.plot(
            x_axis,
            prediction,
            color="#E8752D",
            linewidth=1.4,
            linestyle="--",
            label="VAE prediction",
        )
        axis.set_title(
            f"{material.title()} ({identity.upper()}), sample index {index}, "
            f"$R^2$ = {score:.3f}"
        )
        axis.set_ylabel("Normalized intensity")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, loc="upper right")

    axes[-1].set_xlabel(r"Raman shift (cm$^{-1}$)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
