"""Direct downstream runner placeholder."""
"""
Run downstream molecular-property prediction directly from saved spectra.

This script is intended for the paper's downstream section when checkpoints or
processed training files are not available locally. It reads existing
source/target/prediction spectra CSVs, computes RDKit properties from aligned
SMILES, and compares simple downstream regressors using:

- Source spectrum, e.g. IR
- Target spectrum, e.g. Raman
- VibraFlow predicted target spectrum
- Morgan fingerprint baseline

Default paths point to the current local QM9S IR -> Raman saved results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_RESULT_DIR = Path(
    "D:/VibraFlow-NCS/结果文件/图1/新结果/qm9s_flow_ir2raman_vibradit_seed2_v2"
)
DEFAULT_SMILES_CSV = Path(
    "C:/Users/ethzj/Desktop/VibraFlow_draw/QM9S/flow_ir2raman_smiles_only.csv"
)
DEFAULT_OUTPUT_DIR = Path("C:/Users/ethzj/Desktop/VibraFlow_draw/downstream_direct_qm9s_ir2raman")

PROPERTY_COLUMNS = [
    "MolWt",
    "LogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
]


def compute_rdkit_properties(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {name: np.nan for name in PROPERTY_COLUMNS}
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
    }


def compute_morgan_fingerprints(smiles_list: list[str], radius: int, n_bits: int) -> np.ndarray:
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        AllChem.DataStructs.ConvertToNumpyArray(bitvect, fps[i])
    return fps


def read_smiles(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "smiles" in df.columns:
        return df["smiles"].astype(str).tolist()
    if "SMILES" in df.columns:
        return df["SMILES"].astype(str).tolist()
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str).tolist()
    raise ValueError(f"Cannot identify SMILES column in {path}. Columns={df.columns.tolist()}")


def read_spectrum_csv(path: Path, max_samples: int | None = None) -> np.ndarray:
    print(f"Loading spectra: {path}")
    df = pd.read_csv(path, nrows=max_samples)
    df = df.drop(columns=["index"], errors="ignore")
    return df.to_numpy(dtype=np.float32)


def build_model(model_type: str, seed: int):
    if model_type == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))
    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_feature(
    X: np.ndarray,
    y_all: pd.DataFrame,
    feature_name: str,
    properties: list[str],
    model_type: str,
    seeds: list[int],
    test_size: float,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for prop in properties:
        y = y_all[prop].to_numpy(dtype=np.float32)
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X_prop = X[mask]
        y_prop = y[mask]
        if len(y_prop) < 10:
            print(f"Skip {feature_name} / {prop}: too few valid samples ({len(y_prop)})")
            continue

        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X_prop, y_prop, test_size=test_size, random_state=seed
            )
            model = build_model(model_type, seed)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            rows.append(
                {
                    "Feature": feature_name,
                    "Property": prop,
                    "Model": model_type,
                    "Seed": seed,
                    "N": int(len(y_prop)),
                    "R2": float(r2_score(y_test, pred)),
                    "RMSE": float(np.sqrt(mse)),
                    "MAE": float(mean_absolute_error(y_test, pred)),
                }
            )
            print(
                f"{feature_name:24s} | {prop:18s} | seed={seed} | "
                f"R2={rows[-1]['R2']:.4f}, RMSE={rows[-1]['RMSE']:.4f}"
            )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["Feature", "Property", "Model"], as_index=False)
        .agg(
            N=("N", "first"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
        )
        .fillna(0.0)
    )


def plot_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    feature_order = ["Source IR", "Target Raman", "VibraFlow prediction", "Morgan fingerprint"]
    properties = [p for p in PROPERTY_COLUMNS if p in set(summary_df["Property"])]

    n_props = len(properties)
    fig, axes = plt.subplots(
        n_props,
        1,
        figsize=(8.2, max(2.2, 1.55 * n_props)),
        sharex=True,
        squeeze=False,
    )
    colors = {
        "Source IR": "#4E79A7",
        "Target Raman": "#F28E2B",
        "VibraFlow prediction": "#E15759",
        "Morgan fingerprint": "#59A14F",
    }

    for ax, prop in zip(axes[:, 0], properties):
        sub = summary_df[summary_df["Property"] == prop].copy()
        sub["Feature"] = pd.Categorical(sub["Feature"], categories=feature_order, ordered=True)
        sub = sub.sort_values("Feature")
        ax.bar(
            sub["Feature"].astype(str),
            sub["R2_mean"],
            yerr=sub["R2_std"],
            color=[colors.get(x, "#777777") for x in sub["Feature"].astype(str)],
            edgecolor="black",
            linewidth=0.4,
            capsize=2,
        )
        ax.set_title(prop, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel("R2")
        ax.axhline(0, color="grey45", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1, 0].tick_params(axis="x", rotation=25)
    fig.suptitle("Downstream molecular-property prediction", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "downstream_direct_r2_summary.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "downstream_direct_r2_summary.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct downstream property prediction from saved spectra."
    )
    parser.add_argument("--result_dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--direction", type=str, default="ir2raman", choices=["ir2raman", "raman2ir"])
    parser.add_argument("--smiles_csv", type=Path, default=DEFAULT_SMILES_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_type", type=str, default="ridge", choices=["ridge", "rf"])
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--fp_radius", type=int, default=2)
    parser.add_argument("--fp_n_bits", type=int, default=2048)
    parser.add_argument(
        "--properties",
        nargs="+",
        default=PROPERTY_COLUMNS,
        help=f"RDKit properties to predict. Default: {' '.join(PROPERTY_COLUMNS)}",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed, args.seed + args.num_seeds))

    pred_path = args.result_dir / f"flow_{args.direction}_preds.csv"
    target_path = args.result_dir / f"flow_{args.direction}_targets.csv"
    source_path = args.result_dir / f"flow_{args.direction}_sources.csv"
    for path in [source_path, target_path, pred_path, args.smiles_csv]:
        if not path.exists():
            raise FileNotFoundError(path)

    smiles_list = read_smiles(args.smiles_csv)
    if args.max_samples is not None:
        smiles_list = smiles_list[: args.max_samples]

    properties_df = pd.DataFrame([compute_rdkit_properties(smi) for smi in smiles_list])
    properties_df.insert(0, "smiles", smiles_list)
    properties_df.to_csv(args.output_dir / "downstream_direct_properties.csv", index=False)

    rows: list[dict[str, float | str | int]] = []
    feature_specs = [
        ("Source IR" if args.direction == "ir2raman" else "Source Raman", source_path),
        ("Target Raman" if args.direction == "ir2raman" else "Target IR", target_path),
        ("VibraFlow prediction", pred_path),
    ]

    n_expected = len(properties_df)
    for feature_name, path in feature_specs:
        X = read_spectrum_csv(path, max_samples=args.max_samples)
        n = min(n_expected, X.shape[0])
        rows.extend(
            evaluate_feature(
                X[:n],
                properties_df.iloc[:n].reset_index(drop=True),
                feature_name,
                args.properties,
                args.model_type,
                seeds,
                args.test_size,
            )
        )
        del X

    print("Computing Morgan fingerprints...")
    fp = compute_morgan_fingerprints(smiles_list[:n_expected], args.fp_radius, args.fp_n_bits)
    rows.extend(
        evaluate_feature(
            fp,
            properties_df.iloc[: fp.shape[0]].reset_index(drop=True),
            "Morgan fingerprint",
            args.properties,
            args.model_type,
            seeds,
            args.test_size,
        )
    )

    results_df = pd.DataFrame(rows)
    summary_df = summarize(results_df)
    results_df.to_csv(args.output_dir / "downstream_direct_results_per_seed.csv", index=False)
    summary_df.to_csv(args.output_dir / "downstream_direct_summary.csv", index=False)
    plot_summary(summary_df, args.output_dir)

    print("\nSaved:")
    print(args.output_dir / "downstream_direct_results_per_seed.csv")
    print(args.output_dir / "downstream_direct_summary.csv")
    print(args.output_dir / "downstream_direct_r2_summary.png")


if __name__ == "__main__":
    main()
