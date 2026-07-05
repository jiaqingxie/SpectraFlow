from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


BASE = Path(r"D:/VibraFlow-NCS/结果文件/图1")
OUT_DIR = BASE / "scaffold_r2_barplots_seed2"
R2_DIR = OUT_DIR / "remote_r2"


RUNS = [
    {
        "dataset": "QM9S",
        "direction": "IR->Raman",
        "smiles": BASE / "QM9S结果" / "smiles" / "qm9s_test_smiles_seed2_from_zip_idmatch.txt",
        "r2": R2_DIR / "qm9s_flow_ir2raman_r2_per_sample.csv",
    },
    {
        "dataset": "QM9S",
        "direction": "Raman->IR",
        "smiles": BASE / "QM9S结果" / "smiles" / "qm9s_test_smiles_seed2_from_zip_idmatch.txt",
        "r2": R2_DIR / "qm9s_flow_raman2ir_r2_per_sample.csv",
    },
    {
        "dataset": "QMe14S",
        "direction": "IR->Raman",
        "smiles": BASE / "QMe14S结果" / "qme14s_test_smiles_seed2.txt",
        "r2": R2_DIR / "qme14s_flow_ir2raman_r2_per_sample.csv",
    },
    {
        "dataset": "QMe14S",
        "direction": "Raman->IR",
        "smiles": BASE / "QMe14S结果" / "qme14s_test_smiles_seed2.txt",
        "r2": R2_DIR / "qme14s_flow_raman2ir_r2_per_sample.csv",
    },
]


def read_smiles(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].strip().lower() == "smiles":
        return [x.strip() for x in lines[1:]]
    return [x.strip() for x in lines]


def read_r2(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append((int(row["index"]), float(row["r2"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def scaffold_type(mol: Chem.Mol) -> str:
    rings = mol.GetRingInfo().AtomRings()
    if not rings:
        return "acyclic"

    aromatic = 0
    hetero = 0
    carbocycle = 0
    for ring in rings:
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        is_aromatic = all(atom.GetIsAromatic() for atom in atoms)
        has_hetero = any(atom.GetAtomicNum() not in (1, 6) for atom in atoms)
        aromatic += int(is_aromatic)
        hetero += int(has_hetero)
        carbocycle += int(not has_hetero)

    if aromatic and hetero:
        return "aromatic heterocycle"
    if aromatic:
        return "aromatic carbocycle"
    if hetero:
        return "aliphatic heterocycle"
    if carbocycle:
        return "aliphatic carbocycle"
    return "ring other"


def atom_bin(n: int) -> str:
    if n <= 0:
        return "0"
    if n <= 5:
        return "1-5"
    if n <= 8:
        return "6-8"
    if n <= 12:
        return "9-12"
    return "13+"


def count_bin(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def ring_bin(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    return "4+"


def scaffold_class_name(
    base_type: str,
    ring_count: int,
    scaffold_atoms: int,
    scaffold_hetero: int,
    mol_heavy_atoms: int,
    mol_hetero: int,
) -> str:
    short_type = {
        "aromatic heterocycle": "arom hetero",
        "aromatic carbocycle": "arom carbo",
        "aliphatic heterocycle": "aliph hetero",
        "aliphatic carbocycle": "aliph carbo",
        "acyclic": "acyclic",
    }.get(base_type, base_type)
    if base_type == "acyclic":
        return (
            f"acyclic | MolA{atom_bin(mol_heavy_atoms)} "
            f"| Het{count_bin(mol_hetero)}"
        )
    return (
        f"{short_type} | R{ring_bin(ring_count)} "
        f"| A{atom_bin(scaffold_atoms)} "
        f"| Het{count_bin(scaffold_hetero)}"
    )


def scaffold_features(smiles: str) -> dict[str, object]:
    if not smiles:
        return {
            "valid_smiles": "false",
            "scaffold": "missing_smiles",
            "generic_scaffold": "missing_smiles",
            "scaffold_type": "missing_smiles",
            "scaffold_class": "missing_smiles",
            "ring_count": 0,
            "scaffold_atoms": 0,
            "scaffold_hetero": 0,
            "mol_heavy_atoms": 0,
            "mol_hetero_atoms": 0,
        }
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "valid_smiles": "false",
            "scaffold": "invalid_smiles",
            "generic_scaffold": "invalid_smiles",
            "scaffold_type": "invalid_smiles",
            "scaffold_class": "invalid_smiles",
            "ring_count": 0,
            "scaffold_atoms": 0,
            "scaffold_hetero": 0,
            "mol_heavy_atoms": 0,
            "mol_hetero_atoms": 0,
        }

    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    base_type = scaffold_type(mol)
    ring_count = int(mol.GetRingInfo().NumRings())
    mol_heavy_atoms = int(mol.GetNumHeavyAtoms())
    mol_hetero = int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6)))
    if scaf.GetNumAtoms() == 0:
        scaffold = "acyclic"
        generic = "acyclic"
        scaffold_atoms = 0
        scaffold_hetero = 0
    else:
        scaffold = Chem.MolToSmiles(scaf, isomericSmiles=False)
        generic = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaf), isomericSmiles=False)
        scaffold_atoms = int(scaf.GetNumHeavyAtoms())
        scaffold_hetero = int(sum(1 for atom in scaf.GetAtoms() if atom.GetAtomicNum() not in (1, 6)))

    return {
        "valid_smiles": "true",
        "scaffold": scaffold,
        "generic_scaffold": generic,
        "scaffold_type": base_type,
        "scaffold_class": scaffold_class_name(
            base_type,
            ring_count,
            scaffold_atoms,
            scaffold_hetero,
            mol_heavy_atoms,
            mol_hetero,
        ),
        "ring_count": ring_count,
        "scaffold_atoms": scaffold_atoms,
        "scaffold_hetero": scaffold_hetero,
        "mol_heavy_atoms": mol_heavy_atoms,
        "mol_hetero_atoms": mol_hetero,
    }


def summarise(rows: list[dict[str, object]], group_cols: list[str], min_n: int) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[col] for col in group_cols)].append(row)

    out: list[dict[str, object]] = []
    for key, vals in grouped.items():
        r2 = np.array([float(v["r2"]) for v in vals if not math.isnan(float(v["r2"]))], dtype=float)
        if len(r2) < min_n:
            continue
        item = {col: val for col, val in zip(group_cols, key)}
        item.update(
            {
                "n": int(len(r2)),
                "mean_r2": float(np.mean(r2)),
                "median_r2": float(np.median(r2)),
                "p25_r2": float(np.percentile(r2, 25)),
                "p75_r2": float(np.percentile(r2, 75)),
                "frac_r2_lt_0p5": float(np.mean(r2 < 0.5)),
                "frac_r2_ge_0p9": float(np.mean(r2 >= 0.9)),
            }
        )
        out.append(item)
    out.sort(key=lambda x: (str(x["dataset"]), str(x["direction"]), -float(x["mean_r2"])))
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def label_scaffold(row: dict[str, object]) -> str:
    scaf = str(row["scaffold"])
    generic = str(row["generic_scaffold"])
    if len(scaf) > 34:
        scaf = scaf[:31] + "..."
    if generic != str(row["scaffold"]) and generic not in {"acyclic", "missing_smiles", "invalid_smiles"}:
        return f"{scaf} | n={row['n']}"
    return f"{scaf} | n={row['n']}"


def plot_rank_bars(summary: list[dict[str, object]], mode: str, out_prefix: Path, top_k: int = 12) -> None:
    combos = [
        ("QM9S", "IR->Raman"),
        ("QM9S", "Raman->IR"),
        ("QMe14S", "IR->Raman"),
        ("QMe14S", "Raman->IR"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes_flat = axes.ravel()
    color = "#2B6CB0" if mode == "top" else "#B83232"

    for ax, (dataset, direction) in zip(axes_flat, combos):
        rows = [
            r
            for r in summary
            if r["dataset"] == dataset
            and r["direction"] == direction
            and r["scaffold"] not in {"missing_smiles", "invalid_smiles"}
        ]
        rows = sorted(rows, key=lambda r: float(r["mean_r2"]), reverse=(mode == "top"))[:top_k]
        rows = list(reversed(rows))
        if not rows:
            ax.set_axis_off()
            continue
        labels = [label_scaffold(r) for r in rows]
        values = [float(r["mean_r2"]) for r in rows]
        ax.barh(range(len(rows)), values, color=color, alpha=0.88)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Mean R2")
        ax.set_title(f"{dataset} {direction}")
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
        for y, value in enumerate(values):
            ax.text(min(value + 0.012, 0.98), y, f"{value:.3f}", va="center", fontsize=8)

    fig.suptitle(f"Murcko scaffold {mode} R2 ranking (min n=10)", fontsize=16, fontweight="bold")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_prefix.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def plot_type_bars(type_summary: list[dict[str, object]], out_prefix: Path) -> None:
    combos = [
        ("QM9S", "IR->Raman"),
        ("QM9S", "Raman->IR"),
        ("QMe14S", "IR->Raman"),
        ("QMe14S", "Raman->IR"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (dataset, direction) in zip(axes.ravel(), combos):
        rows = [
            r
            for r in type_summary
            if r["dataset"] == dataset
            and r["direction"] == direction
            and r["scaffold_type"] not in {"missing_smiles", "invalid_smiles"}
        ]
        rows = sorted(rows, key=lambda r: float(r["mean_r2"]))
        labels = [f"{r['scaffold_type']} | n={r['n']}" for r in rows]
        values = [float(r["mean_r2"]) for r in rows]
        ax.barh(range(len(rows)), values, color="#4A5568", alpha=0.86)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Mean R2")
        ax.set_title(f"{dataset} {direction}")
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
        for y, value in enumerate(values):
            ax.text(min(value + 0.012, 0.98), y, f"{value:.3f}", va="center", fontsize=8)
    fig.suptitle("Scaffold type R2 ranking", fontsize=16, fontweight="bold")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_prefix.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def shorten_label(value: object, max_len: int = 42) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


TYPE_COLORS = {
    "aromatic heterocycle": "#E15759",
    "aromatic carbocycle": "#EDC948",
    "aliphatic heterocycle": "#4E79A7",
    "aliphatic carbocycle": "#59A14F",
    "acyclic": "#BAB0AC",
}

TYPE_FROM_SHORT = {
    "arom hetero": "aromatic heterocycle",
    "arom carbo": "aromatic carbocycle",
    "aliph hetero": "aliphatic heterocycle",
    "aliph carbo": "aliphatic carbocycle",
    "acyclic": "acyclic",
}


def scaffold_family(raw_class: str) -> str:
    head = raw_class.split("|", 1)[0].strip()
    return TYPE_FROM_SHORT.get(head, head)


def format_scaffold_class_label(raw_class: str) -> str:
    parts = [p.strip() for p in raw_class.split("|")]
    if not parts:
        return raw_class
    family = scaffold_family(raw_class)
    family_label = " ".join(word.capitalize() for word in family.split())
    if family == "acyclic":
        mol_atoms = parts[1].replace("MolA", "") if len(parts) > 1 else ""
        het = parts[2].replace("Het", "") if len(parts) > 2 else ""
        return f"Acyclic · {mol_atoms} mol. atoms · {het} heteroatoms"
    rings = parts[1].replace("R", "") if len(parts) > 1 else ""
    atoms = parts[2].replace("A", "") if len(parts) > 2 else ""
    het = parts[3].replace("Het", "") if len(parts) > 3 else ""
    rings_label = f"{rings} ring{'s' if rings not in {'1', '4+'} else ''}"
    if rings == "4+":
        rings_label = "4+ rings"
    return f"{family_label} · {rings_label} · {atoms} scaffold atoms · {het} heteroatoms"


def format_scaffold_class_label_short(raw_class: str, n: int) -> str:
    parts = [p.strip() for p in raw_class.split("|")]
    if not parts:
        return raw_class
    family = scaffold_family(raw_class)
    short_family = {
        "aromatic heterocycle": "ArH",
        "aromatic carbocycle": "ArC",
        "aliphatic heterocycle": "AlH",
        "aliphatic carbocycle": "AlC",
        "acyclic": "Acy",
    }.get(family, family[:6])
    if family == "acyclic":
        mol_atoms = parts[1].replace("MolA", "") if len(parts) > 1 else ""
        het = parts[2].replace("Het", "") if len(parts) > 2 else ""
        return f"{short_family} M{mol_atoms} H{het} ({n})"
    rings = parts[1].replace("R", "") if len(parts) > 1 else ""
    atoms = parts[2].replace("A", "") if len(parts) > 2 else ""
    het = parts[3].replace("Het", "") if len(parts) > 3 else ""
    return f"{short_family} R{rings} A{atoms} H{het} ({n})"


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 0.8,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def plot_scaffold_class_bottom_polished(
    class_summary: list[dict[str, object]],
    out_prefix: Path,
    top_k: int = 10,
) -> None:
    apply_plot_style()
    combos = [
        ("QM9S", "IR->Raman", "IR → Raman"),
        ("QM9S", "Raman->IR", "Raman → IR"),
        ("QMe14S", "IR->Raman", "IR → Raman"),
        ("QMe14S", "Raman->IR", "Raman → IR"),
    ]
    excluded = {"missing_smiles", "invalid_smiles"}

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.0))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.13, wspace=0.68)

    label_fs = 9.5
    value_fs = 9.0

    for ax, (dataset, direction, direction_pretty) in zip(axes, combos):
        rows = [
            r
            for r in class_summary
            if r["dataset"] == dataset
            and r["direction"] == direction
            and str(r["scaffold_class"]) not in excluded
        ]
        rows = sorted(rows, key=lambda r: float(r["mean_r2"]))[:top_k]
        if not rows:
            ax.set_axis_off()
            continue

        labels = [
            format_scaffold_class_label_short(str(r["scaffold_class"]), int(r["n"]))
            for r in rows
        ]
        values = [float(r["mean_r2"]) for r in rows]
        colors = [TYPE_COLORS.get(scaffold_family(str(r["scaffold_class"])), "#9AA0A6") for r in rows]
        y_pos = np.arange(len(rows))

        vmin = min(values)
        vmax = max(values)
        span = max(vmax - vmin, 0.08)
        pad = max(0.03, span * 0.08)
        xmin = max(0.0, vmin - pad)
        data_xmax = min(1.0, vmax + pad * 0.10)
        label_space = span * 1.85
        value_pad = span * 0.04
        plot_left = max(0.0, xmin - label_space)
        ax.set_xlim(plot_left, data_xmax + value_pad * 2.2)

        bars = ax.barh(
            y_pos,
            [value - xmin for value in values],
            left=xmin,
            color=colors,
            height=0.78,
            edgecolor="#2F2F2F",
            linewidth=0.45,
            alpha=0.92,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=9.5)
        for y, lbl in zip(y_pos, labels):
            ax.text(
                xmin - value_pad * 1.2,
                y,
                lbl,
                ha="right",
                va="center",
                fontsize=label_fs,
                clip_on=True,
            )
        ax.set_xlabel("Mean R²", fontsize=11)
        ax.set_title(f"{dataset} · {direction_pretty}", fontsize=12, pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", linestyle=(0, (2, 3)), linewidth=0.6, color="#D0D0D0", alpha=0.9)
        ax.set_axisbelow(True)

        for bar, value in zip(bars, values):
            y = bar.get_y() + bar.get_height() / 2
            x_end = bar.get_x() + bar.get_width()
            label = f"{value:.3f}"
            if x_end - xmin > span * 0.34:
                ax.text(
                    x_end - value_pad * 0.35,
                    y,
                    label,
                    va="center",
                    ha="right",
                    fontsize=value_fs,
                    color="white",
                    fontweight="bold",
                    clip_on=True,
                )
            else:
                ax.text(
                    x_end + value_pad * 0.35,
                    y,
                    label,
                    va="center",
                    ha="left",
                    fontsize=value_fs,
                    color="#333333",
                    clip_on=True,
                )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color, edgecolor="#2F2F2F", linewidth=0.45, label=label)
        for label, color in TYPE_COLORS.items()
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.01),
        handlelength=1.0,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    for ext in ("png", "svg", "pdf"):
        fig.savefig(
            out_prefix.with_suffix(f".{ext}"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
    plt.close(fig)


def plot_group_rank_bars(
    summary: list[dict[str, object]],
    group_col: str,
    mode: str,
    title: str,
    out_prefix: Path,
    top_k: int = 14,
) -> None:
    combos = [
        ("QM9S", "IR->Raman"),
        ("QM9S", "Raman->IR"),
        ("QMe14S", "IR->Raman"),
        ("QMe14S", "Raman->IR"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    color = "#2F855A" if mode == "top" else "#C05621"
    excluded = {"missing_smiles", "invalid_smiles"}

    for ax, (dataset, direction) in zip(axes.ravel(), combos):
        rows = [
            r
            for r in summary
            if r["dataset"] == dataset
            and r["direction"] == direction
            and str(r[group_col]) not in excluded
        ]
        rows = sorted(rows, key=lambda r: float(r["mean_r2"]), reverse=(mode == "top"))[:top_k]
        rows = list(reversed(rows))
        if not rows:
            ax.set_axis_off()
            continue
        labels = [f"{shorten_label(r[group_col])} | n={r['n']}" for r in rows]
        values = [float(r["mean_r2"]) for r in rows]
        ax.barh(range(len(rows)), values, color=color, alpha=0.88)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Mean R2")
        ax.set_title(f"{dataset} {direction}")
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
        for y, value in enumerate(values):
            ax.text(min(value + 0.012, 0.98), y, f"{value:.3f}", va="center", fontsize=8)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_prefix.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_cache: dict[str, dict[str, object]] = {}
    per_sample: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []

    for run in RUNS:
        smiles_list = read_smiles(run["smiles"])
        r2_rows = read_r2(run["r2"])
        missing = 0
        out_of_range = 0
        for row_pos, (test_index, r2) in enumerate(r2_rows):
            if 0 <= test_index < len(smiles_list):
                smiles = smiles_list[test_index]
            else:
                smiles = ""
                out_of_range += 1
            if not smiles:
                missing += 1
            if smiles not in feature_cache:
                feature_cache[smiles] = scaffold_features(smiles)
            per_sample.append(
                {
                    "dataset": run["dataset"],
                    "direction": run["direction"],
                    "row_in_result": row_pos,
                    "test_index": test_index,
                    "smiles": smiles,
                    "r2": r2,
                    **feature_cache[smiles],
                }
            )
        checks.append(
            {
                "dataset": run["dataset"],
                "direction": run["direction"],
                "smiles_rows": len(smiles_list),
                "r2_rows": len(r2_rows),
                "min_index": min(i for i, _ in r2_rows),
                "max_index": max(i for i, _ in r2_rows),
                "missing_smiles": missing,
                "out_of_range_index": out_of_range,
            }
        )

    sample_fields = [
        "dataset",
        "direction",
        "row_in_result",
        "test_index",
        "smiles",
        "valid_smiles",
        "scaffold",
        "generic_scaffold",
        "scaffold_type",
        "scaffold_class",
        "ring_count",
        "scaffold_atoms",
        "scaffold_hetero",
        "mol_heavy_atoms",
        "mol_hetero_atoms",
        "r2",
    ]
    write_csv(OUT_DIR / "per_sample_scaffold_r2.csv", per_sample, sample_fields)
    write_csv(
        OUT_DIR / "alignment_checks.csv",
        checks,
        ["dataset", "direction", "smiles_rows", "r2_rows", "min_index", "max_index", "missing_smiles", "out_of_range_index"],
    )

    scaffold_summary = summarise(per_sample, ["dataset", "direction", "scaffold", "generic_scaffold", "scaffold_type"], min_n=10)
    generic_summary = summarise(per_sample, ["dataset", "direction", "generic_scaffold", "scaffold_class"], min_n=20)
    class_summary = summarise(per_sample, ["dataset", "direction", "scaffold_class"], min_n=30)
    type_summary = summarise(per_sample, ["dataset", "direction", "scaffold_type"], min_n=10)
    summary_fields = ["n", "mean_r2", "median_r2", "p25_r2", "p75_r2", "frac_r2_lt_0p5", "frac_r2_ge_0p9"]
    write_csv(
        OUT_DIR / "murcko_scaffold_summary_min10.csv",
        scaffold_summary,
        ["dataset", "direction", "scaffold", "generic_scaffold", "scaffold_type", *summary_fields],
    )
    write_csv(
        OUT_DIR / "scaffold_type_summary.csv",
        type_summary,
        ["dataset", "direction", "scaffold_type", *summary_fields],
    )
    write_csv(
        OUT_DIR / "generic_murcko_summary_min20.csv",
        generic_summary,
        ["dataset", "direction", "generic_scaffold", "scaffold_class", *summary_fields],
    )
    write_csv(
        OUT_DIR / "scaffold_class_summary_min30.csv",
        class_summary,
        ["dataset", "direction", "scaffold_class", *summary_fields],
    )

    top_bottom: list[dict[str, object]] = []
    for dataset in ("QM9S", "QMe14S"):
        for direction in ("IR->Raman", "Raman->IR"):
            rows = [
                r
                for r in scaffold_summary
                if r["dataset"] == dataset
                and r["direction"] == direction
                and r["scaffold"] not in {"missing_smiles", "invalid_smiles"}
            ]
            for rank, row in enumerate(sorted(rows, key=lambda r: float(r["mean_r2"]), reverse=True)[:15], 1):
                top_bottom.append({**row, "rank_type": "top", "rank": rank})
            for rank, row in enumerate(sorted(rows, key=lambda r: float(r["mean_r2"]))[:15], 1):
                top_bottom.append({**row, "rank_type": "bottom", "rank": rank})
    write_csv(
        OUT_DIR / "murcko_scaffold_top_bottom_min10.csv",
        top_bottom,
        ["dataset", "direction", "rank_type", "rank", "scaffold", "generic_scaffold", "scaffold_type", *summary_fields],
    )

    plot_scaffold_class_bottom_polished(
        class_summary,
        OUT_DIR / "scaffold_class_bottom_r2_min30",
    )

    print(f"Wrote outputs to: {OUT_DIR}")
    for row in checks:
        print(row)


if __name__ == "__main__":
    main()
