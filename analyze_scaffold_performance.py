import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


BASE = Path(r"D:/VibraFlow-NCS/结果文件/图1")
OUT_DIR = BASE / "scaffold_analysis_seed2"


DATASETS = [
    {
        "dataset": "QMe14S",
        "smiles": BASE / "QMe14S结果" / "qme14s_test_smiles_seed2.txt",
        "directions": [
            {
                "direction": "IR->Raman",
                "result_dir": BASE / "QMe14S结果" / "qme14s_flow_ir2raman",
                "prefix": "flow_ir2raman",
            },
            {
                "direction": "Raman->IR",
                "result_dir": BASE / "QMe14S结果" / "qme14s_flow_raman2ir",
                "prefix": "flow_raman2ir",
            },
        ],
    },
    {
        "dataset": "QM9S",
        "smiles": BASE / "QM9S结果" / "smiles" / "qm9s_test_smiles_seed2.txt",
        "directions": [
            {
                "direction": "IR->Raman",
                "result_dir": BASE / "QM9S结果" / "qm9s_flow_ir2raman_vibradit",
                "prefix": "flow_ir2raman",
            },
            {
                "direction": "Raman->IR",
                "result_dir": BASE / "QM9S结果" / "qm9s_flow_raman2ir_vibradit",
                "prefix": "flow_raman2ir",
            },
        ],
    },
]


SMARTS = {
    "carbonyl": "[CX3]=[OX1]",
    "carboxyl": "C(=O)[OX2H1,OX1-]",
    "ester": "C(=O)O[#6]",
    "amide": "C(=O)N",
    "nitrile": "C#N",
    "alkyne": "C#C",
    "alkene": "C=C",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "alcohol": "[OX2H][CX4]",
    "ether": "[OD2]([#6])[#6]",
    "halogen": "[F,Cl,Br,I]",
}
SMARTS_MOLS = {name: Chem.MolFromSmarts(pat) for name, pat in SMARTS.items()}


def read_smiles(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].strip().lower() == "smiles":
        lines = lines[1:]
    return [line.strip() for line in lines]


def read_csv_dicts(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value):
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def classify_ring_type(mol):
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    if not atom_rings:
        return "acyclic"

    aromatic_rings = 0
    hetero_rings = 0
    carbocyclic_rings = 0
    for ring in atom_rings:
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        is_aromatic = all(atom.GetIsAromatic() for atom in atoms)
        has_hetero = any(atom.GetAtomicNum() not in (6, 1) for atom in atoms)
        aromatic_rings += int(is_aromatic)
        hetero_rings += int(has_hetero)
        carbocyclic_rings += int(not has_hetero)

    if aromatic_rings and hetero_rings:
        return "aromatic_heterocycle"
    if aromatic_rings:
        return "aromatic_carbocycle"
    if hetero_rings:
        return "aliphatic_heterocycle"
    if carbocyclic_rings:
        return "aliphatic_carbocycle"
    return "ring_other"


def primary_functional_group(mol):
    hits = [name for name, query in SMARTS_MOLS.items() if query is not None and mol.HasSubstructMatch(query)]
    return "+".join(hits[:3]) if hits else "none"


def scaffold_features(smiles):
    if not smiles:
        return {
            "valid_smiles": False,
            "scaffold": "missing_smiles",
            "generic_scaffold": "missing_smiles",
            "scaffold_type": "missing_smiles",
            "functional_group": "missing_smiles",
            "ring_count": 0,
            "heavy_atoms": 0,
        }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "valid_smiles": False,
            "scaffold": "invalid_smiles",
            "generic_scaffold": "invalid_smiles",
            "scaffold_type": "invalid_smiles",
            "functional_group": "invalid_smiles",
            "ring_count": 0,
            "heavy_atoms": 0,
        }

    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold = Chem.MolToSmiles(scaffold_mol, isomericSmiles=False) if scaffold_mol.GetNumAtoms() else "acyclic"
    generic = "acyclic"
    if scaffold != "acyclic":
        generic_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
        generic = Chem.MolToSmiles(generic_mol, isomericSmiles=False)

    return {
        "valid_smiles": True,
        "scaffold": scaffold,
        "generic_scaffold": generic,
        "scaffold_type": classify_ring_type(mol),
        "functional_group": primary_functional_group(mol),
        "ring_count": int(mol.GetRingInfo().NumRings()),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
    }


def summarise(rows, group_cols, min_n=1):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[col] for col in group_cols)
        grouped[key].append(row)

    out = []
    for key, vals in grouped.items():
        r2 = np.array([v["r2"] for v in vals if not math.isnan(v["r2"])], dtype=float)
        if len(r2) < min_n:
            continue
        pearson = np.array([v["pearson"] for v in vals if not math.isnan(v["pearson"])], dtype=float)
        ssim = np.array([v["ssim"] for v in vals if not math.isnan(v["ssim"])], dtype=float)
        js_div = np.array([v["js_div"] for v in vals if not math.isnan(v["js_div"])], dtype=float)
        item = {col: val for col, val in zip(group_cols, key)}
        item.update(
            {
                "n": int(len(r2)),
                "mean_r2": float(np.mean(r2)),
                "median_r2": float(np.median(r2)),
                "p25_r2": float(np.percentile(r2, 25)),
                "p75_r2": float(np.percentile(r2, 75)),
                "frac_r2_ge_0p9": float(np.mean(r2 >= 0.9)),
                "frac_r2_lt_0p5": float(np.mean(r2 < 0.5)),
                "mean_pearson": float(np.mean(pearson)) if len(pearson) else math.nan,
                "mean_ssim": float(np.mean(ssim)) if len(ssim) else math.nan,
                "mean_js_div": float(np.mean(js_div)) if len(js_div) else math.nan,
            }
        )
        out.append(item)
    out.sort(key=lambda x: (x["dataset"], x["direction"], -x["mean_r2"]))
    return out


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_sample = []
    checks = []
    feature_cache = {}

    for spec in DATASETS:
        smiles_list = read_smiles(spec["smiles"])
        for direction in spec["directions"]:
            r2_rows = read_csv_dicts(direction["result_dir"] / f"{direction['prefix']}_r2_per_sample.csv")
            metric_rows = read_csv_dicts(direction["result_dir"] / f"{direction['prefix']}_metrics_per_sample.csv")
            n = min(len(smiles_list), len(r2_rows), len(metric_rows))
            checks.append(
                {
                    "dataset": spec["dataset"],
                    "direction": direction["direction"],
                    "smiles_rows": len(smiles_list),
                    "r2_rows": len(r2_rows),
                    "metric_rows": len(metric_rows),
                    "used_rows": n,
                    "status": "aligned" if len(smiles_list) == len(r2_rows) == len(metric_rows) else "row_count_mismatch",
                }
            )

            for i in range(n):
                smiles = smiles_list[i]
                if smiles not in feature_cache:
                    feature_cache[smiles] = scaffold_features(smiles)
                feat = feature_cache[smiles]
                metric = metric_rows[i]
                per_sample.append(
                    {
                        "dataset": spec["dataset"],
                        "direction": direction["direction"],
                        "sample_index": i,
                        "smiles": smiles,
                        "r2": as_float(r2_rows[i].get("r2")),
                        "pearson": as_float(metric.get("pearson")),
                        "ssim": as_float(metric.get("ssim")),
                        "js_div": as_float(metric.get("js_div")),
                        **feat,
                    }
                )

    type_summary = summarise(per_sample, ["dataset", "direction", "scaffold_type"], min_n=10)
    scaffold_summary = summarise(per_sample, ["dataset", "direction", "scaffold", "generic_scaffold", "scaffold_type"], min_n=10)
    func_summary = summarise(per_sample, ["dataset", "direction", "functional_group"], min_n=10)

    write_csv(
        OUT_DIR / "per_sample_scaffold_metrics.csv",
        per_sample,
        [
            "dataset",
            "direction",
            "sample_index",
            "smiles",
            "valid_smiles",
            "scaffold",
            "generic_scaffold",
            "scaffold_type",
            "functional_group",
            "ring_count",
            "heavy_atoms",
            "r2",
            "pearson",
            "ssim",
            "js_div",
        ],
    )
    write_csv(OUT_DIR / "alignment_checks.csv", checks, ["dataset", "direction", "smiles_rows", "r2_rows", "metric_rows", "used_rows", "status"])
    summary_fields = [
        "dataset",
        "direction",
        "scaffold_type",
        "n",
        "mean_r2",
        "median_r2",
        "p25_r2",
        "p75_r2",
        "frac_r2_ge_0p9",
        "frac_r2_lt_0p5",
        "mean_pearson",
        "mean_ssim",
        "mean_js_div",
    ]
    write_csv(OUT_DIR / "scaffold_type_summary.csv", type_summary, summary_fields)
    write_csv(
        OUT_DIR / "scaffold_summary_min10.csv",
        scaffold_summary,
        ["dataset", "direction", "scaffold", "generic_scaffold", "scaffold_type", *summary_fields[3:]],
    )
    write_csv(
        OUT_DIR / "functional_group_summary.csv",
        func_summary,
        ["dataset", "direction", "functional_group", *summary_fields[3:]],
    )

    report = {
        "alignment_checks": checks,
        "best_scaffold_types": sorted(type_summary, key=lambda x: (-x["mean_r2"], -x["n"]))[:12],
        "worst_scaffold_types": sorted(type_summary, key=lambda x: (x["mean_r2"], -x["n"]))[:12],
        "best_scaffolds_min10": sorted(scaffold_summary, key=lambda x: (-x["mean_r2"], -x["n"]))[:20],
        "worst_scaffolds_min10": sorted(scaffold_summary, key=lambda x: (x["mean_r2"], -x["n"]))[:20],
        "best_functional_groups": sorted(func_summary, key=lambda x: (-x["mean_r2"], -x["n"]))[:12],
        "worst_functional_groups": sorted(func_summary, key=lambda x: (x["mean_r2"], -x["n"]))[:12],
    }
    with (OUT_DIR / "scaffold_analysis_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote analysis to: {OUT_DIR}")
    for check in checks:
        print(check)


if __name__ == "__main__":
    main()
