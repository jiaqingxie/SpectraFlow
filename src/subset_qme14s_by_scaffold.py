"""
按 Murcko 骨架对 QMe14S 单文件光谱做分层抽样，输出子集 ID 列表，供后续 filter_h5_by_ids 或重新跑 process_qme14s。

依据 README：每个 IR/Raman 文件首行为 SMILES；若首行无法解析为 SMILES，会记入 failed 或归入单独桶。
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

from process_qme14s import build_index


def read_first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s:
                return s
    return ""


def murcko_scaffold_key(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaf = GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return "__empty_scaffold__"
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None


def round_robin_sample(
    scaffold_to_ids: dict[str, list[str]],
    subset_size: int,
    max_per_scaffold: int | None,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for sk, ids in scaffold_to_ids.items():
        ids = ids.copy()
        rng.shuffle(ids)
        if max_per_scaffold is not None:
            ids = ids[:max_per_scaffold]
        if ids:
            groups[sk] = ids

    scaffold_keys = list(groups.keys())
    rng.shuffle(scaffold_keys)

    selected: list[str] = []
    ptrs = {k: 0 for k in scaffold_keys}

    while len(selected) < subset_size:
        progressed = False
        for k in scaffold_keys:
            if len(selected) >= subset_size:
                break
            p = ptrs[k]
            if p < len(groups[k]):
                selected.append(groups[k][p])
                ptrs[k] += 1
                progressed = True
        if not progressed:
            break

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified subset of QMe14S by Murcko scaffold")
    parser.add_argument("--ir_dir", type=str, required=True, help="IR_broaden directory (SMILES read from each file line 1)")
    parser.add_argument("--raman_dir", type=str, default=None, help="Optional: verify pairing with Raman ids")
    parser.add_argument("--subset_size", type=int, required=True, help="Target number of paired molecules")
    parser.add_argument(
        "--max_per_scaffold",
        type=int,
        default=None,
        help="Cap molecules per scaffold before round-robin (reduces huge scaffolds dominating)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_ids", type=str, required=True, help="Output text file: one sample id per line")
    parser.add_argument(
        "--output_report",
        type=str,
        default=None,
        help="Optional JSON: scaffold counts, failed SMILES count, etc.",
    )
    args = parser.parse_args()

    ir_dir = Path(args.ir_dir)
    ir_map = build_index(ir_dir)

    if args.raman_dir:
        raman_map = build_index(Path(args.raman_dir))
        common = set(ir_map) & set(raman_map)
        ids_to_use = sorted(common, key=lambda x: int(x))
        print(f"Paired IR+Raman ids: {len(ids_to_use)}")
    else:
        ids_to_use = sorted(ir_map.keys(), key=lambda x: int(x))
        print(f"IR-only ids: {len(ids_to_use)}")

    scaffold_to_ids: dict[str, list[str]] = defaultdict(list)
    failed_smiles = 0

    for sid in tqdm(ids_to_use, desc="Read SMILES + scaffold"):
        path = ir_map[sid]
        first = read_first_line(path)
        key = murcko_scaffold_key(first)
        if key is None:
            failed_smiles += 1
            scaffold_to_ids["__invalid_smiles__"].append(sid)
            continue
        scaffold_to_ids[key].append(sid)

    n_scaffolds = len([k for k in scaffold_to_ids if not k.startswith("__")])
    print(f"Unique scaffolds (excl. error buckets): {n_scaffolds}")
    print(f"Invalid SMILES (first line): {failed_smiles}")

    selected = round_robin_sample(
        dict(scaffold_to_ids),
        subset_size=args.subset_size,
        max_per_scaffold=args.max_per_scaffold,
        seed=args.seed,
    )
    print(f"Selected: {len(selected)} / requested {args.subset_size}")

    out_path = Path(args.output_ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")

    if args.output_report:
        report = {
            "subset_size_requested": args.subset_size,
            "subset_size_actual": len(selected),
            "n_scaffolds": n_scaffolds,
            "failed_smiles_first_line": failed_smiles,
            "bucket_sizes": {k: len(v) for k, v in sorted(scaffold_to_ids.items(), key=lambda x: -len(x[1]))[:50]},
        }
        Path(args.output_report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote report: {args.output_report}")


if __name__ == "__main__":
    main()
