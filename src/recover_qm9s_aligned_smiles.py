"""Export QM9S identities from its original sequential mapping.txt.

The previous ZIP-based implementation used QMe14S identities for QM9S rows.
Its purported anchors were not independent checks. That interface is removed.
The mapping must follow the original QM9S HDF5 row order.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from build_nist_qm9s_overlap import read_qm9s_mapping


def seed_test_indices(n: int, seed: int):
    train = int(0.7 * n)
    val = int(0.15 * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return perm[train + val:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping-txt", type=Path, required=True)
    ap.add_argument("--n", type=int, default=129817)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out_full", default="data/processed/qm9s_h5order_full_smiles.csv")
    ap.add_argument("--out_test", default="data/processed/qm9s_seed2_aligned_smiles.csv")
    args = ap.parse_args()

    full = read_qm9s_mapping(args.mapping_txt, args.n)
    n_empty = sum(1 for s in full if not s)
    print(f"Read {len(full)} smiles; empty: {n_empty}")

    Path(args.out_full).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_full, "w", encoding="utf-8", newline="\n") as f:
        f.write("smiles\n")
        f.writelines(s + "\n" for s in full)
    print("Wrote full H5-order smiles:", args.out_full)

    test_idx = seed_test_indices(len(full), seed=args.seed)
    Path(args.out_test).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_test, "w", encoding="utf-8", newline="\n") as f:
        f.write("smiles\n")
        f.writelines(full[i] + "\n" for i in test_idx)
    print(f"Wrote seed{args.seed} test smiles ({len(test_idx)} rows):", args.out_test)
    print("First 5 test smiles:", [full[i] for i in test_idx[:5]])


if __name__ == "__main__":
    main()
