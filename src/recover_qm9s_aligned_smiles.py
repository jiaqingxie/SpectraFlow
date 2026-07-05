"""
Recover the H5-row-aligned SMILES for QM9S from IR_broaden.zip.

Ground truth: process.py preserves row order, so
    ir_broaden_processed.h5 row i  ==  IR_broaden/IR_{i:06d}.csv
and the FIRST line of each IR_{i}.csv is the SMILES for that spectrum.
Verified against test_flow's own dump_metadata (5/5 gold anchors).

Outputs:
  - full H5-order SMILES (one per H5 row)
  - seed-2 random_split test subset SMILES (aligned to the saved test spectra)
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import torch
from tqdm import tqdm


GOLD = {
    119359: "CCN(C)C(C=O)C#C",
    124668: "OP(O)OP(O)O",
    123020: "O=C1CN2N=NC=C2O1",
    116113: "OC1=NC(=CN=C1)C#C",
    129614: "CC1=CN2C=CCC2=N1",
}


def seed_test_indices(n: int, seed: int):
    train = int(0.7 * n)
    val = int(0.15 * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return perm[train + val:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip_path", default="/mnt/shared-storage-user/xiejiaqing/data/IR_broaden.zip")
    ap.add_argument("--inner_prefix", default="IR_broaden/IR_")
    ap.add_argument("--n", type=int, default=129817)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out_full", default="/home/xiejiaqing/spectrogen_v2/smiles_lists/qm9s_h5order_full_smiles.csv")
    ap.add_argument("--out_test", default="/home/xiejiaqing/spectrogen_v2/smiles_lists/qm9s_seed2_aligned_smiles.csv")
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip_path)

    def read_smiles(row: int) -> str:
        with z.open(f"{args.inner_prefix}{row:06d}.csv") as f:
            return f.readline().decode("utf-8").strip()

    # sanity check anchors first
    bad = [(r, read_smiles(r), s) for r, s in GOLD.items() if read_smiles(r) != s]
    if bad:
        for r, got, want in bad:
            print(f"ANCHOR MISMATCH row {r}: got {got} want {want}")
        raise SystemExit("Gold anchors do not match; aborting.")
    print("Gold anchors OK (5/5).")

    full = [read_smiles(i) for i in tqdm(range(args.n), desc="Reading SMILES")]
    n_empty = sum(1 for s in full if not s)
    print(f"Read {len(full)} smiles; empty: {n_empty}")

    Path(args.out_full).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_full, "w", encoding="utf-8", newline="\n") as f:
        f.write("smiles\n")
        f.writelines(s + "\n" for s in full)
    print("Wrote full H5-order smiles:", args.out_full)

    test_idx = seed_test_indices(len(full), seed=args.seed)
    with open(args.out_test, "w", encoding="utf-8", newline="\n") as f:
        f.write("smiles\n")
        f.writelines(full[i] + "\n" for i in test_idx)
    print(f"Wrote seed{args.seed} test smiles ({len(test_idx)} rows):", args.out_test)
    print("First 5 test smiles:", [full[i] for i in test_idx[:5]])


if __name__ == "__main__":
    main()
