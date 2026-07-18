"""Retrieval-based evaluation for Open Specy cross-modal predictions.

Open Specy does not provide reliable sample-wise IR/Raman pairing, so a
sample-wise regression metric (R^2) on identity-disjoint targets is not a fair
measure. Instead we evaluate whether a generated Raman spectrum retrieves the
correct material identity from a reference bank of real Raman medoids, and we
report the paired Pearson against the matched medoid. A raw IR-as-query control
quantifies how much the cross-modal generator improves retrieval over simply
comparing the input IR against the Raman bank.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_matrix(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    value_columns = [column for column in frame.columns if column != "index"]
    return frame[value_columns].to_numpy(dtype=np.float64)


def zscore(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(norm, 1e-8)


def retrieval_metrics(query: np.ndarray, bank: np.ndarray) -> dict[str, float]:
    query_z = zscore(query)
    bank_z = zscore(bank)
    similarity = query_z @ bank_z.T
    order = np.argsort(-similarity, axis=1)
    truth = np.arange(query.shape[0])
    ranks = np.argmax(order == truth[:, None], axis=1)

    paired_pearson = np.array(
        [np.corrcoef(query[i], bank[i])[0, 1] for i in range(query.shape[0])]
    )

    return {
        "top1": float(np.mean(ranks == 0)),
        "top5": float(np.mean(ranks < 5)),
        "mrr": float(np.mean(1.0 / (ranks + 1))),
        "median_rank": float(np.median(ranks + 1)),
        "paired_pearson": float(np.nanmean(paired_pearson)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="flow_ir2raman")
    args = parser.parse_args()

    preds = load_matrix(args.result_dir / f"{args.prefix}_preds.csv")
    targets = load_matrix(args.result_dir / f"{args.prefix}_targets.csv")
    sources = load_matrix(args.result_dir / f"{args.prefix}_sources.csv")

    generated = retrieval_metrics(preds, targets)
    ir_control = retrieval_metrics(sources, targets)

    summary = pd.DataFrame(
        [
            {"query": "generated_raman", **generated},
            {"query": "raw_ir_control", **ir_control},
        ]
    )
    out_path = args.result_dir / f"{args.prefix}_retrieval_metrics.csv"
    summary.to_csv(out_path, index=False)

    bank_size = targets.shape[0]
    print(f"Reference bank size (Raman medoids): {bank_size}")
    print(f"Random-chance Top-1: {1.0 / bank_size:.4f}")
    print(summary.to_string(index=False))
    print(f"\nSaved retrieval metrics to: {out_path}")


if __name__ == "__main__":
    main()
