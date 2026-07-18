"""Stream large prediction/target CSVs and compute global pointwise metrics."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--chunksize", type=int, default=64)
    args = parser.parse_args()

    totals = {
        "n": 0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_x2": 0.0,
        "sum_y2": 0.0,
        "sum_xy": 0.0,
        "sse": 0.0,
        "sae": 0.0,
    }
    pred_chunks = pd.read_csv(args.predictions, chunksize=args.chunksize)
    target_chunks = pd.read_csv(args.targets, chunksize=args.chunksize)
    chunk_count = 0
    for pred_frame, target_frame in zip(
        pred_chunks, target_chunks, strict=True
    ):
        pred = pred_frame.iloc[:, 1:].to_numpy(dtype=np.float64)
        target = target_frame.iloc[:, 1:].to_numpy(dtype=np.float64)
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
        totals["n"] += pred.size
        totals["sum_x"] += pred.sum()
        totals["sum_y"] += target.sum()
        totals["sum_x2"] += np.square(pred).sum()
        totals["sum_y2"] += np.square(target).sum()
        totals["sum_xy"] += (pred * target).sum()
        error = pred - target
        totals["sse"] += np.square(error).sum()
        totals["sae"] += np.abs(error).sum()
        chunk_count += 1
        if chunk_count % 100 == 0:
            print(f"Processed {chunk_count} chunks", flush=True)

    n = totals["n"]
    target_ss = totals["sum_y2"] - totals["sum_y"] ** 2 / n
    pred_ss = totals["sum_x2"] - totals["sum_x"] ** 2 / n
    covariance = totals["sum_xy"] - totals["sum_x"] * totals["sum_y"] / n
    print(f"Points: {n}")
    print(f"Global R2: {1.0 - totals['sse'] / target_ss:.9f}")
    print(f"Global Pearson: {covariance / np.sqrt(pred_ss * target_ss):.9f}")
    print(f"Global RMSE: {np.sqrt(totals['sse'] / n):.9e}")
    print(f"Global MAE: {totals['sae'] / n:.9e}")


if __name__ == "__main__":
    main()
