"""Inspect whether a QM9S ZIP molecule and processed H5 row are aligned."""
from __future__ import annotations

import argparse
import csv
import zipfile

import h5py
import numpy as np
from scipy.interpolate import interp1d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--source-csv")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--prefix", default="IR_broaden/IR_")
    args = parser.parse_args()

    member = f"{args.prefix}{args.index:06d}.csv"
    with zipfile.ZipFile(args.zip) as archive:
        text = archive.read(member).decode("utf-8")
    print(f"ZIP member: {member}")
    print("First 15 lines:")
    print("\n".join(text.splitlines()[:15]))
    print("Last 5 lines:")
    print("\n".join(text.splitlines()[-5:]))

    with h5py.File(args.h5, "r") as handle:
        print("H5 keys:", list(handle.keys()))
        spectrum = np.asarray(handle["spectra"][args.index])
        axis = np.asarray(handle["x_axis"])
    raw_spectrum = np.fromstring(text.strip().splitlines()[-1], sep=",")
    print(
        "H5 row:",
        args.index,
        "shape:",
        spectrum.shape,
        "range:",
        (float(spectrum.min()), float(spectrum.max())),
        "axis:",
        (float(axis[0]), float(axis[-1])),
    )
    print(
        "ZIP spectrum:",
        raw_spectrum.shape,
        "range:",
        (float(raw_spectrum.min()), float(raw_spectrum.max())),
    )
    if raw_spectrum.shape == spectrum.shape:
        print(
            "ZIP/H5 Pearson:",
            float(np.corrcoef(raw_spectrum, spectrum)[0, 1]),
            "max abs error:",
            float(np.max(np.abs(raw_spectrum - spectrum))),
        )
    else:
        source_grid = np.linspace(0.0, 1.0, raw_spectrum.size)
        target_grid = np.linspace(0.0, 1.0, spectrum.size)
        interpolated = interp1d(source_grid, raw_spectrum)(target_grid)
        print(
            "Interpolated ZIP/H5 Pearson:",
            float(np.corrcoef(interpolated, spectrum)[0, 1]),
            "ranges:",
            (
                (float(interpolated.min()), float(interpolated.max())),
                (float(spectrum.min()), float(spectrum.max())),
            ),
        )

    if args.source_csv:
        with open(args.source_csv, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            source_row = next(
                row for row_number, row in enumerate(reader)
                if row_number == args.index + 1
            )
        source_values = np.asarray(source_row[1:], dtype=np.float64)
        source_for_h5 = source_values
        if source_values.shape != spectrum.shape:
            source_for_h5 = interp1d(
                np.linspace(0.0, 1.0, source_values.size),
                source_values,
            )(np.linspace(0.0, 1.0, spectrum.size))
        print(
            "Source CSV row:",
            "id:",
            source_row[0],
            "shape:",
            source_values.shape,
            "range:",
            (float(source_values.min()), float(source_values.max())),
            "CSV/H5 Pearson:",
            float(np.corrcoef(source_for_h5, spectrum)[0, 1]),
        )


if __name__ == "__main__":
    main()
