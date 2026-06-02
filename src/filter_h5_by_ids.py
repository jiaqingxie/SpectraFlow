"""
根据 sample id 列表从已生成的 ir/raman processed HDF5 中切出子集，行顺序与 id 列表一致。

需与 process_qme14s 使用相同的配对顺序：common_ids = sorted(set(ir)&set(raman), key=int)，
本脚本用 --ir_dir / --raman_dir 重新计算该顺序以建立 id -> row_index。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from process_qme14s import build_index


def common_id_order(ir_dir: Path, raman_dir: Path) -> list[str]:
    ir_map = build_index(ir_dir)
    raman_map = build_index(raman_dir)
    common = sorted(set(ir_map) & set(raman_map), key=lambda x: int(x))
    return common


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter processed H5 by QMe14S sample id list")
    parser.add_argument("--ir_dir", type=str, required=True, help="Original IR_broaden directory (for id order)")
    parser.add_argument("--raman_dir", type=str, required=True, help="Original Raman_broaden directory")
    parser.add_argument("--subset_ids", type=str, required=True, help="Text file: one id per line")
    parser.add_argument("--ir_h5_in", type=str, required=True, help="Input ir_broaden_processed.h5")
    parser.add_argument("--raman_h5_in", type=str, required=True, help="Input raman_broaden_processed.h5")
    parser.add_argument("--ir_h5_out", type=str, required=True)
    parser.add_argument("--raman_h5_out", type=str, required=True)
    args = parser.parse_args()

    ir_dir = Path(args.ir_dir)
    raman_dir = Path(args.raman_dir)
    order = common_id_order(ir_dir, raman_dir)
    id_to_row = {sid: i for i, sid in enumerate(order)}

    subset_lines = Path(args.subset_ids).read_text(encoding="utf-8").strip().splitlines()
    subset_ids = [ln.strip() for ln in subset_lines if ln.strip()]

    indices: list[int] = []
    missing: list[str] = []
    for sid in subset_ids:
        if sid in id_to_row:
            indices.append(id_to_row[sid])
        else:
            missing.append(sid)

    if missing:
        print(f"Warning: {len(missing)} ids not in paired full set (skipped). Example: {missing[:5]}")
    idx = np.array(indices, dtype=np.int64)
    print(f"Keeping {len(idx)} / {len(subset_ids)} requested rows")

    def slice_h5(src: str, dst: str) -> None:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
            spectra = fin["spectra"][idx]
            x_axis = fin["x_axis"][:]
            fout.create_dataset("spectra", data=spectra, compression="gzip", compression_opts=4)
            fout.create_dataset("x_axis", data=x_axis)
            fout.attrs["n_samples"] = len(spectra)
            fout.attrs["spectrum_length"] = spectra.shape[1]
            if "physical_params" in fin:
                fout.create_dataset(
                    "physical_params",
                    data=fin["physical_params"][idx],
                    compression="gzip",
                    compression_opts=4,
                )

    slice_h5(args.ir_h5_in, args.ir_h5_out)
    slice_h5(args.raman_h5_in, args.raman_h5_out)
    print(f"Wrote {args.ir_h5_out}")
    print(f"Wrote {args.raman_h5_out}")


if __name__ == "__main__":
    main()
