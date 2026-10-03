"""Audit local paper artifacts without loading a model or running inference.

Checkpoint inspection reads tensor shapes with a restricted metadata unpickler;
tensor storage is never read. --recompute streams saved prediction CSVs to
recalculate metrics. This is artifact analysis, not checkpoint reproduction.
"""
import argparse
import collections
import io
import json
import pickle
import re
import os
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


class MetadataUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        return {"storage_type": str(pid[1]), "device": str(pid[3])}

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if module == "torch" and name.endswith("Storage"):
            return name
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return lambda storage, offset, size, stride, *rest: {"shape": list(size)}
        raise pickle.UnpicklingError(f"Unsupported metadata global {module}.{name}")


def checkpoint_metadata(path):
    with zipfile.ZipFile(path) as z:
        member = next(n for n in z.namelist() if n.endswith("/data.pkl"))
        obj = MetadataUnpickler(io.BytesIO(z.read(member))).load()
    state = obj.get("model_state_dict", {})
    keys = list(state)
    if any("x_embedder.proj.weight" in k for k in keys):
        backbone = "vibradit"
    elif any("patch_embed" in k for k in keys):
        backbone = "dit"
    elif any("velocity_field.encoder1" in k for k in keys):
        backbone = "unet"
    else:
        backbone = "other"
    match = re.search(r"seed(\d+)", path.name)
    return {
        "backbone_from_state_keys": backbone,
        "seed_from_filename_only": int(match[1]) if match else None,
        "payload_keys": list(obj),
        "train_steps": obj.get("train_steps"),
        "shapes": {k: v["shape"] for k, v in state.items() if isinstance(v, dict)},
        "contains_run_config": any(k in obj for k in ("args", "config", "run_config")),
        "contains_split_ids": any(k in obj for k in ("split_ids", "test_indices", "splits")),
    }


def inspect_h5(path):
    with h5py.File(path) as f:
        item = {"path": str(path), "datasets": {k: list(f[k].shape) for k in f if isinstance(f[k], h5py.Dataset)}}
        if "spectra" in f:
            n = len(f["spectra"])
            item.update(n_samples=n, default_random_split_test_n=n-int(.7*n)-int(.15*n))
        if "x_axis" in f:
            x = f["x_axis"][:]
            item["axis_range"] = [float(x.min()), float(x.max())]
            item["axis_increasing"] = bool(np.all(np.diff(x) > 0))
    return item


def saved_metric_inventory(root):
    rows = []
    for p in sorted((root / "results").rglob("*_r2_per_sample.csv")):
        df = pd.read_csv(p)
        if "r2" not in df:
            continue
        n = len(df)
        row = {"path": str(p.relative_to(root)), "n": n,
               "r2_mean": df.r2.mean(), "r2_median": df.r2.median(),
               "r2_molecule_sd": df.r2.std(ddof=1), "r2_negative_fraction": (df.r2 < 0).mean(),
               "r2_nonfinite": int((~np.isfinite(df.r2)).sum()),
               "index_unique": bool(df["index"].is_unique),
               "index_max": int(df["index"].max()) if n else None,
               "evidence_level": "saved_metrics_only"}
        other = p.with_name(p.name.replace("_r2_per_sample", "_metrics_per_sample"))
        if other.exists():
            metrics = pd.read_csv(other)
            row["other_metrics_n"] = len(metrics)
            row["same_metric_indices"] = bool(np.array_equal(df["index"], metrics["index"]))
            for key in ("pearson", "ssim", "psnr", "js_div"):
                if key in metrics:
                    row[key + "_mean"] = metrics[key].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def metric_arrays(y, pred):
    error = pred - y
    yc, pc = y-y.mean(axis=1, keepdims=True), pred-pred.mean(axis=1, keepdims=True)
    ss = np.square(yc).sum(axis=1)
    sse = np.square(error).sum(axis=1)
    ranges = np.ptp(y, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1-sse/ss
        r2 = np.where(ss == 0, np.where(sse == 0, 1., 0.), r2)
        pearson = (yc*pc).sum(axis=1) / np.sqrt(ss*np.square(pc).sum(axis=1))
        mae = np.abs(error).mean(axis=1)
        nmae = mae/(ranges+1e-8)
    return {"r2": r2, "pearson": pearson, "mae": mae,
            "rmse": np.sqrt(np.square(error).mean(axis=1)), "normalized_mae": nmae}


def legacy_order(side):
    """Indices of physical spectral points in the legacy heatmap flattening."""
    patch = max(p for p in (10, 8, 5, 4, 2, 1) if side % p == 0)
    count = side // patch
    patches = np.arange(side * side).reshape(-1, patch, patch)
    return np.concatenate([
        np.concatenate(patches[i * count:(i + 1) * count], axis=1)
        for i in range(count)
    ], axis=0).ravel()


def audit_qme_exclusions(root, output):
    """Map missing saved indices to the original rows; no model evaluation."""
    import torch
    torch.set_num_threads(1)
    data = Path(os.environ.get('SPECTRAFLOW_RAW_DATA_ROOT', str(root.parent/'data')))/'QMe14S/processed'
    with h5py.File(data/"ir_broaden_processed.h5") as irf, \
            h5py.File(data/"raman_broaden_processed.h5") as rf:
        n = len(irf["spectra"])
        if n != len(rf["spectra"]):
            raise ValueError("QMe14S source/target sizes differ")
        permutation = torch.randperm(n, generator=torch.Generator().manual_seed(2)).numpy()
        tests = permutation[int(.7*n)+int(.15*n):]
        missing_sets = []
        for direction in ("ir2raman", "raman2ir"):
            path = root/"results/verify_fig1_seed2"/f"qme14s_flow_{direction}_vibradit_seed2"/f"flow_{direction}_r2_per_sample.csv"
            saved = pd.read_csv(path)
            missing_sets.append(set(range(len(tests)))-set(saved["index"]))
        missing = np.array(sorted(missing_sets[0] | missing_sets[1]))
        rows = tests[missing]
        order = np.argsort(rows)
        ir = irf["spectra"][rows[order]][np.argsort(order)]
        raman = rf["spectra"][rows[order]][np.argsort(order)]
    details = pd.DataFrame(dict(test_index=missing, h5_row_index=rows,
                                ir_all_zero=(ir == 0).all(axis=1), raman_all_zero=(raman == 0).all(axis=1),
                                ir_range=np.ptp(ir, axis=1), raman_range=np.ptp(raman, axis=1),
                                excluded_ir2raman=[i in missing_sets[0] for i in missing],
                                excluded_raman2ir=[i in missing_sets[1] for i in missing]))
    details.to_csv(output/"qme14s_excluded_test_rows.csv", index=False)
    summary = dict(dataset="QMe14S", seed=2, test_n=len(tests), omitted_n=len(details),
                   identical_omitted_test_indices_both_directions=missing_sets[0] == missing_sets[1],
                   omitted_ir_all_zero_n=int(details.ir_all_zero.sum()),
                   omitted_raman_all_zero_n=int(details.raman_all_zero.sum()),
                   omitted_ir_nonfinite_n=int((~np.isfinite(ir)).any(axis=1).sum()),
                   omitted_raman_nonfinite_n=int((~np.isfinite(raman)).any(axis=1).sum()),
                   row_manifest="qme14s_excluded_test_rows.csv", evidence_level="reference_data_and_saved_index_audit")
    (output/"qme14s_zero_spectrum_exclusions.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def recompute_saved(root, relative_dir, output):
    directory = root / relative_dir
    summaries = []
    for pp in sorted(directory.glob("*_preds.csv")):
        prefix = pp.name.removesuffix("_preds.csv")
        tp = directory / (prefix + "_targets.csv")
        if not tp.exists():
            continue
        rp = directory / (prefix + "_r2_per_sample.csv")
        old = pd.read_csv(rp).set_index("index")["r2"] if rp.exists() else None
        parts = []
        piter = pd.read_csv(pp, chunksize=128)
        titer = pd.read_csv(tp, chunksize=128)
        import itertools
        for pdf, tdf in itertools.zip_longest(piter, titer):
            if pdf is None or tdf is None or not np.array_equal(pdf["index"], tdf["index"]):
                raise ValueError(f"Prediction/target row mismatch: {pp}")
            if list(pdf.columns) != list(tdf.columns):
                raise ValueError(f"Prediction/target columns differ: {pp}")
            row = pd.DataFrame(metric_arrays(tdf.iloc[:, 1:].to_numpy(), pdf.iloc[:, 1:].to_numpy()))
            row.insert(0, "index", pdf["index"].to_numpy())
            if old is not None:
                row["saved_r2_difference"] = row.r2.to_numpy()-old.reindex(row["index"]).to_numpy()
            parts.append(row)
        df = pd.concat(parts, ignore_index=True)
        name = str(relative_dir).replace("/", "__") + "__" + prefix
        df.to_csv(output / (name + "_recomputed.csv"), index=False)
        summary = {"result_dir": str(relative_dir), "prefix": prefix, "n": len(df),
                   "evidence_level": "recomputed_from_saved_predictions"}
        for key in ("r2", "pearson", "mae", "rmse", "normalized_mae"):
            summary[key + "_mean"] = df[key].mean()
        if "saved_r2_difference" in df:
            summary["saved_r2_max_abs_difference"] = df.saved_r2_difference.abs().max()
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("results/reproducibility_audit"))
    parser.add_argument("--extra-data", type=Path, action="append", default=[])
    parser.add_argument("--recompute", type=Path, action="append", default=[])
    parser.add_argument("--only-qme-exclusions", action="store_true",
                        help="Requires PyTorch only to reconstruct the fixed split; runs no model")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.only_qme_exclusions:
        audit_qme_exclusions(root, output)
        return
    records = []
    for path in sorted((root/"checkpoints").rglob("*.pt")):
        item = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size}
        try:
            item.update(checkpoint_metadata(path))
        except (OSError, ValueError, pickle.UnpicklingError, StopIteration) as exc:
            item["metadata_error"] = str(exc)
        records.append(item)
    (output/"checkpoint_inventory.json").write_text(json.dumps(records, indent=2))
    h5s = sorted((root/"data").rglob("*.h5")) + sorted((root/"datasets").rglob("*.h5"))
    for base in args.extra_data:
        h5s += sorted(base.rglob("*.h5"))
    (output/"data_inventory.json").write_text(json.dumps([inspect_h5(p) for p in h5s], indent=2))
    metrics = saved_metric_inventory(root)
    metrics.to_csv(output/"saved_metrics_inventory.csv", index=False)
    print(f"Inspected {len(records)} checkpoints, {len(h5s)} HDF5 files, {len(metrics)} saved metric files. No inference.", flush=True)
    summaries = []
    for directory in args.recompute:
        summaries.extend(recompute_saved(root, directory, output))
        pd.DataFrame(summaries).to_csv(output/"recomputed_summary.csv", index=False)


if __name__ == "__main__":
    main()
