"""Re-evaluate paper Flow checkpoints from HDF5, recording every test molecule.

Defaults to CUDA and fails if unavailable. CPU requires explicit --device cpu.
Uses the legacy patch permutation to reproduce existing weights, rather than
silently changing preprocessing. Run on an allocated compute worker.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from audit_paper_reproducibility import checkpoint_metadata, legacy_order, metric_arrays
from model_flow import ConditionalFlowMatching


ROOT = Path(__file__).resolve().parents[1]
QME = Path(os.environ.get('SPECTRAFLOW_RAW_DATA_ROOT', str(ROOT.parent/'data'))) / 'QMe14S/processed'
TASKS = []
for dataset, data, stem, checkpoint_folder, size in [
    ("qm9s", ROOT/"data/processed", "{mode}_broaden_processed", "qm9s", 60),
    ("qme14s", QME, "{mode}_broaden_processed", "qme14s", 60),
    ("vibench_full", ROOT/"data/processed", "vibench_test_full_{mode}_processed", "vibench", 32),
]:
    for source, target in [("ir", "raman"), ("raman", "ir")]:
        TASKS.append(dict(name=f"{dataset}_{source}2{target}_seed2", dataset=dataset,
                          source=str(data/(stem.format(mode=source)+".h5")),
                          target=str(data/(stem.format(mode=target)+".h5")),
                          checkpoint=str(ROOT/"checkpoints"/checkpoint_folder/f"flow_{source}2{target}_vibradit_best_seed2.pt"),
                          source_mode=source, target_mode=target, side=size, seed=2, split="random_test",
                          suite="id", model_type="flow", preserve_order=False))
for domain in ("qm9", "mols", "pahs", "zinc15", "geom", "peptide", "peptide_mod", "nist_ir"):
    for source, target in [("ir", "raman"), ("raman", "ir")]:
        for seed in (0, 1):
            TASKS.append(dict(name=f"{domain}_ood_{source}2{target}_seed{seed}", dataset=domain,
                              source=str(ROOT/"data/processed"/f"{domain}_test_{source}_processed.h5"),
                              target=str(ROOT/"data/processed"/f"{domain}_test_{target}_processed.h5"),
                              checkpoint=str(ROOT/"checkpoints/vibench_ood"/f"flow_{source}2{target}_vibradit_best_seed{seed}.pt"),
                              source_mode=source, target_mode=target, side=32, seed=seed,
                              split="random_test" if domain == "qm9" else "full", suite="ood",
                              model_type="flow", preserve_order=False))
for source, target in [("ir", "raman"), ("raman", "ir")]:
    TASKS.append(dict(name=f"qm9s_direct_{source}2{target}_seed2", dataset="qm9s",
                      source=str(ROOT/"data/processed"/f"{source}_broaden_processed.h5"),
                      target=str(ROOT/"data/processed"/f"{target}_broaden_processed.h5"),
                      checkpoint=str(ROOT/"checkpoints/qm9s_direct"/f"direct_{source}2{target}_vibradit_best_seed2.pt"),
                      source_mode=source, target_mode=target, side=60, seed=2, split="random_test",
                      suite="direct", model_type="direct", preserve_order=False))
TASKS.append(dict(name="rruff_external_ir2raman_seed42", dataset="rruff_external",
                  source=str(ROOT/"datasets/openspecy_rruff_576/test_ir.h5"),
                  target=str(ROOT/"datasets/openspecy_rruff_576/test_raman.h5"),
                  checkpoint=str(ROOT/"checkpoints/openspecy_rruff_576_ordered_p1/flow_ir2raman_vibradit_best_seed42.pt"),
                  source_mode="ir", target_mode="raman", side=24, seed=42, split="full",
                  suite="experimental", model_type="flow", preserve_order=True))
TASKS.append(dict(name="nist_corrected_available_ir2raman_seed2", dataset="nist_corrected_available",
                  source=str(ROOT/"reproduction_audit/experimental/corrected_nist_available_ir.h5"),
                  target=str(ROOT/"reproduction_audit/experimental/corrected_nist_available_raman.h5"),
                  checkpoint=str(ROOT/"checkpoints/qm9s/flow_ir2raman_vibradit_best_seed2.pt"),
                  source_mode="ir", target_mode="raman", side=60, seed=2, split="full",
                  suite="experimental", model_type="flow", preserve_order=False,
                  source_axis_assumption="ascending 399-4000 cm^-1, from manuscript protocol",
                  identity_manifest="reproduction_audit/experimental/corrected_nist_available_pairs.csv"))


def read_rows(dataset, indices):
    order = np.argsort(indices)
    return dataset[np.asarray(indices)[order]][np.argsort(order)]


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_archived_anchors(task, output):
    if task["suite"] != "id":
        return {"status": "no_principal_saved_prediction_mapping"}
    direction = task["source_mode"]+"2"+task["target_mode"]
    dataset = task["dataset"]
    if dataset == "vibench_full":
        directory = ROOT/"results"/f"vibench_full_flow_{direction}"
    else:
        directory = ROOT/"results/verify_fig1_seed2"/f"{dataset}_flow_{direction}_vibradit_seed2"
    arrays = np.load(output/"first_batch_predictions.npz")
    needed = set(range(len(arrays["target"])))
    found = {}
    for chunk in pd.read_csv(directory/f"flow_{direction}_preds.csv", chunksize=128):
        for _, row in chunk[chunk["index"].isin(needed)].iterrows():
            found[int(row["index"])] = row.iloc[1:].to_numpy(dtype=float)
        if set(found) == needed:
            break
    common = sorted(found)
    if not common:
        return {"status": "no_archived_anchor_indices"}
    previous = np.stack([found[index] for index in common])
    current = arrays["prediction"][common]
    scales = np.ptp(arrays["target"][common], axis=1, keepdims=True)+1e-8
    max_normalized_difference = float(np.max(np.abs(current-previous)/scales))
    return dict(status="matched" if max_normalized_difference < 1e-4 else "differs",
                archived_prediction_dir=str(directory.relative_to(ROOT)), n_anchors=len(common),
                max_abs_native_prediction_difference=float(np.max(np.abs(current-previous))),
                max_abs_normalized_prediction_difference=max_normalized_difference,
                normalized_absolute_tolerance=1e-4)


def reproduce(task, args):
    task = dict(task)
    if args.protocol == "paper" and task["suite"] == "ood":
        task["split"] = "full"
    device = torch.device(args.device)
    output = args.output_dir/task["name"]
    output.mkdir(parents=True, exist_ok=True)
    meta = checkpoint_metadata(Path(task["checkpoint"]))
    shapes = meta["shapes"]
    hidden, _, patch = shapes["velocity_field.x_embedder.proj.weight"]
    depth = len({k.split("blocks.")[1].split(".")[0] for k in shapes if "velocity_field.blocks." in k})
    model = ConditionalFlowMatching(image_size=(task["side"],)*2, backbone="vibradit",
                                    dit_hidden_dim=hidden, dit_depth=depth,
                                    dit_patch_size=patch, dit_num_heads=args.heads).to(device)
    payload = torch.load(task["checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    direct_time = float(payload.get("direct_time", 0.))
    del payload
    model.eval()
    permutation = np.arange(task["side"]**2) if task["preserve_order"] else legacy_order(task["side"])
    undo = np.argsort(permutation)
    length = task["side"]**2
    start = time.monotonic()
    rows = []
    with h5py.File(task["source"], "r") as sf, h5py.File(task["target"], "r") as tf:
        n = len(sf["spectra"])
        if sf["spectra"].shape != tf["spectra"].shape or sf["spectra"].shape[1] != length:
            raise ValueError(f"HDF5 shapes differ from task: {task['name']}")
        if not np.array_equal(sf["x_axis"][:], tf["x_axis"][:]):
            raise ValueError(f"Source/target wavenumber axes differ: {task['name']}")
        if task["split"] == "random_test":
            indices = torch.randperm(n, generator=torch.Generator().manual_seed(task["seed"])).numpy()
            indices = indices[int(.7*n)+int(.15*n):]
        else:
            indices = np.arange(n)
        if args.limit:
            indices = indices[:args.limit]
        np.save(output/"h5_row_indices.npy", indices)
        for begin in range(0, len(indices), args.batch_size):
            selected = indices[begin:begin+args.batch_size]
            source = read_rows(sf["spectra"], selected).astype(np.float32)
            target = read_rows(tf["spectra"], selected).astype(np.float32)
            source_min, target_min = source.min(axis=1, keepdims=True), target.min(axis=1, keepdims=True)
            source_range = np.ptp(source, axis=1, keepdims=True)
            target_range = np.ptp(target, axis=1, keepdims=True)
            source_norm = (source-source_min)/(source_range+1e-8)
            target_norm = (target-target_min)/(target_range+1e-8)
            input_tensor = torch.from_numpy(source_norm[:, permutation].reshape(-1, 1, task["side"], task["side"])).to(device)
            with torch.inference_mode():
                target_mode = 0 if task["target_mode"] == "ir" else 2
                if task["model_type"] == "direct":
                    t = torch.full((len(selected),), direct_time, dtype=input_tensor.dtype, device=device)
                    generated = input_tensor + model.velocity_field(input_tensor, t, condition=input_tensor,
                                                                     target_mode=target_mode)
                    # The released Direct trainer returns unprojected residual
                    # predictions. Clipping would be an additional protocol choice.
                else:
                    generated = model.sample(input_tensor, target_mode=target_mode,
                                             num_steps=args.steps, use_rk4=False)
            pred_norm = generated.cpu().numpy().reshape(-1, length)[:, undo]
            pred = pred_norm*target_range+target_min
            # Match existing evaluator: both reference and prediction are
            # denormalized from float32 normalized heatmaps.
            reference = target_norm*target_range+target_min
            values = metric_arrays(reference, pred)
            df = pd.DataFrame(values)
            df.insert(0, "h5_row_index", selected)
            df.insert(0, "test_index", np.arange(begin, begin+len(selected)))
            # Existing tests skip the whole molecule on undefined Pearson,
            # PSNR, MAPE or other core metrics. Record rather than hide it.
            df["target_range"] = target_range.ravel()
            df["source_range"] = source_range.ravel()
            df["finite_core_metrics"] = np.isfinite(df[["r2", "pearson", "mae", "rmse"]]).all(axis=1)
            rows.append(df)
            if begin == 0:
                np.savez_compressed(output/"first_batch_predictions.npz", source=source, target=reference,
                                    prediction=pred, x_axis=tf["x_axis"][:], h5_row_indices=selected)
            if (begin//args.batch_size) % 100 == 0:
                print(f"{task['name']}: {begin+len(selected)}/{len(indices)}", flush=True)
    all_metrics = pd.concat(rows, ignore_index=True)
    all_metrics.to_csv(output/"metrics_all_molecules.csv", index=False)
    finite = all_metrics[all_metrics.finite_core_metrics]
    summary = dict(task, evaluated_n=len(all_metrics), finite_core_n=len(finite),
                   undefined_core_n=len(all_metrics)-len(finite), device=str(device),
                   gpu_name=torch.cuda.get_device_name() if device.type == "cuda" else None,
                   torch_version=torch.__version__, inference_seconds=time.monotonic()-start,
                   checkpoint_sha256=hash_file(task["checkpoint"]),
                   preprocessing="physical_order" if task["preserve_order"] else "legacy_patch_permutation",
                   solver="direct_residual" if task["model_type"] == "direct" else "Euler",
                   num_steps=1 if task["model_type"] == "direct" else args.steps,
                   attention_heads=args.heads, attention_heads_provenance="explicit_argument_not_inferred_from_weights",
                   evaluation_protocol=args.protocol, target_statistics_used_for_native_intensity_metrics=True,
                   evidence_level="checkpoint_inference_subset" if args.limit else "checkpoint_inference_full")
    for key in ("r2", "pearson", "mae", "rmse", "normalized_mae"):
        summary[key+"_mean_all_defined"] = float(all_metrics[key].mean())
        summary[key+"_mean_finite_core"] = float(finite[key].mean())
    summary["saved_prediction_anchor_check"] = compare_archived_anchors(task, output)
    (output/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--suite", choices=("id", "ood", "direct", "experimental", "all"), default="id")
    parser.add_argument("--protocol", choices=("archived", "paper"), default="archived",
                        help="OOD QM9: archived reproduces the saved random subset; paper evaluates the supplied file in full")
    parser.add_argument("--heads", type=int, default=6,
                        help="Attention head count must match the run configuration; it cannot be inferred from weight shapes")
    parser.add_argument("--task-config", type=Path,
                        help="Optional JSON list of task dictionaries; relative file paths resolve against the repository root")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"reproduction_audit/checkpoint_runs")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()
    available = TASKS
    if args.task_config:
        available = json.loads(args.task_config.read_text())
        required = {'name','dataset','source','target','checkpoint','source_mode','target_mode','side','seed','split','suite','model_type','preserve_order'}
        if not isinstance(available,list) or not available:
            raise ValueError('Task config must be a nonempty JSON list')
        if len({task['name'] for task in available}) != len(available):
            raise ValueError('Duplicate task names')
        for task in available:
            if required-set(task):
                raise ValueError(f'Missing task fields: {required-set(task)}')
            for key in ('source','target','checkpoint'):
                path = Path(task[key]).expanduser()
                task[key] = str(path if path.is_absolute() else ROOT/path)
    if args.list_tasks:
        print("\n".join(t["name"] for t in available))
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Allocate a GPU worker or explicitly request --device cpu.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    tasks = [t for t in available if (t["name"] in args.task if args.task else
             args.suite == "all" or t["suite"] == args.suite)]
    if args.task and set(args.task)-{t["name"] for t in available}:
        raise ValueError("Unknown task name")
    if not tasks:
        raise ValueError("No matching tasks")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for task in tasks:
        summaries.append(reproduce(task, args))
        pd.DataFrame(summaries).to_csv(args.output_dir/"checkpoint_summary.csv", index=False)


if __name__ == "__main__":
    main()
