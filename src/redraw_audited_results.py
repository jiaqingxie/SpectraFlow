"""Draw reviewable Results figures from audited, existing numerical artifacts.

Follows Chen Liu's scientific-figure-making skill (figures4papers), with
publication-sized sans-serif text, semantic colors and editable vector export.
These figures do not substitute molecule variability for training-seed SD.
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green": "#8BCF8B",
           "red": "#B64342", "neutral": "#767676", "teal": "#42949E"}
ROOT = Path(__file__).resolve().parents[1]
PROPERTIES = ["LogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
              "RingCount", "FractionCSP3", "NumAromaticRings", "MolMR", "LabuteASA"]
DOMAINS = ["qm9", "mols", "pahs", "zinc15", "geom", "peptide", "peptide_mod", "nist_ir"]
LABELS = {"qm9": "QM9 source", "mols": "Mols", "pahs": "PAHs", "zinc15": "ZINC15",
          "geom": "GEOM", "peptide": "Peptide", "peptide_mod": "Peptide-mod", "nist_ir": "NIST-IR"}


def apply_publication_style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
                         "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 1., "legend.frameon": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})


def finalize_figure(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = out_path.with_suffix("."+extension)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.08)
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def label_panel(ax, text):
    ax.text(-.14, 1.06, text, transform=ax.transAxes, fontweight="bold", fontsize=12)


def figure_translation(output, audit, bundle=None):
    inventory = pd.read_csv((bundle or audit)/"saved_metrics_inventory.csv")
    primary = pd.read_csv((bundle or audit)/"recomputed_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11., 4.3), gridspec_kw={"width_ratios": [1., 1.35, 1.35]})
    datasets = ["qm9s", "qme14s", "vibench_full"]
    for j, (direction, color) in enumerate([( "ir2raman", PALETTE["blue_main"]), ("raman2ir", PALETTE["teal"])]):
        values = []
        for dataset in datasets:
            row = primary[primary.result_dir.str.contains(dataset+"_flow_"+direction)].iloc[0]
            values.append(row.r2_mean)
        y = np.arange(3)+(j-.5)*.18
        axes[0].plot(values, y, "o", color=color, label="IR → Raman" if j == 0 else "Raman → IR")
        for x, yy in zip(values, y):
            axes[0].text(x+.007, yy, f"{x:.3f}", va="center", fontsize=8)
    axes[0].set(yticks=np.arange(3), yticklabels=["QM9S", "QMe14S", "ViBench-Full"],
                xlim=(.72, 1.01), ylim=(2.5, -.5), xlabel="Mean per-molecule $R^2$", title="In-distribution · saved seed 2")
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.2), fontsize=8)
    provenance = primary[["result_dir", "n", "r2_mean", "pearson_mean", "rmse_mean"]].to_dict("records")
    for ax, direction in zip(axes[1:], ["ir2raman", "raman2ir"]):
        for i, domain in enumerate(DOMAINS):
            rows = []
            for method in ["vae", "flow"]:
                path = f"results/{domain}_ood_{method}_{direction}" + ("_vibradit" if method == "flow" else "") + f"/{method}_{direction}_r2_per_sample.csv"
                selected = inventory[inventory.path == path]
                if len(selected) != 1:
                    raise ValueError(f"Missing or ambiguous metric artifact: {path}")
                row = selected.iloc[0]
                rows.append(row.r2_mean)
                provenance.append({"path": path, "n": int(row.n), "r2_mean": row.r2_mean})
            ax.plot(rows, [i, i], color="#CFCECE", linewidth=2.2, zorder=1)
            ax.plot(rows[0], i, "s", color=PALETTE["red"], markersize=4.5, label="VAE" if i == 0 else None)
            ax.plot(rows[1], i, "o", color=PALETTE["blue_main"], markersize=5, label="SpectraFlow" if i == 0 else None)
        ax.axvline(0, color="#BBBBBB", linewidth=.8, linestyle="--")
        ax.set(yticks=np.arange(8), yticklabels=[LABELS[d] for d in DOMAINS], ylim=(7.6, -.6),
               xlim=(-.22, 1.0), xlabel="Mean per-molecule $R^2$", title="IR → Raman" if direction == "ir2raman" else "Raman → IR")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), fontsize=8, ncol=2)
    for ax, label in zip(axes, "abc"):
        label_panel(ax, label)
    fig.tight_layout(pad=1.8)
    paths = finalize_figure(fig, output/"result_1_translation")
    (output/"result_1_source_data.json").write_text(json.dumps(provenance, indent=2))
    return paths


def select_saved_rows(path, indices):
    found = {}
    for chunk in pd.read_csv(path, chunksize=128):
        for _, row in chunk[chunk["index"].isin(indices)].iterrows():
            found[int(row["index"])] = row.iloc[1:].to_numpy(dtype=float)
        if len(found) == len(indices):
            break
    if set(found) != set(indices):
        raise ValueError(f"Selected indices missing from {path}")
    return found


def figure_examples(output, audit, bundle=None):
    fig = plt.figure(figsize=(9.4, 6.0))
    grid = fig.add_gridspec(2, 2, hspace=.53, wspace=.28)
    if bundle:
        examples = np.load(bundle/"qm9s_quantile_examples.npz", allow_pickle=False)
        wave = examples["x_axis"]
        selection = pd.read_csv(bundle/"qm9s_quantile_selection.csv")
    else:
        with h5py.File(ROOT/"data/processed/ir_broaden_processed.h5") as f:
            wave = f["x_axis"][:]
    records = []
    for i, direction in enumerate(["ir2raman", "raman2ir"]):
        dirname = f"results/verify_fig1_seed2/qm9s_flow_{direction}_vibradit_seed2"
        selected = []
        if bundle:
            for _, row in selection[selection.direction == direction].iterrows():
                selected.append((int(row["index"]), float(row.error_percentile)/100, row))
        else:
            metrics = pd.read_csv(audit/(dirname.replace("/", "__")+f"__flow_{direction}_recomputed.csv"))
            for percentile in (.5, .9):
                value = metrics.normalized_mae.quantile(percentile)
                row = metrics.iloc[(metrics.normalized_mae-value).abs().argmin()]
                selected.append((int(row["index"]), percentile, row))
        indices = [item[0] for item in selected]
        if bundle:
            target = {idx: examples[f"{direction}_target"][j] for j, idx in enumerate(indices)}
            pred = {idx: examples[f"{direction}_prediction"][j] for j, idx in enumerate(indices)}
        else:
            target = select_saved_rows(ROOT/dirname/f"flow_{direction}_targets.csv", indices)
            pred = select_saved_rows(ROOT/dirname/f"flow_{direction}_preds.csv", indices)
        for j, (index, quantile, row) in enumerate(selected):
            inner = grid[i, j].subgridspec(2, 1, height_ratios=[3., 1.], hspace=.12)
            ax = fig.add_subplot(inner[0]); residual_ax = fig.add_subplot(inner[1], sharex=ax)
            y = target[index]; p = pred[index]
            scale = np.ptp(y)+1e-8
            yn, pn = (y-y.min())/scale, (p-y.min())/scale
            ax.plot(wave, yn, color="#767676", linewidth=1.1, label="Reference")
            ax.plot(wave, pn, color=PALETTE["blue_main"], linewidth=1., alpha=.85, label="Prediction")
            arrow = "IR → Raman" if direction == "ir2raman" else "Raman → IR"
            ax.set_title(f"QM9S · {arrow} · {int(quantile*100)}th error percentile", fontsize=9)
            ax.text(.99, .96, f"Test index {index}\n$R^2$ = {row.r2:.3f}\nnMAE = {row.normalized_mae:.4f}",
                    ha="right", va="top", transform=ax.transAxes, fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2))
            ax.set_ylabel("Normalized intensity"); ax.tick_params(labelbottom=False)
            if i == 0 and j == 0:
                ax.legend(loc="upper left", fontsize=7)
            residual_ax.plot(wave, pn-yn, color=PALETTE["red"], linewidth=.8)
            residual_ax.axhline(0, color="#BBBBBB", linewidth=.6)
            residual_ax.set(xlabel="Wavenumber (cm$^{-1}$)", ylabel="Residual")
            label_panel(ax, "abcd"[i*2+j])
            records.append(dict(dataset="QM9S", direction=direction, test_index=index,
                                selection_metric="normalized_mae", error_percentile=int(quantile*100),
                                r2=row.r2, normalized_mae=row.normalized_mae, result_dir=dirname))
    paths = finalize_figure(fig, output/"result_2_quantile_examples")
    pd.DataFrame(records).to_csv(output/"result_2_source_data.csv", index=False)
    return paths


def figure_downstream(output, bundle=None):
    source = bundle/"downstream_metrics.csv" if bundle else ROOT/"results/vibench_ood_downstream_plots/vibench_ood_downstream_metrics.csv"
    df = pd.read_csv(source)
    df = df[df.property.isin(PROPERTIES)]
    if len(df) != 8*10*5 or df.duplicated(["dataset", "property", "feature"]).any():
        raise ValueError("Unexpected downstream property/feature coverage")
    pivot = df.pivot(index=["dataset", "property"], columns="feature", values="r2_mean")
    means = pivot.groupby("dataset").mean().reindex(DOMAINS)
    gains = (pivot["Flow + Morgan FP"]-pivot["Morgan fingerprint"]).unstack("property").reindex(index=DOMAINS, columns=PROPERTIES)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), gridspec_kw={"width_ratios": [1., 2.]})
    for j, (feature, label, color, marker) in enumerate([
        ("Flow embedding", "Flow", PALETTE["blue_main"], "o"),
        ("Morgan fingerprint", "Morgan FP", PALETTE["neutral"], "s"),
        ("Flow + Morgan FP", "Flow + FP", PALETTE["teal"], "D"),
    ]):
        axes[0].plot(means[feature], np.arange(8)+(j-1)*.14, linestyle="none", marker=marker,
                     markersize=4.5, color=color, label=label)
    axes[0].set(yticks=np.arange(8), yticklabels=[LABELS[d] for d in DOMAINS], ylim=(7.65, -.65),
                xlim=(-.05, 1.03), xlabel="Mean $R^2$ across ten descriptors", title="Fixed features · fitted ridge probes")
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.18), fontsize=8)
    vmax = max(abs(gains.min().min()), abs(gains.max().max()))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("gain", [PALETTE["red"], "white", PALETTE["blue_main"]])
    im = axes[1].imshow(gains, cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
    axes[1].set(yticks=np.arange(8), yticklabels=[LABELS[d] for d in DOMAINS],
                xticks=np.arange(10), xticklabels=PROPERTIES, title="Adding Flow to Morgan fingerprints")
    axes[1].tick_params(axis="x", rotation=55, labelsize=8)
    for (i, j), value in np.ndenumerate(gains.to_numpy()):
        axes[1].text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=6.5,
                     color="white" if abs(value) > .65*vmax else "#272727")
    fig.colorbar(im, ax=axes[1], label="$\Delta R^2$ (Flow + FP minus FP)", shrink=.8)
    for ax, label in zip(axes, "ab"):
        label_panel(ax, label)
    fig.tight_layout(pad=1.8)
    paths = finalize_figure(fig, output/"result_3_downstream")
    df.to_csv(output/"result_3_source_data.csv", index=False)
    gains.to_csv(output/"result_3_fingerprint_gains.csv")
    return paths


def figure_experimental(output, audit, bundle=None):
    source = bundle if bundle else audit/"experimental"
    metrics = pd.read_csv(source/"rruff_recomputed_per_pair.csv")
    arrays = np.load(source/"rruff_physical_order_predictions.npz", allow_pickle=False)
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.1))
    for column, label, color in [("pearson", "Raw", PALETTE["blue_main"]),
                                 ("pearson_baseline_corrected", "AsLS corrected", PALETTE["teal"])]:
        values = np.sort(metrics[column].dropna().to_numpy())
        axes[0, 0].step(values, np.arange(1, len(values)+1)/len(values), where="post", label=label, color=color)
        median = float(np.median(values))
        axes[0, 0].axvline(median, color=color, linewidth=.8, linestyle="--")
    axes[0, 0].set(xlim=(-1, 1), ylim=(0, 1.02), xlabel="Per-pair Pearson $r$",
                   ylabel="Cumulative fraction", title="147 external RRUFF pairs")
    axes[0, 0].text(.03, .96, "Median: raw 0.309 · corrected 0.340", va="top",
                     transform=axes[0, 0].transAxes, fontsize=8)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    selected = metrics[metrics.r2 > .5].sort_values("pearson", ascending=False).drop_duplicates("mineral_name").head(3)
    if len(selected) != 3:
        raise ValueError("Not enough distinct eligible RRUFF examples")
    for ax, (_, row) in zip(axes.ravel()[1:], selected.iterrows()):
        location = np.flatnonzero(arrays["indices"] == row["index"])[0]
        ax.plot(arrays["x_axis"], arrays["target"][location], color=PALETTE["neutral"], linewidth=1., label="Reference Raman")
        ax.plot(arrays["x_axis"], arrays["prediction"][location], color=PALETTE["blue_main"], linewidth=1., alpha=.85, label="Prediction")
        ax.set(xlabel="Wavenumber (cm$^{-1}$)", ylabel="Normalized intensity",
               title=f"{str(row.mineral_name).title()} · {str(row.rruff_id).upper()}\nSelected high correlation: $r$ = {row.pearson:.3f}, $R^2$ = {row.r2:.3f}")
    axes[0, 1].legend(loc="upper right", fontsize=7)
    for ax, label in zip(axes.ravel(), "abcd"):
        label_panel(ax, label)
    fig.tight_layout(pad=1.8, h_pad=2.8, w_pad=2.0)
    paths = finalize_figure(fig, output/"result_4_rruff_external")
    metrics.to_csv(output/"result_4_source_data.csv", index=False)
    selected.to_csv(output/"result_4_selected_examples.csv", index=False)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=ROOT/"reproduction_audit")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"figures/reproduction")
    parser.add_argument("--source-bundle", type=Path,
                        help="Redraw from the compact published bundle without datasets or checkpoints")
    args = parser.parse_args()
    args.audit_dir = args.audit_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.source_bundle:
        args.source_bundle = args.source_bundle.resolve()
    args.audit_dir = args.audit_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.source_bundle:
        args.source_bundle = args.source_bundle.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()
    outputs = figure_translation(args.output_dir, args.audit_dir, args.source_bundle)
    outputs += figure_examples(args.output_dir, args.audit_dir, args.source_bundle)
    outputs += figure_downstream(args.output_dir, args.source_bundle)
    outputs += figure_experimental(args.output_dir, args.audit_dir, args.source_bundle)
    (args.output_dir/"figure_manifest.json").write_text(json.dumps({
        "outputs": outputs,
        "style_reference": "https://github.com/ChenLiu-1996/figures4papers/tree/main/scientific-figure-making",
        "evidence": "existing saved predictions and probe summaries; see source data files",
        "training_seed_uncertainty": "unverified; not drawn",
        "captions": {
            "result_1": "Panel a: recalculated means from saved seed-2 predictions. Panels b,c: individual saved VAE/Flow evaluations, rather than the manuscript's seed-averaged OOD table. QM9 is the source-domain test subset. No seed uncertainty is inferred.",
            "result_2": "Samples selected algorithmically nearest to the 50th and 90th percentile of normalized MAE over the full saved QM9S seed-2 test results. Vertical normalization uses the reference target range for evaluation. No physical conversion trajectory is claimed.",
            "result_3": "Saved five-split ridge summaries for ten RDKit descriptors. Panel a averages per-property R2 means; panel b displays signed gains including decreases. These are newly fitted probes on each domain, not zero-shot property prediction.",
            "result_4": "Panel a shows all 147 RRUFF-ID-disjoint external pairs, with raw and AsLS-corrected Pearson distributions. Panels b-d use raw R2 > 0.5 and descending raw Pearson, retaining one example per mineral name; they are selected high-correlation examples. Saved spectra were restored to physical order and checked against source HDF5 before baseline fitting. 72 of 135 test mineral names overlap development. The NIST figure is excluded because its archived molecular identity matching is invalid."
        }}, indent=2))
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
