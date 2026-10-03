"""Export compact numerical inputs for redrawing all four audited Results figures."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from redraw_audited_results import ROOT, select_saved_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=ROOT/'reproduction_audit')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'figures/reproduction/source_bundle')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        'saved_metrics_inventory.csv': args.audit_dir/'saved_metrics_inventory.csv',
        'recomputed_summary.csv': args.audit_dir/'recomputed_summary.csv',
        'downstream_metrics.csv': ROOT/'results/vibench_ood_downstream_plots/vibench_ood_downstream_metrics.csv',
        'rruff_recomputed_per_pair.csv': args.audit_dir/'experimental/rruff_recomputed_per_pair.csv',
        'rruff_physical_order_predictions.npz': args.audit_dir/'experimental/rruff_physical_order_predictions.npz',
    }
    for destination, source in files.items():
        shutil.copy2(source, args.output_dir/destination)
    with h5py.File(ROOT/'data/processed/ir_broaden_processed.h5') as f:
        arrays = {'x_axis': f['x_axis'][:]}
    selections = []
    for direction in ('ir2raman', 'raman2ir'):
        directory = f'results/verify_fig1_seed2/qm9s_flow_{direction}_vibradit_seed2'
        frame = pd.read_csv(args.audit_dir/(directory.replace('/','__')+f'__flow_{direction}_recomputed.csv'))
        indices = []
        for percentile in (.5,.9):
            value = frame.normalized_mae.quantile(percentile)
            row = frame.iloc[(frame.normalized_mae-value).abs().argmin()]
            index = int(row['index'])
            indices.append(index)
            selections.append(dict(index=index,direction=direction,error_percentile=int(percentile*100),
                                   r2=float(row.r2),normalized_mae=float(row.normalized_mae),
                                   original_result_dir=directory,selection_population_n=len(frame)))
        for kind, suffix in [('target','targets'),('prediction','preds')]:
            selected = select_saved_rows(ROOT/directory/f'flow_{direction}_{suffix}.csv', indices)
            arrays[f'{direction}_{kind}'] = np.stack([selected[i] for i in indices])
    np.savez_compressed(args.output_dir/'qm9s_quantile_examples.npz', **arrays)
    pd.DataFrame(selections).to_csv(args.output_dir/'qm9s_quantile_selection.csv',index=False)
    artifacts = {p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in sorted(args.output_dir.iterdir()) if p.is_file() and p.name!='bundle_manifest.json'}
    manifest = dict(purpose='Figure reconstruction from published numerical source inputs; no model weights',
                    files=artifacts,provenance=dict(qm9s='Saved fixed seed-2 test predictions; quantile-selected examples',
                    rruff='147 external ID-disjoint pairs, restored to physical wavenumber order',
                    downstream='Saved five-split ridge summaries; independently fitted per domain'),
                    evidence_limits=['Does not establish four-seed checkpoint reproduction',
                                     'Archived QM9 OOD evaluated a re-split subset',
                                     'Original NIST identity matching is invalid and excluded from figures'])
    (args.output_dir/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Exported {len(artifacts)} compact figure source files to {args.output_dir}')


if __name__=='__main__':
    main()
