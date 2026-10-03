"""Materialize immutable identity/scaffold splits and a finite run manifest."""
import json
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from .common import ROOT, OUT, RAW_DATA, PROTOCOL_VERSION, digest, save_json, validate_split


@lru_cache(maxsize=600000)
def chemical_keys(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return '', '', ''
    stereo = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    identity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms():
        group = 'ring:' + Chem.MolToSmiles(scaffold, isomericSmiles=False)
    else:
        # Empty Murcko scaffolds cannot define independent acyclic families.
        generic = Chem.RWMol(mol)
        for atom in generic.GetAtoms():
            atom.SetAtomicNum(0)
            atom.SetFormalCharge(0)
            atom.SetIsotope(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(True)
            atom.SetIsAromatic(False)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for bond in generic.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
            bond.SetStereo(Chem.BondStereo.STEREONONE)
        # Dummy atoms support hypervalent S/P topologies without invalid C valence.
        group = 'acyclic_topology:' + Chem.MolToSmiles(generic.GetMol(), isomericSmiles=False)
    return stereo, identity, group


def group_partition(groups, seed=2):
    import hashlib
    def label(group):
        value = int(hashlib.sha256(f'{PROTOCOL_VERSION}:{seed}:{group}'.encode()).hexdigest()[:16], 16) / 2**64
        return 'train' if value < .7 else 'valid' if value < .8 else 'calibration' if value < .85 else 'test'
    mapping = {g: label(g) for g in set(groups)}
    return np.array([mapping[g] for g in groups], dtype=object)


def spectrum_quality(source, target):
    good, reasons = [], []
    with h5py.File(source) as s, h5py.File(target) as t:
        if s['spectra'].shape != t['spectra'].shape:
            raise ValueError('Paired shapes differ')
        if not np.array_equal(s['x_axis'][:], t['x_axis'][:]):
            raise ValueError('Paired axes differ')
        for start in range(0, len(s['spectra']), 512):
            a, b = s['spectra'][start:start+512], t['spectra'][start:start+512]
            finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            nonconstant = (np.ptp(a, axis=1) > 1e-8) & (np.ptp(b, axis=1) > 1e-8)
            good.extend(finite & nonconstant)
            reasons.extend(np.where(~finite, 'nonfinite_spectrum', np.where(~nonconstant, 'constant_source_or_target', '')))
    return np.array(good), reasons


def load_smiles(path):
    return [line.strip().split('\t')[-1] for line in Path(path).read_text().splitlines()]


def prepare(output=OUT):
    RDLogger.DisableLog('rdApp.*')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'splits').mkdir(exist_ok=True)
    catalog = []
    qm9map = RAW_DATA/'qm9s/mapping.txt'
    qm9rows = [line.split('\t', 1) for line in qm9map.read_text().splitlines()]
    if [int(x[0]) for x in qm9rows] != list(range(1, len(qm9rows)+1)):
        raise ValueError('QM9S source mapping IDs are not sequential')
    inputs = [('qm9s', ROOT/'data/processed/ir_broaden_processed.h5',
               ROOT/'data/processed/raman_broaden_processed.h5', [x[1] for x in qm9rows], str(qm9map))]
    qme = pd.read_csv(ROOT/'data/processed/qme14s_id_smiles.csv')
    if not np.array_equal(qme.row_index, np.arange(len(qme))):
        raise ValueError('QMe14S metadata row indices are not sequential')
    inputs.append(('qme14s', RAW_DATA/'QMe14S/processed/ir_broaden_processed.h5',
                   RAW_DATA/'QMe14S/processed/raman_broaden_processed.h5', qme.smiles.tolist(),
                   str(ROOT/'data/processed/qme14s_id_smiles.csv')))
    with h5py.File(ROOT/'data/processed/vibench_test_full_ir_processed.h5') as f:
        domains = f.attrs['datasets'].split(',')
    combined_smiles, domain_labels = [], []
    domain_records = []
    for domain in domains:
        ip = ROOT/'data/processed'/f'{domain}_test_ir_smiles.txt'
        rp = ROOT/'data/processed'/f'{domain}_test_raman_smiles.txt'
        a, b = load_smiles(ip), load_smiles(rp)
        if a != b:
            raise ValueError(f'{domain} source/target SMILES lists differ')
        with h5py.File(ROOT/'data/processed'/f'{domain}_test_ir_processed.h5') as f:
            if len(a) != len(f['spectra']):
                raise ValueError(f'{domain} SMILES/HDF5 counts differ')
        domain_records.append((domain, a, ip))
        combined_smiles.extend(a)
        domain_labels.extend([domain] * len(a))
    inputs.append(('vibench_full', ROOT/'data/processed/vibench_test_full_ir_processed.h5',
                   ROOT/'data/processed/vibench_test_full_raman_processed.h5', combined_smiles,
                   'merged HDF5 datasets attribute plus matching per-domain SMILES lists'))
    identity_sets = {}
    for name, source, target, smiles, provenance in inputs:
        with h5py.File(source) as f:
            if len(smiles) != len(f['spectra']):
                raise ValueError(f'{name} metadata count mismatch')
            side = int(np.sqrt(f['spectra'].shape[1]))
        print(f'Preparing {name}: {len(smiles)} rows', flush=True)
        keys = [chemical_keys(s) for s in smiles]
        frame = pd.DataFrame(keys, columns=['stereo_identity', 'identity', 'scaffold'])
        frame.insert(0, 'row_id', np.arange(len(frame)))
        frame['smiles'] = smiles
        frame['domain'] = domain_labels if name == 'vibench_full' else name
        good, reasons = spectrum_quality(source, target)
        frame['excluded_reason'] = reasons
        frame.loc[frame.identity == '', 'excluded_reason'] = 'invalid_smiles'
        good &= frame.identity.to_numpy() != ''
        identity_sets[name] = set(frame.loc[good, 'identity'])
        for split_type in ('identity', 'scaffold'):
            split = frame.copy()
            split['split_group'] = split[split_type]
            split['split'] = group_partition(split.split_group)
            split.loc[~good, 'split'] = 'excluded'
            validate_split(split)
            destination = output/'splits'/f'{name}_{split_type}.csv'
            split.to_csv(destination, index=False)
            entry = dict(dataset=name, protocol=split_type, source=str(source), target=str(target),
                         split=str(destination), split_sha256=digest(destination), side=side,
                         patch=20 if side == 60 else 8, epochs=100,
                         counts=split.split.value_counts().to_dict(), metadata_provenance=provenance,
                         source_metadata_verified_by='row count/order contract; molecular source anchor audit separately required')
            catalog.append(entry)
    # Shared identities across ViBench domains and the QM9S/QMe14S resources.
    for domain, smiles, _ in domain_records:
        identity_sets['vibench_'+domain] = {chemical_keys(s)[1] for s in smiles} - {''}
    overlap = []
    for a, aset in identity_sets.items():
        for b, bset in identity_sets.items():
            if a < b:
                overlap.append(dict(dataset_a=a, dataset_b=b, unique_a=len(aset), unique_b=len(bset),
                                    shared_connectivity_identities=len(aset & bset)))
    pd.DataFrame(overlap).to_csv(output/'identity_overlap.csv', index=False)
    # Preserve every available external RRUFF row; split the complete pool by ID/name.
    directory = ROOT/'datasets/openspecy_rruff_576'
    metadata = pd.read_csv(ROOT/'datasets/openspecy_cov100/metadata.csv')
    names = metadata.dropna(subset=['rruffid']).assign(identity=lambda x: x.rruffid.str.lower()).drop_duplicates('identity').set_index('identity').spectrum_identity
    tables = [pd.read_csv(directory/f'{part}_pairs.csv').assign(original_partition=part) for part in ('train','test')]
    rruff = pd.concat(tables, ignore_index=True)
    rruff['row_id'] = np.arange(len(rruff))
    rruff['identity'] = rruff.identity_key
    rruff['mineral_name'] = rruff.identity.map(names)
    if rruff.mineral_name.isna().any():
        raise ValueError('Unknown RRUFF material name')
    # A compact new HDF5 is needed because source files have separate row spaces.
    (output/'data').mkdir(exist_ok=True)
    for mode in ('ir','raman'):
        dest = output/'data'/f'rruff_{mode}.h5'
        with h5py.File(directory/f'train_{mode}.h5') as a, h5py.File(directory/f'test_{mode}.h5') as b:
            if not np.array_equal(a['x_axis'][:], b['x_axis'][:]):
                raise ValueError('RRUFF axes differ')
            with h5py.File(dest, 'w') as f:
                f.create_dataset('spectra', data=np.concatenate([a['spectra'][:], b['spectra'][:]]), compression='gzip')
                f.create_dataset('x_axis', data=a['x_axis'][:])
    for protocol, key in [('identity','identity'), ('mineral','mineral_name')]:
        frame = rruff.copy()
        frame['split_group'] = frame[key]
        frame['split'] = group_partition(frame.split_group)
        good, reason = spectrum_quality(output/'data/rruff_ir.h5', output/'data/rruff_raman.h5')
        frame['excluded_reason'] = reason
        frame.loc[~good, 'split'] = 'excluded'
        validate_split(frame)
        path = output/'splits'/f'rruff_{protocol}.csv'
        frame.to_csv(path, index=False)
        catalog.append(dict(dataset='rruff', protocol=protocol, source=str(output/'data/rruff_ir.h5'),
                            target=str(output/'data/rruff_raman.h5'), split=str(path), split_sha256=digest(path),
                            side=24, patch=4, epochs=30, counts=frame.split.value_counts().to_dict(),
                            note='New independent benchmark, not a reproduction of the original external split'))
    save_json(output/'catalog.json', catalog)
    jobs = []
    for data in catalog:
        directions = ('ir2raman',) if data['dataset']=='rruff' else ('ir2raman','raman2ir')
        for direction in directions:
            for method in ('flow','direct'):
                for seed in range(4):
                    variants = ['full']
                    if method=='flow' and data['protocol']=='identity' and data['dataset'] in ('qm9s','qme14s'):
                        variants += ['no_peak','no_shape','no_ot','basic','legacy_order']
                    for variant in variants:
                        name = f"{data['dataset']}_{data['protocol']}_{direction}_{method}_{variant}_seed{seed}"
                        jobs.append(dict(**data, name=name, direction=direction, method=method, variant=variant,
                                         seed=seed, split_seed=2, batch_size=32, learning_rate=.0002,
                                         hidden=384, depth=8, heads=6, sigma_min=.01, endpoint_weight=.5,
                                         endpoint_probability=.1, steps=8, protocol_version=PROTOCOL_VERSION,
                                         budget='equal_updates', max_train_seconds=None))
    save_json(output/'run_manifest.json', jobs)
    save_json(output/'preparation_summary.json', dict(protocol_version=PROTOCOL_VERSION,
              datasets=catalog, training_jobs=len(jobs), status='prepared_not_trained',
              split_rule='SHA256 group assignment: 70% train, 10% validation, 5% calibration, 15% test in expectation',
              acyclic_scaffold_rule='generic full molecular topology, preserving chain/branch connectivity',
              missing_evidence=['same-specimen paired experimental acquisition metadata',
                                'provenance of original OOD attention-head and run configuration',
                                'uncertainty calibration requires newly trained seed ensemble']))
    print(f'Prepared {len(catalog)} datasets/protocols and {len(jobs)} training runs', flush=True)
    return catalog
