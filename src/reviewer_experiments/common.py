"""Shared protocol, lazy spectral data and physical-wavenumber metrics."""
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reviewer_experiments'
RAW_DATA = Path(os.environ.get('SPECTRAFLOW_RAW_DATA_ROOT', str(ROOT.parent / 'data'))).expanduser().resolve()
PROTOCOL_VERSION = 'reviewer-v1'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(temporary, path)


def normalized(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min(axis=-1, keepdims=True)) / (np.ptp(x, axis=-1, keepdims=True) + 1e-8)


def rows(ds, ids):
    unique, inverse = np.unique(ids, return_inverse=True)
    return ds[unique][inverse]


def validate_split(frame):
    if frame.row_id.duplicated().any():
        raise ValueError('A source row appears more than once in the split')
    if not set(frame.split).issubset({'train', 'valid', 'calibration', 'test', 'excluded'}):
        raise ValueError('Unknown split label')
    active = frame[frame.split != 'excluded']
    for name in ('train', 'valid', 'calibration', 'test'):
        if not (active.split == name).any():
            raise ValueError(f'Empty {name} split')
    for key in ('identity', 'split_group'):
        if active.groupby(key).split.nunique().max() != 1:
            raise ValueError(f'{key} leaks between partitions')


def peak_metrics(target, prediction, wave, tolerance=20., prominence=.05):
    wave = np.asarray(wave)
    spacing = float(np.median(np.diff(wave)))
    if spacing <= 0:
        raise ValueError('Peak metrics require increasing physical wavenumbers')
    distance = max(1, int(np.ceil(10 / spacing)))
    yp, pp = normalized(target), normalized(prediction)
    yi = find_peaks(yp, prominence=prominence, distance=distance)[0]
    pi = find_peaks(pp, prominence=prominence, distance=distance)[0]
    if len(yi) and len(pi):
        separation = np.abs(wave[yi, None] - wave[pi][None, :])
        cost = np.where(separation <= tolerance, separation,
                        (max(len(yi), len(pi)) + 1) * (tolerance + 1))
        a, b = linear_sum_assignment(cost)
        good = separation[a, b] <= tolerance
        a, b = a[good], b[good]
    else:
        a, b = np.array([], dtype=int), np.array([], dtype=int)
    matches = len(a)
    precision = matches / len(pi) if len(pi) else (1. if not len(yi) else 0.)
    recall = matches / len(yi) if len(yi) else (1. if not len(pi) else 0.)
    strongest = set(np.argsort(yp[yi])[-5:])
    return dict(peak_precision=precision, peak_recall=recall,
                peak_f1=2 * precision * recall / (precision + recall) if precision + recall else 0.,
                peak_position_mae=float(np.mean(np.abs(wave[yi[a]] - wave[pi[b]]))) if matches else np.nan,
                peak_height_mae=float(np.mean(np.abs(yp[yi[a]] - pp[pi[b]]))) if matches else np.nan,
                strong5_recall=len(strongest & set(a)) / len(strongest) if strongest else np.nan,
                reference_peak_n=len(yi), predicted_peak_n=len(pi))


def metrics(y, p, wave=None):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    err = p - y
    yc, pc = y - y.mean(axis=1, keepdims=True), p - p.mean(axis=1, keepdims=True)
    denominator = np.sum(yc * yc, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = 1 - np.sum(err * err, axis=1) / denominator
        corr = np.sum(yc * pc, axis=1) / np.sqrt(denominator * np.sum(pc * pc, axis=1))
    frame = pd.DataFrame(dict(r2=r2, pearson=corr, nmae=np.abs(err).mean(axis=1),
                              nrmse=np.sqrt((err * err).mean(axis=1))))
    if wave is not None:
        frame = pd.concat([frame, pd.DataFrame([peak_metrics(a, b, wave) for a, b in zip(y, p)])], axis=1)
    return frame


def summary_metrics(frame):
    result = {}
    for key in frame.select_dtypes(include='number'):
        s = frame[key].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s):
            result[key] = dict(mean=float(s.mean()), median=float(s.median()), defined_n=len(s))
    return result


class LazyPairs:
    """Process-local HDF5 handles; no eager copies or shared worker handles."""
    def __init__(self, source, target, ids, permutation=None):
        self.source_path, self.target_path = str(source), str(target)
        self.ids = np.asarray(ids, dtype=np.int64)
        self.permutation = permutation
        self._files = None
        with h5py.File(source) as s, h5py.File(target) as t:
            if s['spectra'].shape != t['spectra'].shape:
                raise ValueError('Source/target dimensions differ')
            self.length = s['spectra'].shape[1]
            self.side = int(np.sqrt(self.length))
            if self.side ** 2 != self.length:
                raise ValueError('Backbone expects square-length spectra')
            if not np.array_equal(s['x_axis'][:], t['x_axis'][:]):
                raise ValueError('Source/target axes differ')
            self.wave = s['x_axis'][:]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        import torch
        if self._files is None:
            self._files = (h5py.File(self.source_path, 'r'), h5py.File(self.target_path, 'r'))
        row = self.ids[index]
        source, target = [normalized(f['spectra'][row]) for f in self._files]
        if self.permutation is not None:
            source, target = source[self.permutation], target[self.permutation]
        return (torch.from_numpy(source.copy().reshape(1, self.side, self.side)),
                torch.from_numpy(target.copy().reshape(1, self.side, self.side)), int(row))

    def __getstate__(self):
        return {**self.__dict__, '_files': None}

    def close(self):
        if self._files:
            for f in self._files:
                f.close()
            self._files = None
