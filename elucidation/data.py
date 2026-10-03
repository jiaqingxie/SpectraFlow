from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def load_processed_spectra(csv_path: str) -> np.ndarray:
    """
    Load spectra CSV with automatic format detection.

    Supported layouts:
    1) processed-style: first row is x-axis, first col is sample index
       -> use df.iloc[1:, 1:]
    2) index-only table: first col is index, no x-axis row
       -> use df.iloc[:, 1:]
    3) plain table: full matrix is spectra
       -> use df.iloc[:, :]

    This makes it compatible with both *_processed.csv and raw boraden-style exports.
    """
    raw = pd.read_csv(csv_path, header=None)
    num = _to_numeric_df(raw)

    # Heuristic: detect x-axis header row
    first_row = num.iloc[0, :]
    row_numeric_ratio = first_row.notna().mean()
    has_axis_row = row_numeric_ratio > 0.7

    # Heuristic: detect index column
    col0 = num.iloc[:, 0]
    col0_body = col0.iloc[1:] if has_axis_row else col0
    col_numeric_ratio = col0_body.notna().mean()
    has_index_col = col_numeric_ratio > 0.9

    if has_axis_row and has_index_col:
        mat = num.iloc[1:, 1:]
    elif has_index_col:
        mat = num.iloc[:, 1:]
    elif has_axis_row:
        mat = num.iloc[1:, :]
    else:
        mat = num

    # Drop rows/cols that are fully NaN and fill remaining NaN with 0
    mat = mat.dropna(axis=0, how="all").dropna(axis=1, how="all").fillna(0.0)
    arr = mat.values.astype(np.float32)
    return arr


def parse_smiles_line(line: str) -> str:
    s = line.strip()
    if not s:
        return ""
    if "\t" in s:
        parts = s.split("\t")
        return parts[-1].strip()
    if "," in s:
        parts = s.split(",")
        return parts[-1].strip()
    if " " in s:
        parts = s.split()
        if len(parts) >= 2:
            return parts[-1].strip()
    return s


def load_smiles(smiles_txt: str) -> list[str]:
    lines = Path(smiles_txt).read_text(encoding="utf-8").splitlines()
    smiles = [parse_smiles_line(x) for x in lines]
    smiles = [x for x in smiles if x]
    return smiles


class SpectrumSmilesDataset(Dataset):
    def __init__(
        self,
        ir_csv: str,
        raman_csv: str,
        smiles_txt: str,
        tokenizer,
        input_modality: str = "ir",
        max_smiles_len: int = 256,
    ):
        super().__init__()
        self.ir = load_processed_spectra(ir_csv)
        self.raman = load_processed_spectra(raman_csv)
        self.smiles = load_smiles(smiles_txt)
        self.tokenizer = tokenizer
        self.input_modality = input_modality
        self.max_smiles_len = max_smiles_len

        n = min(len(self.ir), len(self.raman), len(self.smiles))
        self.ir = self.ir[:n]
        self.raman = self.raman[:n]
        self.smiles = self.smiles[:n]

    def __len__(self) -> int:
        return len(self.smiles)

    def _build_input(self, i: int) -> np.ndarray:
        if self.input_modality == "ir":
            x = self.ir[i][None, :]  # (1, L)
        elif self.input_modality == "raman":
            x = self.raman[i][None, :]  # (1, L)
        elif self.input_modality == "ir_raman":
            # vib2mol-style multimodal spectral input: two channels
            x = np.stack([self.ir[i], self.raman[i]], axis=0)  # (2, L)
        else:
            raise ValueError(f"Unknown input_modality: {self.input_modality}")
        return x.astype(np.float32)

    def __getitem__(self, i: int):
        x = self._build_input(i)
        y_ids = self.tokenizer.encode(self.smiles[i], add_bos=True, add_eos=True)
        y_ids = y_ids[: self.max_smiles_len]
        return torch.from_numpy(x), torch.tensor(y_ids, dtype=torch.long)


def collate_fn(batch, pad_id: int):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # (B, L)

    max_len = max(len(y) for y in ys)
    y_pad = torch.full((len(ys), max_len), pad_id, dtype=torch.long)
    for i, y in enumerate(ys):
        y_pad[i, : len(y)] = y
    return x, y_pad
