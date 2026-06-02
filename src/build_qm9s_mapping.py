"""
Build index-SMILES mapping from QM9S per-molecule CSV files.

Assumed CSV format (from readme):
  column0: QM9S index
  column1: QM9 original index
  column2: SMILES
  column3: total atom count

For robustness, this script tries:
  1) column name "smiles" (case-insensitive)
  2) third column (index 2)

Output format (mapping.txt):
  <index>\t<SMILES>
where <index> comes from filename stem (xxxxxx.csv -> xxxxxx).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm


def extract_smiles_from_csv(csv_path: Path) -> str | None:
    """Extract SMILES string from one CSV file."""
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception:
        return None

    if not rows:
        return None

    header = rows[0]
    data_row = rows[1] if len(rows) > 1 else rows[0]

    # QM9S per-file CSV special format:
    # row with first cell == "0" stores metadata:
    #   col1: qm9s index, col2: qm9 index, col3: smiles, col4: atom count
    # (because the first column is row id, smiles is at position 3 here)
    for row in rows[1:]:
        if not row:
            continue
        row_id = row[0].strip()
        if row_id in {"0", "0.0"} and len(row) > 3:
            s = row[3].strip()
            if s:
                return s

    # 1) Prefer named "smiles" column
    smiles_col = None
    for i, col in enumerate(header):
        if col.strip().lower() == "smiles":
            smiles_col = i
            break

    if smiles_col is not None and smiles_col < len(data_row):
        s = data_row[smiles_col].strip()
        if s:
            return s

    # 2) Fallback: third column (index 2) in a normal flat table
    if len(data_row) > 2:
        s = data_row[2].strip()
        if s:
            return s

    return None


def build_mapping(csv_dir: Path, output_file: Path) -> tuple[int, int]:
    """Build mapping file from all CSVs in csv_dir."""
    csv_files = sorted(csv_dir.glob("*.csv"))
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    fail_count = 0

    with output_file.open("w", encoding="utf-8", newline="\n") as out:
        pbar = tqdm(csv_files, desc="Building index-SMILES mapping", unit="file")
        for file_path in pbar:
            pbar.set_postfix_str(file_path.name)
            index = file_path.stem
            smiles = extract_smiles_from_csv(file_path)
            if smiles is None:
                fail_count += 1
                continue
            out.write(f"{index}\t{smiles}\n")
            ok_count += 1

    return ok_count, fail_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build index-SMILES mapping from QM9S CSV directory."
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="/mnt/shared-storage-user/xiejiaqing/data/qm9s/qm9s_csv",
        help="Directory containing per-molecule CSV files (xxxxxx.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/shared-storage-user/xiejiaqing/data/qm9s/mapping.txt",
        help="Output mapping file path.",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    output_file = Path(args.output)

    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    ok_count, fail_count = build_mapping(csv_dir, output_file)
    total = ok_count + fail_count
    print(f"Done. Total CSVs: {total}")
    print(f"Mapped: {ok_count}")
    print(f"Failed: {fail_count}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()

