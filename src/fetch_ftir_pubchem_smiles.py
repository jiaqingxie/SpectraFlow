r"""Fetch PubChem metadata for CID-named FTIR spectra.

Example:
    python src/fetch_ftir_pubchem_smiles.py ^
      --ftir-dir "C:\Users\ethzj\Downloads\dataset\dataset\ftir" ^
      --output "datasets\ftir_pubchem_metadata.csv"
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


SPLITS = ("train", "valid", "test")
PROPERTIES = (
    "Title,IUPACName,MolecularFormula,CanonicalSMILES,IsomericSMILES,"
    "InChI,InChIKey"
)
PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids}/"
    f"property/{PROPERTIES}/JSON"
)
OUTPUT_FIELDS = (
    "split",
    "cid",
    "spectrum_path",
    "label_path",
    "title",
    "iupac_name",
    "molecular_formula",
    "canonical_smiles",
    "isomeric_smiles",
    "inchi",
    "inchikey",
    "status",
)


def discover_spectra(ftir_dir: Path) -> list[dict[str, str]]:
    records = []
    for split in SPLITS:
        split_dir = ftir_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for spectrum_path in sorted(split_dir.glob("*.npy")):
            cid = spectrum_path.stem
            if not cid.isdigit():
                print(f"Skipping non-numeric filename: {spectrum_path.name}")
                continue
            label_path = spectrum_path.with_suffix(".txt")
            records.append(
                {
                    "split": split,
                    "cid": cid,
                    "spectrum_path": str(spectrum_path.resolve()),
                    "label_path": str(label_path.resolve()) if label_path.exists() else "",
                }
            )
    return records


def load_cache(cache_path: Path) -> dict[str, dict[str, str]]:
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(cid): metadata for cid, metadata in data.items()}


def save_cache(cache_path: Path, cache: dict[str, dict[str, str]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2, sort_keys=True)
    temporary_path.replace(cache_path)


def normalize_property(item: dict) -> dict[str, str]:
    # PubChem's current API returns ConnectivitySMILES and SMILES for the
    # requested CanonicalSMILES and IsomericSMILES fields, respectively.
    return {
        "title": str(item.get("Title", "")),
        "iupac_name": str(item.get("IUPACName", "")),
        "molecular_formula": str(item.get("MolecularFormula", "")),
        "canonical_smiles": str(
            item.get("ConnectivitySMILES", item.get("CanonicalSMILES", ""))
        ),
        "isomeric_smiles": str(
            item.get("SMILES", item.get("IsomericSMILES", ""))
        ),
        "inchi": str(item.get("InChI", "")),
        "inchikey": str(item.get("InChIKey", "")),
        "status": "ok",
    }


def request_batch(
    cids: list[str], timeout: float, max_retries: int
) -> dict[str, dict[str, str]]:
    url = PUBCHEM_URL.format(cids=urllib.parse.quote(",".join(cids), safe=","))
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SpectroGen-FTIR-metadata/1.0"},
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            properties = payload.get("PropertyTable", {}).get("Properties", [])
            return {
                str(item["CID"]): normalize_property(item)
                for item in properties
                if "CID" in item
            }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            if attempt >= max_retries:
                raise
            retry_after = getattr(error, "headers", {}).get("Retry-After")
            wait_seconds = float(retry_after) if retry_after else 2**attempt
            print(
                f"Request failed ({error}); retrying in {wait_seconds:.1f}s "
                f"[{attempt + 1}/{max_retries}]"
            )
            time.sleep(wait_seconds)
    return {}


def fetch_missing(
    cids: list[str],
    cache: dict[str, dict[str, str]],
    cache_path: Path,
    batch_size: int,
    delay: float,
    timeout: float,
    max_retries: int,
) -> None:
    missing = [cid for cid in dict.fromkeys(cids) if cid not in cache]
    total_batches = (len(missing) + batch_size - 1) // batch_size
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset : offset + batch_size]
        batch_number = offset // batch_size + 1
        print(
            f"Fetching batch {batch_number}/{total_batches} "
            f"({len(batch)} CIDs)..."
        )
        try:
            fetched = request_batch(batch, timeout, max_retries)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            print(f"Batch failed after retries: {error}")
            fetched = {}

        for cid in batch:
            cache[cid] = fetched.get(
                cid,
                {
                    "title": "",
                    "iupac_name": "",
                    "molecular_formula": "",
                    "canonical_smiles": "",
                    "isomeric_smiles": "",
                    "inchi": "",
                    "inchikey": "",
                    "status": "not_found_or_request_failed",
                },
            )
        save_cache(cache_path, cache)
        if offset + batch_size < len(missing):
            time.sleep(delay)


def write_output(
    output_path: Path,
    records: list[dict[str, str]],
    cache: dict[str, dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({**record, **cache.get(record["cid"], {})})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve CID-named FTIR spectra to PubChem SMILES."
    )
    parser.add_argument("--ftir-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="JSON cache path (default: <output>.cache.json).",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N spectra, useful for a smoke test.",
    )
    args = parser.parse_args()

    if not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be between 1 and 100")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    records = discover_spectra(args.ftir_dir)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError(f"No numeric .npy spectra found under {args.ftir_dir}")

    cache_path = args.cache or args.output.with_suffix(
        args.output.suffix + ".cache.json"
    )
    cache = load_cache(cache_path)
    print(f"Discovered {len(records)} spectra; cache contains {len(cache)} CIDs.")

    fetch_missing(
        [record["cid"] for record in records],
        cache,
        cache_path,
        args.batch_size,
        args.delay,
        args.timeout,
        args.max_retries,
    )
    write_output(args.output, records, cache)

    found = sum(cache[record["cid"]]["status"] == "ok" for record in records)
    print(f"Resolved: {found}/{len(records)}")
    print(f"Output: {args.output.resolve()}")
    print(f"Cache:  {cache_path.resolve()}")


if __name__ == "__main__":
    main()
