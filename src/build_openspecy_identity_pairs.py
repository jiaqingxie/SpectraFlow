"""Build identity-disjoint Open Specy pairs for model training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def minmax(spectra: np.ndarray) -> np.ndarray:
    minimum = spectra.min(axis=1, keepdims=True)
    maximum = spectra.max(axis=1, keepdims=True)
    return (spectra - minimum) / np.maximum(maximum - minimum, 1e-8)


def resample(spectra: np.ndarray, source_x: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    output = np.empty((len(spectra), len(target_x)), dtype=np.float32)
    for index, spectrum in enumerate(spectra):
        output[index] = np.interp(target_x, source_x, spectrum)
    return minmax(output).astype(np.float32)


def medoid_index(spectra: np.ndarray) -> int:
    normalized = minmax(spectra)
    centered = normalized - normalized.mean(axis=1, keepdims=True)
    centered /= np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
    centroid = centered.mean(axis=0)
    return int(np.argmax(centered @ centroid))


def write_dataset(path: Path, spectra: np.ndarray, x_axis: np.ndarray) -> None:
    with h5py.File(path.with_suffix(".h5"), "w") as handle:
        handle.create_dataset("spectra", data=spectra, compression="gzip", compression_opts=4)
        handle.create_dataset("x_axis", data=x_axis)
        handle.create_dataset(
            "physical_params",
            data=np.zeros((len(spectra), 7), dtype=np.float32),
        )
    path.write_text("OpenSpecy data are stored in the matching HDF5 file.\n", encoding="utf-8")


def build_training_pairs(
    metadata: pd.DataFrame,
    spectra: np.ndarray,
    identities: list[str],
    pairs_per_identity: int,
    rng: np.random.Generator,
    allowed_raman_indices: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ir_rows: list[np.ndarray] = []
    raman_rows: list[np.ndarray] = []
    records: list[dict[str, object]] = []

    for identity in identities:
        identity_rows = metadata.index[metadata["identity_key"] == identity].to_numpy()
        ir_indices = identity_rows[metadata.loc[identity_rows, "modality"].to_numpy() == "ir"]
        raman_indices = identity_rows[
            metadata.loc[identity_rows, "modality"].to_numpy() == "raman"
        ]
        if allowed_raman_indices is not None:
            raman_indices = np.array(
                [index for index in raman_indices if int(index) in allowed_raman_indices],
                dtype=int,
            )
        if len(raman_indices) == 0:
            continue
        candidate_pairs = np.array(
            [(int(ir_index), int(raman_index))
             for ir_index in ir_indices for raman_index in raman_indices],
            dtype=int,
        )
        rng.shuffle(candidate_pairs)

        for pair_index, (ir_index, raman_index) in enumerate(
            candidate_pairs[:pairs_per_identity]
        ):
            ir_rows.append(spectra[ir_index])
            raman_rows.append(spectra[raman_index])
            records.append(
                {
                    "identity_key": identity,
                    "pair_index": pair_index,
                    "ir_sample_name": metadata.at[ir_index, "sample_name"],
                    "raman_sample_name": metadata.at[raman_index, "sample_name"],
                }
            )

    return np.stack(ir_rows), np.stack(raman_rows), pd.DataFrame(records)


def select_within_identity_holdout(
    metadata: pd.DataFrame,
    spectra: np.ndarray,
    identities: list[str],
    test_fraction: float,
    rng: np.random.Generator,
    dedup_decimals: int,
) -> tuple[set[int], set[int], pd.DataFrame]:
    """Hold out entire duplicate groups while retaining each identity in training."""
    groups: list[tuple[str, list[int]]] = []
    group_count_by_identity: dict[str, int] = {}
    for identity in identities:
        rows = metadata.index[
            (metadata["identity_key"] == identity) & (metadata["modality"] == "raman")
        ].to_list()
        duplicate_groups: dict[bytes, list[int]] = {}
        for raw_index in rows:
            index = int(raw_index)
            key = np.round(spectra[index], decimals=dedup_decimals).tobytes()
            duplicate_groups.setdefault(key, []).append(index)
        identity_groups = [
            (identity, member_indices)
            for member_indices in duplicate_groups.values()
        ]
        groups.extend(identity_groups)
        group_count_by_identity[identity] = len(identity_groups)

    desired_test_count = max(1, round(len(groups) * test_fraction))
    candidate_group_ids = rng.permutation(len(groups))
    remaining_groups = dict(group_count_by_identity)
    test_group_ids: set[int] = set()
    for raw_group_id in candidate_group_ids:
        group_id = int(raw_group_id)
        identity, _ = groups[group_id]
        if remaining_groups[identity] <= 1:
            continue
        test_group_ids.add(group_id)
        remaining_groups[identity] -= 1
        if len(test_group_ids) >= desired_test_count:
            break

    if len(test_group_ids) < desired_test_count:
        raise ValueError(
            f"Could only hold out {len(test_group_ids)} Raman duplicate groups without "
            f"removing an identity from training; requested {desired_test_count}"
        )

    train_indices: set[int] = set()
    test_representatives: set[int] = set()
    split_records: list[dict[str, object]] = []
    for group_id, (identity, member_indices) in enumerate(groups):
        is_test = group_id in test_group_ids
        representative = int(member_indices[0])
        if is_test:
            test_representatives.add(representative)
        else:
            train_indices.update(member_indices)
        for index in member_indices:
            split_records.append(
                {
                    "identity_key": identity,
                    "raman_sample_name": metadata.at[index, "sample_name"],
                    "duplicate_group": group_id,
                    "split": (
                        "test_representative"
                        if is_test and index == representative
                        else "test_duplicate_excluded"
                        if is_test
                        else "train"
                    ),
                }
            )

    return train_indices, test_representatives, pd.DataFrame(split_records)


def build_within_identity_test_pairs(
    metadata: pd.DataFrame,
    spectra: np.ndarray,
    test_raman_indices: set[int],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Pair held-out Raman measurements with the identity's IR medoid."""
    ir_rows: list[np.ndarray] = []
    raman_rows: list[np.ndarray] = []
    records: list[dict[str, object]] = []

    for raman_index in sorted(test_raman_indices):
        identity = str(metadata.at[raman_index, "identity_key"])
        ir_indices = metadata.index[
            (metadata["identity_key"] == identity) & (metadata["modality"] == "ir")
        ].to_numpy()
        ir_index = int(ir_indices[medoid_index(spectra[ir_indices])])
        ir_rows.append(spectra[ir_index])
        raman_rows.append(spectra[raman_index])
        records.append(
            {
                "identity_key": identity,
                "ir_sample_name": metadata.at[ir_index, "sample_name"],
                "raman_sample_name": metadata.at[raman_index, "sample_name"],
                "split_type": "held_out_raman_measurement",
            }
        )

    return np.stack(ir_rows), np.stack(raman_rows), pd.DataFrame(records)


def build_medoid_pairs(
    metadata: pd.DataFrame,
    spectra: np.ndarray,
    identities: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ir_rows: list[np.ndarray] = []
    raman_rows: list[np.ndarray] = []
    records: list[dict[str, object]] = []

    for identity in identities:
        identity_rows = metadata.index[metadata["identity_key"] == identity].to_numpy()
        ir_indices = identity_rows[metadata.loc[identity_rows, "modality"].to_numpy() == "ir"]
        raman_indices = identity_rows[
            metadata.loc[identity_rows, "modality"].to_numpy() == "raman"
        ]
        ir_index = int(ir_indices[medoid_index(spectra[ir_indices])])
        raman_index = int(raman_indices[medoid_index(spectra[raman_indices])])
        ir_rows.append(spectra[ir_index])
        raman_rows.append(spectra[raman_index])
        records.append(
            {
                "identity_key": identity,
                "ir_sample_name": metadata.at[ir_index, "sample_name"],
                "raman_sample_name": metadata.at[raman_index, "sample_name"],
            }
        )

    return np.stack(ir_rows), np.stack(raman_rows), pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("datasets/openspecy_cov100"))
    parser.add_argument("--output_dir", type=Path, default=Path("datasets/openspecy_rruff_paired"))
    parser.add_argument(
        "--pair_key",
        type=str,
        default="rruffid",
        help="Metadata field defining a true cross-modal specimen pair",
    )
    parser.add_argument("--target_size", type=int, default=3600)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--pairs_per_identity", type=int, default=8)
    parser.add_argument(
        "--dedup_decimals",
        type=int,
        default=6,
        help="Decimal precision used to group duplicate normalized Raman spectra",
    )
    parser.add_argument(
        "--split_mode",
        choices=["identity_disjoint", "within_identity"],
        default="identity_disjoint",
        help=(
            "identity_disjoint holds out complete identities; within_identity holds "
            "out unique Raman measurements while retaining their identities in training"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test_fraction must be strictly between 0 and 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metadata = pd.read_csv(args.input_dir / "metadata.csv")
    if args.pair_key not in all_metadata:
        raise ValueError(f"Pair key {args.pair_key!r} is absent from the metadata")
    pair_key = all_metadata[args.pair_key].astype("string").str.strip().str.lower()
    metadata = all_metadata[pair_key.notna() & pair_key.ne("")].copy()
    metadata["identity_key"] = pair_key.loc[metadata.index]
    modality_count = metadata.groupby("identity_key")["modality"].nunique()
    paired_keys = modality_count[modality_count.eq(2)].index
    metadata = metadata[metadata["identity_key"].isin(paired_keys)].copy()

    with h5py.File(args.input_dir / "spectra.h5", "r") as handle:
        source_x = handle["wavenumber"][:].astype(np.float32)
        raw_spectra = handle["spectra"][:].astype(np.float32)
    if raw_spectra.shape == (len(source_x), len(all_metadata)):
        raw_spectra = raw_spectra.T
    if raw_spectra.shape != (len(all_metadata), len(source_x)):
        raise ValueError(
            f"Unexpected spectra shape {raw_spectra.shape}; expected "
            f"({len(all_metadata)}, {len(source_x)})"
        )

    if not np.isfinite(raw_spectra).all():
        raise ValueError("The input benchmark contains missing or non-finite intensities")

    target_x = np.linspace(
        float(source_x[0]),
        float(source_x[-1]),
        args.target_size,
        dtype=np.float32,
    )
    spectra = resample(raw_spectra, source_x, target_x)

    identities = sorted(metadata["identity_key"].unique())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(identities)

    if args.split_mode == "identity_disjoint":
        test_count = max(1, round(len(identities) * args.test_fraction))
        test_identities = sorted(identities[:test_count])
        train_identities = sorted(identities[test_count:])
        train_ir, train_raman, train_pairs = build_training_pairs(
            metadata,
            spectra,
            train_identities,
            args.pairs_per_identity,
            rng,
        )
        test_ir, test_raman, test_pairs = build_medoid_pairs(
            metadata,
            spectra,
            test_identities,
        )
        split = pd.DataFrame(
            {
                "identity_key": train_identities + test_identities,
                "split": ["train"] * len(train_identities) + ["test"] * len(test_identities),
            }
        )
    else:
        train_identities = sorted(identities)
        train_raman_indices, test_raman_indices, spectrum_split = (
            select_within_identity_holdout(
                metadata,
                spectra,
                train_identities,
                args.test_fraction,
                rng,
                args.dedup_decimals,
            )
        )
        train_ir, train_raman, train_pairs = build_training_pairs(
            metadata,
            spectra,
            train_identities,
            args.pairs_per_identity,
            rng,
            allowed_raman_indices=train_raman_indices,
        )
        test_ir, test_raman, test_pairs = build_within_identity_test_pairs(
            metadata,
            spectra,
            test_raman_indices,
        )
        test_identities = sorted(test_pairs["identity_key"].unique())
        split = pd.DataFrame(
            {
                "identity_key": train_identities,
                "split": [
                    "train_and_test" if identity in set(test_identities) else "train_only"
                    for identity in train_identities
                ],
            }
        )
        spectrum_split.to_csv(args.output_dir / "raman_spectrum_split.csv", index=False)

    write_dataset(args.output_dir / "train_ir.csv", train_ir, target_x)
    write_dataset(args.output_dir / "train_raman.csv", train_raman, target_x)
    write_dataset(args.output_dir / "test_ir.csv", test_ir, target_x)
    write_dataset(args.output_dir / "test_raman.csv", test_raman, target_x)
    train_pairs.to_csv(args.output_dir / "train_pairs.csv", index=False)
    test_pairs.to_csv(args.output_dir / "test_pairs.csv", index=False)

    split.to_csv(args.output_dir / "identity_split.csv", index=False)

    print(f"Split mode: {args.split_mode}")
    print(f"Train identities: {len(train_identities)}")
    print(f"Test identities: {len(test_identities)}")
    print(f"Training pairs: {len(train_ir)}")
    print(f"Test pairs: {len(test_ir)}")
    if args.split_mode == "within_identity":
        print("Raman spectrum split:")
        print(spectrum_split["split"].value_counts().to_string())
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
