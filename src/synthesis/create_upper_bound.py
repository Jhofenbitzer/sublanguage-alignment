#!/usr/bin/env python3
"""
Create the second, disjoint biomedical sample used as the upper bound.

The full language corpora are external and are not expected to be stored in the repository. For every top-level ``biomedical_*.json`` file, this script excludes IDs in the corresponding original sample and draws a second sample without replacement. The historical defaults are 30,000 records and seed 42.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
from pathlib import Path
from typing import Any

try:
    from .common import read_json, validate_unique_ids, write_json
except ImportError:  # Direct execution: python src/synthesis/create_upper_bound.py
    from common import read_json, validate_unique_ids, write_json


LOGGER = logging.getLogger(__name__)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array containing only record objects."""
    data = read_json(path)
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON array in {path}; found {type(data).__name__}.")
    if not all(isinstance(record, dict) for record in data):
        bad_index = next(i for i, record in enumerate(data) if not isinstance(record, dict))
        raise TypeError(f"Expected an object at index {bad_index} in {path}.")
    return data


def file_seed(base_seed: int, filename: str) -> int:
    """Reproduce the original per-file seed derivation exactly."""
    material = f"{base_seed}:{filename}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def sample_disjoint_records(
    full_records: list[dict[str, Any]],
    original_sample: list[dict[str, Any]],
    *,
    full_path: Path,
    sample_path: Path,
    sample_size: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Draw records whose IDs are absent from the original experimental sample."""
    full_ids = validate_unique_ids(full_records, location=str(full_path))
    original_ids = validate_unique_ids(original_sample, location=str(sample_path))

    unknown_ids = original_ids - full_ids
    if unknown_ids:
        raise ValueError(
            f"{sample_path} contains {len(unknown_ids)} IDs absent from {full_path}. "
            f"Examples: {sorted(unknown_ids)[:10]}"
        )

    available = [record for record in full_records if record["id"] not in original_ids]
    if len(available) < sample_size:
        raise ValueError(
            f"Cannot sample {sample_size:,} records from {full_path.name}: "
            f"only {len(available):,} remain after excluding {len(original_ids):,} IDs."
        )

    sampled = rng.sample(available, sample_size)
    sampled_ids = validate_unique_ids(sampled, location=f"sampled output for {full_path}")
    overlap = sampled_ids & original_ids
    if overlap:
        raise AssertionError(
            f"Internal error: upper-bound sample overlaps the original by {len(overlap)} IDs."
        )

    return sampled, {
        "n_full": len(full_records),
        "n_original": len(original_sample),
        "n_available": len(available),
        "n_sampled": len(sampled),
    }


def discover_source_files(input_dir: Path) -> list[Path]:
    """Find full biomedical corpora without descending into sample directories."""
    return sorted(path for path in input_dir.glob("biomedical_*.json") if path.is_file())


def process_directory(
    input_dir: Path,
    original_sample_dir: Path,
    output_dir: Path,
    *,
    sample_size: int,
    seed: int,
    overwrite: bool,
) -> None:
    source_paths = discover_source_files(input_dir)
    if not source_paths:
        raise FileNotFoundError(f"No top-level biomedical_*.json files found in {input_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = missing = existing = 0

    for source_path in source_paths:
        sample_path = original_sample_dir / source_path.name
        output_path = output_dir / source_path.name

        if not sample_path.is_file():
            LOGGER.warning("Skipping %s: missing original sample %s", source_path.name, sample_path)
            missing += 1
            continue
        if output_path.exists() and not overwrite:
            LOGGER.warning("Skipping existing output %s (use --overwrite to replace it)", output_path)
            existing += 1
            continue

        full_records = load_records(source_path)
        original_sample = load_records(sample_path)
        sampled, stats = sample_disjoint_records(
            full_records,
            original_sample,
            full_path=source_path,
            sample_path=sample_path,
            sample_size=sample_size,
            rng=random.Random(file_seed(seed, source_path.name)),
        )
        write_json(output_path, sampled)
        LOGGER.info(
            "Upper bound created | language_file=%s | full=%d | original=%d | "
            "available=%d | sampled=%d | output=%s",
            source_path.name,
            stats["n_full"],
            stats["n_original"],
            stats["n_available"],
            stats["n_sampled"],
            output_path,
        )
        processed += 1

    LOGGER.info(
        "Finished | processed=%d | skipped_missing_original=%d | skipped_existing=%d",
        processed,
        missing,
        existing,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("biomedical_per_language_0226"),
        help="External directory containing full biomedical_<language>.json corpora.",
    )
    parser.add_argument(
        "--original-sample-dir",
        "--existing-sample-dir",
        dest="original_sample_dir",
        type=Path,
        help="Original sample directory (default: <input-dir>/downsampled_30k).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Upper-bound directory (default: <input-dir>/downsampled_30k_second).",
    )
    parser.add_argument("--sample-size", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    original_dir = (
        args.original_sample_dir.resolve()
        if args.original_sample_dir is not None
        else input_dir / "downsampled_30k"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / "downsampled_30k_second"
    )
    if not original_dir.is_dir():
        raise NotADirectoryError(f"Original sample directory does not exist: {original_dir}")
    if output_dir in {input_dir, original_dir}:
        raise ValueError("Output directory must differ from the full corpus and original sample directories.")

    process_directory(
        input_dir,
        original_dir,
        output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
