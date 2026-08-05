#!/usr/bin/env python3
"""Create 30%-length prefixes from biomedical sample files.

The historical preprocessing normalized whitespace and retained
``ceil(number_of_words * 0.30)`` whitespace-delimited words, with at least one
word. Those choices are preserved because the resulting prefixes are part of
the experimental data contract.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    from .common import require_biomedical, write_json
except ImportError:  # Direct execution
    from common import require_biomedical, write_json


LOGGER = logging.getLogger(__name__)


def normalize_whitespace(text: str) -> str:
    """Reproduce the whitespace normalization used for the experiment."""
    return " ".join(text.strip().split())


def truncate_by_fraction(text: str, fraction: float, min_keep: int) -> str:
    """Keep the leading ceil(fraction * n) whitespace-delimited words."""
    normalized = normalize_whitespace(text)
    if not normalized:
        return ""
    words = normalized.split(" ")
    keep = min(len(words), max(min_keep, math.ceil(len(words) * fraction)))
    return " ".join(words[:keep])


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSON object/array or JSONL file, rejecting malformed record types."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}.")
            records.append(record)
        return records

    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        raise TypeError(f"Expected a JSON object or array in {path}.")
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise TypeError(f"Expected an object at index {index} in {path}.")
    return value


def iter_input_files(input_dir: Path, *, recursive: bool) -> Iterator[Path]:
    """Yield input files in deterministic relative-path order."""
    patterns = ("**/*.json", "**/*.jsonl") if recursive else ("*.json", "*.jsonl")
    paths = {path for pattern in patterns for path in input_dir.glob(pattern) if path.is_file()}
    yield from sorted(paths, key=lambda path: path.relative_to(input_dir).as_posix())


def process_file(
    input_path: Path,
    output_path: Path,
    *,
    text_field: str,
    fraction: float,
    min_keep: int,
) -> tuple[int, int]:
    """Transform one sample file and return (written, skipped-text-field) counts."""
    output: list[dict[str, Any]] = []
    skipped = 0

    for index, record in enumerate(read_records(input_path)):
        require_biomedical(record, location=f"{input_path}, record {index}")
        text = record.get(text_field)
        if not isinstance(text, str):
            skipped += 1
            continue

        original = normalize_whitespace(text)
        transformed = dict(record)
        transformed.pop(text_field, None)
        transformed["original"] = original
        transformed["truncated"] = truncate_by_fraction(original, fraction, min_keep)
        output.append(transformed)

    write_json(output_path, output)
    return len(output), skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", "--input_dir", dest="input_dir", type=Path, required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--text-field", "--text_field", dest="text_field", default="text")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.30,
        help="Fraction of whitespace-delimited words to keep (historical default: 0.30).",
    )
    parser.add_argument("--min-keep", "--min_keep", dest="min_keep", type=int, default=1)
    parser.add_argument("--indent", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be in (0, 1]")
    if args.min_keep <= 0:
        parser.error("--min-keep must be positive")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if args.recursive and (output_dir == input_dir or input_dir in output_dir.parents):
        raise ValueError("For recursive processing, --output-dir must be outside --input-dir.")

    inputs = list(iter_input_files(input_dir, recursive=args.recursive))
    if not inputs:
        raise FileNotFoundError(f"No JSON or JSONL files found in {input_dir}.")

    targets: dict[Path, Path] = {}
    for input_path in inputs:
        target = (output_dir / input_path.relative_to(input_dir)).with_suffix(".json")
        if target in targets:
            raise ValueError(f"Output collision: {targets[target]} and {input_path} both map to {target}.")
        targets[target] = input_path

    total_records = total_skipped = 0
    for output_path, input_path in targets.items():
        written, skipped = process_file(
            input_path,
            output_path,
            text_field=args.text_field,
            fraction=args.fraction,
            min_keep=args.min_keep,
        )
        total_records += written
        total_skipped += skipped
        LOGGER.info("Truncated %s -> %s | records=%d | skipped=%d", input_path, output_path, written, skipped)

    LOGGER.info(
        "Finished | files=%d | records=%d | skipped_missing_or_nontext=%d | fraction=%.2f",
        len(inputs),
        total_records,
        total_skipped,
        args.fraction,
    )


if __name__ == "__main__":
    main()
