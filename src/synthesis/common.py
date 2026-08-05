"""Shared, dependency-free helpers for biomedical synthesis scripts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


BIOMEDICAL_SOURCE = "biomedical"
TEXT_FIELDS: tuple[str, ...] = ("original", "text", "sentence")


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON document."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    """Write readable UTF-8 JSON and create the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def collect_json_paths(path: Path) -> list[Path]:
    """Return one JSON file or the sorted direct-child JSON files in a directory."""
    if path.is_file():
        if path.suffix.casefold() != ".json":
            raise ValueError(f"Expected a .json file: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    paths = sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.casefold() == ".json"
    )
    if not paths:
        raise ValueError(f"No direct-child .json files found in: {path}")
    return paths


def require_biomedical(item: Mapping[str, Any], *, location: str) -> None:
    """Fail closed when a generation input is not explicitly biomedical."""
    source = item.get("source")
    normalized = str(source).strip().casefold() if source is not None else ""
    if normalized != BIOMEDICAL_SOURCE:
        raise ValueError(
            f"Only biomedical records are allowed; found source={source!r} at {location}."
        )


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 64-bit seed from a base seed and record identifiers."""
    material = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def safe_name(value: str) -> str:
    """Convert metadata used in output filenames to a portable lowercase name."""
    return re.sub(r"[^a-z0-9_-]+", "-", value.strip().casefold()).strip("-") or "unknown"


def model_short_name(model_id: str) -> str:
    """Reproduce the original lowercase Hugging Face repository label."""
    return model_id.rstrip("/").split("/")[-1].lower().replace(" ", "_")


def validate_unique_ids(records: Sequence[Mapping[str, Any]], *, location: str) -> set[str]:
    """Return validated record IDs, rejecting missing, invalid, or duplicate values."""
    ids: list[str] = []
    for index, record in enumerate(records):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(
                f"Record {index} at {location} must have a non-empty string 'id'; "
                f"found {record_id!r}."
            )
        ids.append(record_id)

    unique_ids = set(ids)
    if len(ids) != len(unique_ids):
        seen: set[str] = set()
        duplicates: list[str] = []
        for record_id in ids:
            if record_id in seen:
                duplicates.append(record_id)
            seen.add(record_id)
        raise ValueError(
            f"Duplicate IDs at {location}: {sorted(set(duplicates))[:10]} "
            f"({len(ids)} records, {len(unique_ids)} unique IDs)."
        )
    return unique_ids
