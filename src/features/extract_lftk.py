#!/usr/bin/env python3
"""Extract LFTK features from generated and reference texts.

The extractor accepts JSON arrays, single JSON objects, and JSONL files. It
supports completion filenames containing date, model, language, and domain;
prompt-generation files named only by model; and reference files such as
``data/general/deu_test.json`` and ``data/biomedical/de_test.json``.

Records are grouped by date, model, and domain. Each group is written as a CSV
under ``<output-root>/<date>/<model>/``. Language-specific spaCy pipelines are
cached and reused across groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
BIOMEDICAL = "biomedical"
NO_DATE = "no_date"

LANGUAGE_ALIASES = {
    "german": "de",
    "de": "de",
    "deu": "de",
    "deu-de": "de",
    "ger": "de",
    "english": "en",
    "en": "en",
    "eng": "en",
    "eng-us": "en",
    "eng-gb": "en",
    "french": "fr",
    "fr": "fr",
    "fra": "fr",
    "fre": "fr",
    "spanish": "es",
    "es": "es",
    "spa": "es",
    "italian": "it",
    "it": "it",
    "ita": "it",
    "dutch": "nl",
    "nl": "nl",
    "nld": "nl",
    "dut": "nl",
    "polish": "pl",
    "pl": "pl",
    "pol": "pl",
    "romanian": "ro",
    "ro": "ro",
    "ron": "ro",
    "rum": "ro",
    "swedish": "sv",
    "sv": "sv",
    "swe": "sv",
    "chinese": "zh",
    "zh": "zh",
    "zho": "zh",
    "chi": "zh",
    "croatian": "hr",
    "hr": "hr",
    "hrv": "hr",
}

SPACY_MODEL_STEMS = {
    "de": "de_core_news",
    "en": "en_core_web",
    "fr": "fr_core_news",
    "es": "es_core_news",
    "it": "it_core_news",
    "nl": "nl_core_news",
    "pl": "pl_core_news",
    "ro": "ro_core_news",
    "sv": "sv_core_news",
    "zh": "zh_core_web",
    "hr": "hr_core_news",
}

GENERAL_DOMAIN_LABELS = {
    "general",
    "web",
    "newscrawl",
    "wikipedia",
    "news",
}

DATE_PREFIX_RE = re.compile(r"^(?:completions|generations)_(\d{8})\.(.+)$", re.IGNORECASE)
DOMAIN_LANGUAGE_RE = re.compile(r"^(biomedical|general)_([^.]+)$", re.IGNORECASE)
SAMPLE_RE = re.compile(r"^([a-z]{2,3})(?:_(?:test|upperbound))?$", re.IGNORECASE)
SOURCE_PREFIX_RE = re.compile(r"^biomedical_[^.]+\.(.+)$", re.IGNORECASE)

METADATA_FIELDS = [
    "model",
    "country",
    "source_file",
    "doc_index",
    "doc_id",
    "area",
    "subspecialty",
    "modality",
    "contributor_code",
    "forced_prefix_fallback",
]


@dataclass(frozen=True)
class FileInfo:
    """Metadata encoded in an input filename."""

    date: str
    model: str
    language: str
    domain: str


@dataclass
class LoadStats:
    """Counts collected while parsing one JSON or JSONL file."""

    valid_records: int = 0
    malformed_records: int = 0


@dataclass(frozen=True)
class PreparedRecord:
    """Text and metadata ready for language-specific feature extraction."""

    text: str
    language: str
    metadata: dict[str, Any]


def normalize_language_tag(language: str) -> str:
    """Map supported language names and ISO aliases to two-letter codes."""
    normalized = language.strip().casefold().replace("_", "-")
    return LANGUAGE_ALIASES.get(normalized, normalized)


def normalize_domain(domain: str) -> str:
    """Map an explicit domain label to biomedical or general."""
    return BIOMEDICAL if domain.strip().casefold() == BIOMEDICAL else "general"


def safe_path_component(value: str) -> str:
    """Convert metadata to a portable directory or filename component."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-")
    return safe or "unknown"


def parse_input_filename(path: Path) -> FileInfo:
    """Parse supported structured, model-only, and reference filenames."""
    if path.suffix.casefold() not in {".json", ".jsonl"}:
        raise ValueError(f"Expected a .json or .jsonl file: {path}")

    stem = path.name[: -len(path.suffix)]
    domain_language_match = DOMAIN_LANGUAGE_RE.fullmatch(stem)
    if domain_language_match:
        domain, language = domain_language_match.groups()
        path_parts = {part.casefold() for part in path.parts}
        if path_parts & {"upperbound", "upper_bound"}:
            model = "upper_bound"
        elif path_parts & {"biomedical", "general"}:
            model = "real"
        else:
            model = "upper_bound" if domain.casefold() == BIOMEDICAL else "real"
        return FileInfo(NO_DATE, model, language, normalize_domain(domain))

    sample_match = SAMPLE_RE.fullmatch(stem)
    path_parts = {part.casefold() for part in path.parts}
    sample_domain = BIOMEDICAL if "biomedical" in path_parts else "general" if "general" in path_parts else ""
    if sample_match and sample_domain:
        model = "upper_bound" if path_parts & {"upperbound", "upper_bound"} else "real"
        return FileInfo(NO_DATE, model, sample_match.group(1), sample_domain)

    date = NO_DATE
    body = stem
    date_match = DATE_PREFIX_RE.fullmatch(body)
    if date_match:
        date, body = date_match.groups()
    else:
        source_match = SOURCE_PREFIX_RE.fullmatch(body)
        if source_match:
            body = source_match.group(1)

    parts = body.rsplit(".", maxsplit=2)
    if len(parts) == 3:
        model, language, domain = parts
        domain_key = domain.strip().casefold()
        if date != NO_DATE or domain_key == BIOMEDICAL or domain_key in GENERAL_DOMAIN_LABELS:
            return FileInfo(date, model, language, normalize_domain(domain))

    # Prompt-based generation writes one JSONL file named after the model.
    return FileInfo(date, body, "", "")


def iter_input_files(input_dir: Path, *, recursive: bool) -> Iterable[Path]:
    """Yield JSON and JSONL files in deterministic relative-path order."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    pattern = "**/*" if recursive else "*"
    paths = [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.casefold() in {".json", ".jsonl"}
    ]
    yield from sorted(paths, key=lambda path: path.relative_to(input_dir).as_posix())


def _validate_json_records(value: Any, path: Path) -> list[dict[str, Any]]:
    """Validate a parsed JSON object or array and return record objects."""
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        raise TypeError(f"Expected a JSON object or array in {path}; found {type(value).__name__}")
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise TypeError(f"Expected an object at index {index} in {path}")
    return value


def load_records(
    path: Path,
    *,
    malformed_policy: str,
    max_malformed_records: int,
) -> tuple[list[dict[str, Any]], LoadStats]:
    """Load JSON or JSONL records and optionally skip malformed JSONL lines."""
    text = path.read_text(encoding="utf-8").strip()
    stats = LoadStats()
    if not text:
        return [], stats

    try:
        records = _validate_json_records(json.loads(text), path)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError(f"found {type(record).__name__}, expected object")
            except (json.JSONDecodeError, TypeError) as error:
                stats.malformed_records += 1
                if malformed_policy == "error":
                    raise ValueError(f"Malformed record at {path}:{line_number}: {error}") from error
                if stats.malformed_records > max_malformed_records:
                    raise ValueError(
                        f"Malformed-record limit exceeded in {path}: "
                        f"{stats.malformed_records} > {max_malformed_records}"
                    ) from error
                LOGGER.warning("Skipping malformed record at %s:%d: %s", path, line_number, error)
                continue
            records.append(record)

    stats.valid_records = len(records)
    return records, stats


def resolve_domain(record: dict[str, Any], file_info: FileInfo) -> str:
    """Resolve domain from record metadata, falling back to the filename."""
    source = record.get("source")
    if isinstance(source, str) and source.strip():
        return normalize_domain(source)
    if file_info.domain:
        return file_info.domain
    raise ValueError("Cannot determine record domain from 'source' or filename")


def resolve_model(record: dict[str, Any], file_info: FileInfo) -> str:
    """Resolve the model label from record metadata or the filename."""
    model = record.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return file_info.model


def resolve_language(record: dict[str, Any], file_info: FileInfo) -> str:
    """Resolve a record language and normalize it for spaCy model selection."""
    for field in ("language_tag", "language"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return normalize_language_tag(value)
    if file_info.language:
        return normalize_language_tag(file_info.language)
    raise ValueError("Cannot determine record language from metadata or filename")


def extract_text(record: dict[str, Any], text_fields: Sequence[str]) -> str | None:
    """Return the first non-empty string from the configured text fields."""
    for field in text_fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def make_metadata(
    record: dict[str, Any],
    *,
    model: str,
    language: str,
    input_path: Path,
    row_index: int,
) -> dict[str, Any]:
    """Build the metadata columns prepended to every feature row."""
    source_file = record.get("source_file")
    if not isinstance(source_file, str) or not source_file.strip():
        source_file = input_path.name
    document_id = record.get("id", row_index)
    return {
        "model": model,
        "country": language,
        "source_file": source_file,
        "doc_index": record.get("doc_index", row_index),
        "doc_id": str(document_id),
        "area": record.get("area", ""),
        "subspecialty": record.get("subspecialty", ""),
        "modality": record.get("modality", ""),
        "contributor_code": record.get("contributor_code", ""),
        "forced_prefix_fallback": record.get("forced_prefix_fallback", ""),
    }


def language_to_spacy_model(language: str, model_size: str) -> str:
    """Return the spaCy pipeline name for a supported language code."""
    normalized = normalize_language_tag(language)
    if model_size not in {"sm", "md", "lg"}:
        raise ValueError(f"Unsupported spaCy model size: {model_size}")
    try:
        stem = SPACY_MODEL_STEMS[normalized]
    except KeyError as error:
        raise ValueError(f"No spaCy model configured for language {language!r}") from error
    return f"{stem}_{model_size}"


def build_nlp(spacy: Any, model_name: str, disable: Sequence[str]) -> Any:
    """Load a spaCy pipeline with selected components disabled."""
    disabled = [component.strip() for component in disable if component.strip()]
    LOGGER.info("Loading spaCy model %s | disabled=%s", model_name, disabled)
    try:
        return spacy.load(model_name, disable=disabled)
    except OSError as error:
        raise RuntimeError(
            f"Could not load {model_name!r}. Install it with: python -m spacy download {model_name}"
        ) from error


def get_feature_keys(lftk: Any) -> list[str]:
    """Return the ordered list of feature keys exposed by LFTK."""
    keys = lftk.search_features(return_format="list_key")
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise TypeError("LFTK did not return a list of feature keys")
    LOGGER.info("Loaded %d LFTK feature keys", len(keys))
    return keys


def extract_features(
    lftk: Any,
    tqdm: Any,
    nlp: Any,
    texts: list[str],
    feature_keys: list[str],
    *,
    n_process: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Run spaCy in batches and extract all configured LFTK features per document."""
    features: list[dict[str, Any]] = []
    docs = nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
    for document_index, document in enumerate(
        tqdm(docs, total=len(texts), desc=f"LFTK (batch={batch_size}, workers={n_process})")
    ):
        extractor = lftk.Extractor(docs=document)
        try:
            values = extractor.extract(features=feature_keys)
        except TypeError:
            values = extractor.extract()
        if not isinstance(values, dict):
            raise TypeError(
                f"LFTK returned {type(values).__name__} for document {document_index}; expected dict"
            )
        features.append(values)
    return features


def write_error_counts(path: Path, counts: dict[tuple[str, str], int]) -> None:
    """Write malformed-record counts by model and filename language."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "language", "error_count"])
        writer.writeheader()
        for (model, language), count in sorted(counts.items()):
            writer.writerow({"model": model, "language": language, "error_count": count})


def process_input_directory(
    input_dir: Path,
    output_root: Path,
    *,
    lftk: Any,
    spacy: Any,
    tqdm: Any,
    n_process: int,
    batch_size: int,
    spacy_disable: Sequence[str],
    model_size: str,
    language_filter: str,
    text_fields: Sequence[str],
    recursive: bool,
    malformed_policy: str,
    max_malformed_records: int,
    flat_output: bool = False,
) -> None:
    """Extract features from all selected records in an input directory."""
    input_paths = list(iter_input_files(input_dir, recursive=recursive))
    if not input_paths:
        raise FileNotFoundError(f"No JSON or JSONL files found in {input_dir}")

    requested_language = (
        normalize_language_tag(language_filter) if language_filter != "all" else "all"
    )
    grouped: dict[tuple[str, str, str], list[PreparedRecord]] = {}
    error_counts: dict[tuple[str, str], int] = {}
    missing_text = language_filtered = 0

    for input_path in input_paths:
        file_info = parse_input_filename(input_path)
        file_language = normalize_language_tag(file_info.language) if file_info.language else "unknown"
        error_key = (file_info.model, file_language)
        error_counts.setdefault(error_key, 0)

        records, stats = load_records(
            input_path,
            malformed_policy=malformed_policy,
            max_malformed_records=max_malformed_records,
        )
        error_counts[error_key] += stats.malformed_records

        for row_index, record in enumerate(records):
            domain = resolve_domain(record, file_info)
            language = resolve_language(record, file_info)
            if requested_language != "all" and language != requested_language:
                language_filtered += 1
                continue
            text = extract_text(record, text_fields)
            if text is None:
                missing_text += 1
                continue
            model = resolve_model(record, file_info)
            metadata = make_metadata(
                record,
                model=model,
                language=language,
                input_path=input_path,
                row_index=row_index,
            )
            grouped.setdefault((file_info.date, model, domain), []).append(
                PreparedRecord(text, language, metadata)
            )

    output_root.mkdir(parents=True, exist_ok=True)
    write_error_counts(output_root / "error_count.csv", error_counts)
    if not grouped:
        raise ValueError("No usable records matched the extraction settings")

    if flat_output:
        suffix = f"_{requested_language}" if requested_language != "all" else ""
        flat_paths: dict[Path, tuple[str, str, str]] = {}
        for date, model, domain in grouped:
            safe_model = safe_path_component(model)
            candidate = output_root / f"features_{safe_model}_{domain}{suffix}.csv"
            collision = flat_paths.get(candidate)
            if collision is not None:
                raise ValueError(
                    f"--flat-output would overwrite {candidate}: groups "
                    f"{collision} and {(date, model, domain)} both resolve to "
                    "the same filename. Drop --flat-output, or invoke once "
                    "per model/date/language so groups can't collide."
                )
            flat_paths[candidate] = (date, model, domain)

    feature_keys = get_feature_keys(lftk)
    nlp_cache: dict[str, Any] = {}
    total_written = 0

    for (date, model, domain), records in sorted(grouped.items()):
        by_language: dict[str, list[PreparedRecord]] = {}
        for record in records:
            by_language.setdefault(record.language, []).append(record)

        safe_model = safe_path_component(model)
        output_dir = output_root if flat_output else output_root / date / safe_model
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{requested_language}" if requested_language != "all" else ""
        output_path = output_dir / f"features_{safe_model}_{domain}{suffix}.csv"

        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=METADATA_FIELDS + feature_keys,
                extrasaction="ignore",
            )
            writer.writeheader()
            for language, language_records in by_language.items():
                spacy_model = language_to_spacy_model(language, model_size)
                if spacy_model not in nlp_cache:
                    nlp_cache[spacy_model] = build_nlp(spacy, spacy_model, spacy_disable)
                values = extract_features(
                    lftk,
                    tqdm,
                    nlp_cache[spacy_model],
                    [record.text for record in language_records],
                    feature_keys,
                    n_process=n_process,
                    batch_size=batch_size,
                )
                for record, record_features in zip(language_records, values, strict=True):
                    writer.writerow({**record.metadata, **record_features})
                    total_written += 1
        LOGGER.info("Wrote %d records to %s", len(records), output_path)

    LOGGER.info(
        "Finished | feature_rows=%d | missing_text=%d | language_filtered=%d",
        total_written,
        missing_text,
        language_filtered,
    )


def parse_text_fields(specification: str) -> tuple[str, ...]:
    """Parse a comma-separated, ordered list of record text fields."""
    fields = tuple(field.strip() for field in specification.split(",") if field.strip())
    if not fields:
        raise ValueError("At least one text field is required")
    return fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/features/completion_features"),
    )
    parser.add_argument("--n-process", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--spacy-disable", default="textcat,entity_linker")
    parser.add_argument("--spacy-model-size", choices=["sm", "md", "lg"], default="lg")
    parser.add_argument("--language-filter", default="all")
    parser.add_argument(
        "--text-fields",
        default="synthesized,text,original,sentence",
        help=(
            "Ordered fields containing text (default: synthesized,text,original,sentence)."
        ),
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help=(
            "Write features_<model>_<domain>[_<lang>].csv directly under "
            "--output-root instead of nesting under <date>/<model>/. Only "
            "safe when one invocation covers a single date/model/domain "
            "group, e.g. one model processed one language at a time into "
            "its own data/features/models/<feature-family>/ directory; "
            "raises if two groups would collide on the same filename."
        ),
    )
    parser.add_argument(
        "--malformed-record-policy",
        choices=["error", "skip"],
        default="skip",
    )
    parser.add_argument("--max-malformed-records", type=int, default=30_000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.n_process <= 0:
        parser.error("--n-process must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_malformed_records < 0:
        parser.error("--max-malformed-records cannot be negative")
    try:
        args.text_fields = parse_text_fields(args.text_fields)
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    try:
        import lftk
        import spacy
        from tqdm.auto import tqdm
    except ImportError as error:
        raise RuntimeError("Feature extraction requires lftk, spaCy, and tqdm") from error

    disabled = [component.strip() for component in args.spacy_disable.split(",") if component.strip()]
    process_input_directory(
        args.input_dir,
        args.output_root,
        lftk=lftk,
        spacy=spacy,
        tqdm=tqdm,
        n_process=args.n_process,
        batch_size=args.batch_size,
        spacy_disable=disabled,
        model_size=args.spacy_model_size,
        language_filter=args.language_filter,
        text_fields=args.text_fields,
        recursive=args.recursive,
        malformed_policy=args.malformed_record_policy,
        max_malformed_records=args.max_malformed_records,
        flat_output=args.flat_output,
    )


if __name__ == "__main__":
    main()
