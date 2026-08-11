#!/usr/bin/env python3
"""Build per-language JSON corpora from raw general and biomedical text files.

This script can:
1) read general reference corpora sentence files (Wikipedia/Newscrawl/Web) and
   write one downsampled JSON file per language, and
2) read biomedical plain-text corpora and write both the full per-language
   JSON corpus and the original downsampled (<=30k per language) sample that
   `create_upper_bound.py` expects as its `--original-sample-dir`.

Behaviour is configured via a YAML file. A minimal example configuration
(e.g. at `config/sample_corpora.yml`) could look like:

    general_input_dir: /path/to/general_raw
    general_output_dir: /path/to/general_per_language
    general_max_per_source: 10000
    general_seed: 13

    txt_input_dir: /path/to/biomedical_raw
    txt_output_dir: /path/to/biomedical_per_language
    txt_max_per_source: 30000
    txt_seed: 13

Loading the YAML config requires PyYAML; the per-record parsing and sampling
logic below has no such dependency and is covered by the dependency-free
suite in `tests/test_synthesis.py`.
"""
from __future__ import annotations

import argparse
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from .common import validate_unique_ids, write_json
except ImportError:  # Direct execution: python src/synthesis/sample_corpora.py
    from common import validate_unique_ids, write_json


LOGGER = logging.getLogger(__name__)

# --- language code -> human-readable language name (for JSON output) ---
LANGUAGE_CODE_TO_NAME: Dict[str, str] = {
    "deu": "German",
    "de": "German",
    "ell": "Greek",
    "el": "Greek",
    "eng": "English",
    "en": "English",
    "fra": "French",
    "fr": "French",
    "ita": "Italian",
    "it": "Italian",
    "hrv": "Croatian",
    "hr": "Croatian",
    "nld": "Dutch",
    "nl": "Dutch",
    "pol": "Polish",
    "pl": "Polish",
    "ron": "Romanian",
    "ro": "Romanian",
    "spa": "Spanish",
    "es": "Spanish",
    "swe": "Swedish",
    "sv": "Swedish",
    "zho": "Chinese",
    "zh": "Chinese",
}

# --- text quality filters ---
_SENT_SPLIT_RE = re.compile(r"[.!?]+")


def language_code_to_name(code: str) -> str:
    """Map ISO-639-3(-like) codes to human-readable language names.

    Falls back to the original code if it is unrecognized.
    """
    return LANGUAGE_CODE_TO_NAME.get(code.strip().lower(), code)


def normalize_group_name(name: str) -> str:
    """Normalize a language code for safe file names."""
    name = name.strip().lower()
    for ch in (" ", "/", "\\"):
        name = name.replace(ch, "_")
    return name


def _count_tokens(text: str) -> int:
    """Count whitespace-delimited tokens (simple, robust)."""
    return len(text.split())


def _count_sentences(text: str) -> int:
    """Approximate sentence count based on terminal punctuation.

    Counts non-empty segments after splitting on ., !, ? runs. Intentionally
    lightweight (no heavy NLP deps) since this only gates corpus hygiene.
    """
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    return len(parts)


def is_acceptable_text(text: str, *, min_tokens: int = 3, max_sentences: int = 1) -> bool:
    """Return True if `text` passes basic corpus hygiene constraints.

    Constraints:
    - at least `min_tokens` tokens
    - no more than `max_sentences` sentences
    """
    if not text:
        return False
    if _count_tokens(text) < min_tokens:
        return False
    if _count_sentences(text) > max_sentences:
        return False
    return True


# --- general corpus (Wikipedia/Newscrawl/Web) helpers ---

@dataclass(frozen=True)
class GeneralCorpusFileInfo:
    language: str       # base language used for grouping, e.g. "deu", "zho"
    language_tag: str   # full tag from filename, e.g. "deu-de", "zho-cn"
    source: str         # one of: "wikipedia", "newscrawl", "web"


_GENERAL_CORPUS_RE = re.compile(
    r"^(?P<lang>[a-z]{3}(?:-[a-z]{2,5})?)_(?P<src>wikipedia|newscrawl|web)(-.*)*_.*-sentences\.txt$",
    flags=re.IGNORECASE,
)


def parse_general_corpus_filename(path: Path) -> GeneralCorpusFileInfo:
    """Extract language and source type from a general corpus filename."""
    m = _GENERAL_CORPUS_RE.match(path.name)
    if not m:
        raise ValueError(
            "General corpus file name does not match expected pattern "
            "'<lang>_<wikipedia|newscrawl|web>_...-sentences.txt': "
            f"{path.name}"
        )
    lang_tag = m.group("lang").lower()
    lang_base = lang_tag.split("-", 1)[0]
    src = m.group("src").lower()
    return GeneralCorpusFileInfo(language=lang_base, language_tag=lang_tag, source=src)


def read_general_corpus_txt(path: Path) -> Iterable[Dict[str, Any]]:
    """Stream sentence records from a general corpus *.txt file.

    Expected format per line: `<id>\t<sentence>`. Each source file numbers
    sentences independently starting at 1, so the emitted `id` is qualified
    by source (`<source>:<line-id>`) to stay unique once wikipedia/newscrawl/
    web are combined into one per-language file.
    """
    info = parse_general_corpus_filename(path)
    with path.open("r", encoding="utf8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" in line:
                left, text = line.split("\t", 1)
            else:
                # Fallback: some corpora may be space-separated; treat first token as id.
                parts = line.split(maxsplit=1)
                left, text = (parts[0], "") if len(parts) == 1 else (parts[0], parts[1])
            left = left.strip()
            text = text.strip()
            if not text:
                continue
            if not is_acceptable_text(text, min_tokens=3, max_sentences=1):
                continue

            yield {
                "language": language_code_to_name(info.language),
                "language_tag": info.language_tag,
                "source": info.source,
                "id": f"{info.source}:{left or line_no}",
                "text": text,
            }


def build_general_per_language(
    input_dir: Path,
    *,
    file_glob: str = "*-sentences.txt",
) -> Dict[str, List[Dict[str, Any]]]:
    """Read Wikipedia/Newscrawl/Web sentence files and group them per language."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    paths = sorted(input_dir.glob(file_glob))
    if not paths:
        LOGGER.warning("No general corpus files found in %s with glob %r", input_dir, file_glob)
        return groups

    for p in paths:
        try:
            info = parse_general_corpus_filename(p)
        except ValueError:
            LOGGER.warning(
                "Skipping general corpus file with unexpected name (won't be sampled): %s", p
            )
            continue
        LOGGER.info(
            "Reading general corpus: %s (lang=%s, tag=%s, source=%s)",
            p, info.language, info.language_tag, info.source,
        )
        for rec in read_general_corpus_txt(p):
            groups.setdefault(info.language, []).append(rec)

    return groups


def write_general_per_language_json(
    groups: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    *,
    normalize_names: bool = True,
    max_per_source: int = 10000,
    seed: int = 13,
) -> None:
    """Write one JSON array per language into `output_dir`."""
    for lang, records in groups.items():
        if not records:
            continue
        fname = normalize_group_name(lang) if normalize_names else lang
        out_path = output_dir / f"general_{fname}.json"

        selected_records, did_downsample, stats = downsample_records_per_source(
            records, max_per_source=max_per_source, seed=seed,
        )
        for src_key, (before, after) in stats.items():
            if before != after:
                LOGGER.info(
                    "Downsampled general lang=%s source=%s: %d -> %d (seed=%d)",
                    lang, src_key, before, after, int(seed),
                )

        validate_unique_ids(selected_records, location=str(out_path))
        LOGGER.info(
            "Writing %d general sentences to %s%s",
            len(selected_records), out_path, " (capped per source)" if did_downsample else "",
        )
        write_json(out_path, selected_records)


# --- biomedical corpus TXT helpers ---

def detect_language_code_from_filename(path: Path) -> str:
    """Best-effort language code detection from a biomedical file name.

    Supports both 2-letter (ISO-639-1) and 3-letter (ISO-639-3) codes.
    Tries, in order: a clean leading alpha token of the stem, a `_`/`-`
    delimited leading token, and finally a standalone alpha token anywhere
    in the stem. Falls back to "unknown".
    """
    stem = path.stem.lower()

    if len(stem) >= 3 and stem[:3].isalpha():
        return stem[:3]
    if len(stem) >= 2 and stem[:2].isalpha():
        return stem[:2]

    for sep in ("_", "-"):
        tok = stem.split(sep, 1)[0]
        if len(tok) in (2, 3) and tok.isalpha():
            return tok

    m = re.search(r"\b([a-z]{3})\b", stem)
    if m:
        return m.group(1)
    m = re.search(r"\b([a-z]{2})\b", stem)
    if m:
        return m.group(1)

    return "unknown"


def read_biomedical_txt(path: Path) -> Iterable[Dict[str, Any]]:
    """Stream sentence records from a biomedical *.txt file (one sentence per line).

    `id` is built from `<file_stem>:<line_no>` to stay unique across the
    per-language file set.
    """
    lang_code = detect_language_code_from_filename(path)
    doc_id = path.stem

    with path.open("r", encoding="utf8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            if not is_acceptable_text(text, min_tokens=3, max_sentences=1):
                continue

            yield {
                "language": language_code_to_name(lang_code),
                "language_tag": lang_code,
                "source": "biomedical",
                "id": f"{doc_id}:{line_no}",
                "text": text,
            }


def build_biomedical_per_language(
    input_dir: Path,
    *,
    file_glob: str = "*.txt",
) -> Dict[str, List[Dict[str, Any]]]:
    """Read biomedical *.txt files and group them per language."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    paths = sorted(input_dir.glob(file_glob))
    if not paths:
        LOGGER.warning("No biomedical files found in %s with glob %r", input_dir, file_glob)
        return groups

    for p in paths:
        if not p.is_file():
            continue
        lang_code = detect_language_code_from_filename(p)
        if lang_code == "unknown":
            LOGGER.warning("Could not detect a language code for %s; grouping under 'unknown'", p)
        LOGGER.info("Reading biomedical text: %s (lang=%s)", p, lang_code)
        for rec in read_biomedical_txt(p):
            groups.setdefault(lang_code, []).append(rec)

    return groups


def write_biomedical_per_language_json(
    groups: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    *,
    normalize_names: bool = True,
    max_per_source: int = 30000,
    seed: int = 13,
) -> None:
    """Write the full per-language corpus, plus a downsampled (<=30k) sample.

    The downsampled sample is written to `output_dir/downsampled_30k/`, which
    is `create_upper_bound.py`'s default `--original-sample-dir`.
    """
    for lang, records in groups.items():
        if not records:
            continue
        fname = normalize_group_name(lang) if normalize_names else lang
        out_path = output_dir / f"biomedical_{fname}.json"

        validate_unique_ids(records, location=str(out_path))
        LOGGER.info("Writing %d biomedical docs to %s", len(records), out_path)
        write_json(out_path, records)

        if max_per_source <= 0:
            continue

        down_records, did_downsample, stats = downsample_records_per_source(
            records, max_per_source=max_per_source, seed=seed,
        )
        if not did_downsample:
            continue

        down_path = output_dir / "downsampled_30k" / f"biomedical_{fname}.json"
        for src_key, (before, after) in stats.items():
            if before != after:
                LOGGER.info(
                    "Downsampled biomedical lang=%s source=%s: %d -> %d (seed=%d)",
                    lang, src_key, before, after, int(seed),
                )
        LOGGER.info("Writing %d downsampled biomedical docs to %s", len(down_records), down_path)
        write_json(down_path, down_records)


# --- downsampling helper (shared by both domains) ---

def downsample_records_per_source(
    records: List[Dict[str, Any]],
    *,
    max_per_source: int,
    seed: int,
) -> tuple[List[Dict[str, Any]], bool, Dict[str, tuple[int, int]]]:
    """Downsample records so that each `source` occurs at most `max_per_source` times.

    Sampling is done per source with a fixed RNG seed for reproducibility.
    Biomedical records all share `source="biomedical"`, so this reduces to a
    single seeded sample of up to `max_per_source` records for that domain.

    Returns
    -------
    selected_records:
        The possibly downsampled records, in original order.
    did_downsample:
        True if any source had to be downsampled.
    stats:
        Mapping source -> (before, after) counts.
    """
    if max_per_source <= 0:
        return records, False, {}

    by_source: Dict[str, List[tuple[int, Dict[str, Any]]]] = {}
    for idx, rec in enumerate(records):
        src = rec.get("source")
        src_key = src.strip() if isinstance(src, str) and src.strip() else "unknown"
        by_source.setdefault(src_key, []).append((idx, rec))

    rng = random.Random(int(seed))
    selected: List[tuple[int, Dict[str, Any]]] = []
    did_downsample = False
    stats: Dict[str, tuple[int, int]] = {}

    for src_key, items in by_source.items():
        before = len(items)
        after = before
        if before > max_per_source:
            did_downsample = True
            items = rng.sample(items, k=max_per_source)
            after = len(items)
        stats[src_key] = (before, after)
        selected.extend(items)

    selected.sort(key=lambda t: t[0])
    return [rec for _, rec in selected], did_downsample, stats


# --- config-driven entrypoint ---

def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration. Requires PyYAML."""
    import yaml

    with path.open("r", encoding="utf8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must be a mapping, got {type(config)}")
    return config


def run(config_path: Path) -> None:
    """Run the corpus-sampling pipeline based on a YAML config."""
    config = load_config(config_path)

    general_input_dir_val = config.get("general_input_dir")
    if general_input_dir_val:
        general_input_dir = Path(general_input_dir_val)
        general_output_dir = Path(config.get("general_output_dir", "general_per_language"))
        general_file_glob = str(config.get("general_file_glob", "*-sentences.txt"))
        general_normalize_names = bool(config.get("general_normalize_language_names", True))
        general_max_per_source = int(config.get("general_max_per_source", 10000))
        general_seed = int(config.get("general_seed", 13))

        LOGGER.info("Loading general corpora from %s", general_input_dir)
        general_groups = build_general_per_language(general_input_dir, file_glob=general_file_glob)
        LOGGER.info("Found %d languages in general corpora", len(general_groups))
        write_general_per_language_json(
            general_groups,
            general_output_dir,
            normalize_names=general_normalize_names,
            max_per_source=general_max_per_source,
            seed=general_seed,
        )
    else:
        LOGGER.info(
            "No general corpus processing configured (set 'general_input_dir' in the YAML config to enable)."
        )

    txt_input_dir_val = config.get("txt_input_dir")
    if txt_input_dir_val:
        txt_input_dir = Path(txt_input_dir_val)
        txt_output_dir = Path(config.get("txt_output_dir", "biomedical_per_language"))
        txt_file_glob = str(config.get("txt_file_glob", "*.txt"))
        txt_normalize_names = bool(config.get("txt_normalize_language_names", True))
        txt_max_per_source = int(config.get("txt_max_per_source", 30000))
        txt_seed = int(config.get("txt_seed", 13))

        LOGGER.info("Loading biomedical texts from %s", txt_input_dir)
        biomedical_groups = build_biomedical_per_language(txt_input_dir, file_glob=txt_file_glob)
        LOGGER.info("Found %d languages in biomedical texts", len(biomedical_groups))
        write_biomedical_per_language_json(
            biomedical_groups,
            txt_output_dir,
            normalize_names=txt_normalize_names,
            max_per_source=txt_max_per_source,
            seed=txt_seed,
        )
    else:
        LOGGER.info(
            "No biomedical processing configured (set 'txt_input_dir' in the YAML config to enable)."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-language JSON corpora (and the original biomedical "
            "downsample) from raw general and biomedical text files, based on "
            "a YAML configuration."
        )
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=Path("config/sample_corpora.yml"),
        help="Path to YAML configuration file (default: config/sample_corpora.yml)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    run(args.config)


if __name__ == "__main__":
    main()
