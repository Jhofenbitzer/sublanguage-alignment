#!/usr/bin/env python3
"""Generate biomedical continuations from language-specific unigram models.

The learning corpus excludes both protected evaluation sets before balancing
languages: (1) the real experimental sample and (2) the upper-bound sample.
The script verifies each exclusion set independently and refuses to run when
matching coverage is below the requested threshold.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .common import (
        TEXT_FIELDS,
        collect_json_paths,
        read_json,
        require_biomedical,
        safe_name,
        stable_seed,
        write_json,
    )
except ImportError:  # Direct execution
    from common import (
        TEXT_FIELDS,
        collect_json_paths,
        read_json,
        require_biomedical,
        safe_name,
        stable_seed,
        write_json,
    )


LOGGER = logging.getLogger(__name__)
APOSTROPHES = {"'", "\u2019", "\u02bc"}
HYPHENS = {"-", "\u2010", "\u2011"}
WHITESPACE_RE = re.compile(r"\s+")


def extract_sentence(entry: Any) -> str | None:
    """Extract a non-empty sentence from a string or supported record object."""
    if isinstance(entry, str):
        text = entry
    elif isinstance(entry, dict):
        text = next(
            (entry[field] for field in TEXT_FIELDS if isinstance(entry.get(field), str)),
            None,
        )
        if text is None:
            return None
    else:
        return None
    stripped = text.strip()
    return stripped or None


def load_sentence_file(path: Path) -> list[str]:
    """Load usable sentence strings from one corpus JSON array."""
    value = read_json(path)
    if not isinstance(value, list):
        raise ValueError(f"Corpus JSON must be an array: {path}")

    sentences: list[str] = []
    skipped = 0
    for entry in value:
        sentence = extract_sentence(entry)
        if sentence is None:
            skipped += 1
        else:
            sentences.append(sentence)
    if skipped:
        LOGGER.warning("Skipped %d records without usable text in %s", skipped, path)
    if not sentences:
        raise ValueError(f"No usable sentences found in {path}")
    return sentences


def normalize_language_key(value: str) -> str:
    """Normalize corpus stems and metadata while retaining language subtags.

    Examples: ``biomedical_de`` -> ``de``, ``de_test`` -> ``de``, and
    ``pt-BR`` -> ``pt-br``.
    """
    key = unicodedata.normalize("NFKC", value).strip().casefold()
    key = re.sub(r"^biomedical(?:[-_\s]+)?", "", key)
    key = re.sub(r"(?:[-_\s]+)?(?:upper[-_\s]?bound|test)$", "", key)
    key = re.sub(r"[_\s]+", "-", key)
    key = re.sub(r"[^a-z0-9-]+", "", key)
    key = re.sub(r"-+", "-", key).strip("-")
    return key


def load_corpora_by_language(directory: Path) -> dict[str, list[str]]:
    """Load one direct-child JSON corpus per normalized language key."""
    corpora: dict[str, list[str]] = {}
    for path in collect_json_paths(directory):
        language = normalize_language_key(path.stem)
        if not language:
            raise ValueError(f"Could not derive a language key from {path.name}")
        if language in corpora:
            raise ValueError(f"Duplicate language key {language!r} in {directory}")
        corpora[language] = load_sentence_file(path)
        LOGGER.info("Loaded corpus | language=%s | sentences=%d | file=%s", language, len(corpora[language]), path)
    return corpora


def is_letter_or_number(character: str) -> bool:
    return unicodedata.category(character).startswith(("L", "N"))


def tokenize_text(text: str) -> list[str]:
    """Tokenize Unicode words, retaining internal apostrophes and hyphens."""
    normalized = unicodedata.normalize("NFKC", text)
    tokens: list[str] = []
    current: list[str] = []

    for index, character in enumerate(normalized):
        category = unicodedata.category(character)
        if is_letter_or_number(character):
            current.append(character)
            continue
        if category.startswith("M") and current:
            current.append(character)
            continue
        if character in APOSTROPHES | HYPHENS:
            has_next_word = index + 1 < len(normalized) and is_letter_or_number(normalized[index + 1])
            if current and has_next_word:
                current.append("'" if character in APOSTROPHES else "-")
                continue
        if current:
            tokens.append("".join(current).casefold())
            current = []
    if current:
        tokens.append("".join(current).casefold())
    return tokens


def sentence_fingerprint(text: str) -> str:
    """Normalize Unicode and whitespace for conservative text-level exclusion."""
    normalized = unicodedata.normalize("NFKC", text)
    return WHITESPACE_RE.sub(" ", normalized).strip()


@dataclass(frozen=True)
class ProtectionStats:
    label: str
    total: int
    unique: int
    matched_unique: int
    matching_train_records: int

    @property
    def coverage(self) -> float:
        return 1.0 if self.unique == 0 else self.matched_unique / self.unique


def exclude_protected_sentences(
    training: Sequence[str],
    protected_sets: Mapping[str, Sequence[str]],
    *,
    language: str,
    minimum_coverage: float,
) -> tuple[list[str], list[ProtectionStats]]:
    """Verify each protected set against the original training pool, then exclude their union."""
    training_keys = [sentence_fingerprint(sentence) for sentence in training]
    training_unique = set(training_keys)
    protected_union: set[str] = set()
    reports: list[ProtectionStats] = []

    for label, sentences in protected_sets.items():
        keys = {sentence_fingerprint(sentence) for sentence in sentences}
        keys.discard("")
        matched = training_unique & keys
        report = ProtectionStats(
            label=label,
            total=len(sentences),
            unique=len(keys),
            matched_unique=len(matched),
            matching_train_records=sum(key in keys for key in training_keys),
        )
        reports.append(report)
        LOGGER.info(
            "Protected-set check | language=%s | set=%s | total=%d | unique=%d | "
            "matched_unique=%d | coverage=%.4f | matching_train_records=%d",
            language,
            label,
            report.total,
            report.unique,
            report.matched_unique,
            report.coverage,
            report.matching_train_records,
        )
        if report.coverage < minimum_coverage:
            raise RuntimeError(
                f"Protected-set coverage too low for {language!r}/{label}: "
                f"{report.coverage:.4%} < {minimum_coverage:.4%}."
            )
        protected_union.update(keys)

    kept = [sentence for sentence, key in zip(training, training_keys) if key not in protected_union]
    remaining_overlap = {sentence_fingerprint(sentence) for sentence in kept} & protected_union
    if remaining_overlap:
        raise AssertionError(f"Protected-set exclusion failed for {language}")
    if not kept:
        raise RuntimeError(f"No training sentences remain for {language} after exclusion")

    LOGGER.info(
        "Protected sets excluded | language=%s | before=%d | removed=%d | after=%d",
        language,
        len(training),
        len(training) - len(kept),
        len(kept),
    )
    return kept, reports


def align_and_exclude_corpora(
    training: Mapping[str, list[str]],
    real_sample: Mapping[str, list[str]],
    upper_bound: Mapping[str, list[str]],
    *,
    minimum_coverage: float,
) -> dict[str, list[str]]:
    """Require identical language coverage and exclude both evaluation samples."""
    expected = set(training)
    for label, corpus in (("real_sample", real_sample), ("upper_bound", upper_bound)):
        if set(corpus) != expected:
            raise ValueError(
                f"Language mismatch for {label}: missing={sorted(expected - set(corpus))}, "
                f"unexpected={sorted(set(corpus) - expected)}"
            )

    filtered: dict[str, list[str]] = {}
    for language in sorted(training):
        filtered[language], _ = exclude_protected_sentences(
            training[language],
            {"real_sample": real_sample[language], "upper_bound": upper_bound[language]},
            language=language,
            minimum_coverage=minimum_coverage,
        )
    return filtered


def sample_maximum_equal_size(
    sentences_by_language: Mapping[str, list[str]],
    seed: int,
) -> tuple[dict[str, list[str]], int]:
    """Sample the largest possible equal sentence count across languages."""
    if not sentences_by_language:
        raise ValueError("No language corpora available")
    target = min(map(len, sentences_by_language.values()))
    if target <= 0:
        raise ValueError("At least one language has no training sentences")

    sampled: dict[str, list[str]] = {}
    for language in sorted(sentences_by_language):
        sentences = sentences_by_language[language]
        sampled[language] = (
            list(sentences)
            if len(sentences) == target
            else random.Random(stable_seed(seed, f"learning-subset|{language}")).sample(sentences, target)
        )
        LOGGER.info("Balanced training | language=%s | available=%d | selected=%d", language, len(sentences), target)
    return sampled, target


@dataclass(frozen=True)
class UnigramDistribution:
    population: tuple[str, ...]
    cumulative_weights: tuple[float, ...]
    vocabulary_size: int
    token_count: int


def build_unigram_distribution(
    sentences: Iterable[str],
    alpha: float,
    *,
    language: str,
) -> UnigramDistribution:
    counter: Counter[str] = Counter()
    sentence_count = 0
    for sentence_count, sentence in enumerate(sentences, start=1):
        counter.update(tokenize_text(sentence))
        if sentence_count % 100_000 == 0:
            LOGGER.info("Counter progress | language=%s | sentences=%d | tokens=%d", language, sentence_count, counter.total())
    if not counter:
        raise RuntimeError(f"Empty unigram vocabulary for {language}")

    population = tuple(counter)
    cumulative = tuple(itertools.accumulate(float(counter[token]) ** alpha for token in population))
    distribution = UnigramDistribution(population, cumulative, len(population), counter.total())
    LOGGER.info(
        "Unigram built | language=%s | sentences=%d | vocabulary=%d | tokens=%d | alpha=%.4f",
        language,
        sentence_count,
        distribution.vocabulary_size,
        distribution.token_count,
        alpha,
    )
    return distribution


def load_evaluation_items(input_path: Path) -> list[dict[str, Any]]:
    """Load and validate biomedical evaluation items."""
    items: list[dict[str, Any]] = []
    for path in collect_json_paths(input_path):
        value = read_json(path)
        if not isinstance(value, list):
            raise ValueError(f"Evaluation JSON must be an array: {path}")
        for index, entry in enumerate(value):
            if not isinstance(entry, dict):
                raise ValueError(f"Evaluation entry {index} is not an object in {path}")
            require_biomedical(entry, location=f"{path}, record {index}")
            if "truncated" not in entry:
                raise ValueError(f"Missing 'truncated' at {path}, record {index}")
            if "language_tag" not in entry and "language" not in entry:
                raise ValueError(f"Missing language metadata at {path}, record {index}")
            item = dict(entry)
            item.setdefault("source_file", path.name)
            item.setdefault("doc_index", index)
            items.append(item)
    if not items:
        raise ValueError(f"No evaluation items found in {input_path}")
    return items


def candidate_language_keys(item: Mapping[str, Any]) -> list[str]:
    candidates: list[str] = []
    for field in ("language_tag", "language"):
        if item.get(field) is None:
            continue
        key = normalize_language_key(str(item[field]))
        if key:
            candidates.extend((key, key.split("-", 1)[0]))
    return list(dict.fromkeys(candidates))


def resolve_language_key(item: Mapping[str, Any], available: Mapping[str, Any]) -> str:
    for candidate in candidate_language_keys(item):
        if candidate in available:
            return candidate
    raise KeyError(
        f"Could not map id={item.get('id')!r}, language_tag={item.get('language_tag')!r}, "
        f"language={item.get('language')!r}; available={sorted(available)}"
    )


def item_rng(seed: int, item: Mapping[str, Any], language: str) -> random.Random:
    """Reproduce the original order-independent per-item seed derivation."""
    return random.Random(
        stable_seed(seed, language, item.get("id", ""), item.get("truncated", ""))
    )


def output_path_for_group(base: Path, language: str, source: str) -> Path:
    return Path(f"{base}.biounigram.{safe_name(language)}.{safe_name(source)}.json")


def generate_outputs(
    items: Sequence[dict[str, Any]],
    distributions: Mapping[str, UnigramDistribution],
    *,
    output_base: Path,
    alpha: float,
    seed: int,
    learning_sentences: int,
) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in items:
        language = resolve_language_key(item, distributions)
        prefix = str(item["truncated"])
        prefix_length = len(tokenize_text(prefix))
        continuation_length = 2 * prefix_length
        sampled = item_rng(seed, item, language).choices(
            distributions[language].population,
            cum_weights=distributions[language].cumulative_weights,
            k=continuation_length,
        ) if continuation_length else []

        synthesized = prefix.rstrip()
        if sampled:
            synthesized += (" " if synthesized else "") + " ".join(sampled)
        synthesized = synthesized.rstrip().rstrip(".!?") + "."

        result = dict(item)
        result.update(
            synthesized=synthesized,
            model="biounigram",
            alpha=alpha,
            sampling_language=language,
            prefix_token_length=prefix_length,
            sampled_token_length=continuation_length,
            learning_sentences_per_language=learning_sentences,
            seed=seed,
            protected_sets=["real_sample", "upper_bound"],
        )
        groups.setdefault((language, "biomedical"), []).append(result)

    for (language, source), group in sorted(groups.items()):
        path = output_path_for_group(output_base, language, source)
        write_json(path, group)
        LOGGER.info("Wrote output | file=%s | records=%d", path, len(group))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Biomedical truncated JSON file or directory.")
    parser.add_argument("--train-dir", type=Path, required=True, help="External learning corpus directory.")
    parser.add_argument("--test-dir", type=Path, required=True, help="Real experimental sample directory.")
    parser.add_argument("--upper-bound-dir", type=Path, required=True, help="Disjoint upper-bound sample directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output basename.")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-exclusion-coverage", type=float, default=1.0)
    args = parser.parse_args()
    if args.alpha <= 0:
        parser.error("--alpha must be positive")
    if not 0 <= args.min_exclusion_coverage <= 1:
        parser.error("--min-exclusion-coverage must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    items = load_evaluation_items(args.input)
    training = load_corpora_by_language(args.train_dir)
    real_sample = load_corpora_by_language(args.test_dir)
    upper_bound = load_corpora_by_language(args.upper_bound_dir)
    filtered = align_and_exclude_corpora(
        training,
        real_sample,
        upper_bound,
        minimum_coverage=args.min_exclusion_coverage,
    )
    balanced, learning_sentences = sample_maximum_equal_size(filtered, args.seed)
    distributions = {
        language: build_unigram_distribution(sentences, args.alpha, language=language)
        for language, sentences in sorted(balanced.items())
    }
    generate_outputs(
        items,
        distributions,
        output_base=args.output,
        alpha=args.alpha,
        seed=args.seed,
        learning_sentences=learning_sentences,
    )


if __name__ == "__main__":
    main()
