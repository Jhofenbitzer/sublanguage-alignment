"""Dependency-free tests for synthesis preprocessing and planning logic."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from src.synthesis.common import write_json
from src.synthesis.create_upper_bound import file_seed, sample_disjoint_records
from src.synthesis.generate_biogpt import load_items as load_biogpt_items
from src.synthesis.generate_completion_based import load_items as load_completion_items
from src.synthesis.generate_prompt_based import build_generation_plan
from src.synthesis.generate_unigram import (
    exclude_protected_sentences,
    normalize_language_key,
    sample_maximum_equal_size,
    tokenize_text,
)
from src.synthesis.truncate_samples import truncate_by_fraction


class SynthesisTests(unittest.TestCase):
    def test_historical_truncation_keeps_30_percent_rounded_up(self) -> None:
        text = "one two three four five six seven eight nine ten eleven"
        self.assertEqual(truncate_by_fraction(text, 0.30, 1), "one two three four")

    def test_upper_bound_is_deterministic_and_disjoint(self) -> None:
        full = [{"id": str(index)} for index in range(20)]
        original = full[:5]
        seed = file_seed(42, "biomedical_de.json")
        first, _ = sample_disjoint_records(
            full,
            original,
            full_path=Path("full.json"),
            sample_path=Path("original.json"),
            sample_size=5,
            rng=random.Random(seed),
        )
        second, _ = sample_disjoint_records(
            full,
            original,
            full_path=Path("full.json"),
            sample_path=Path("original.json"),
            sample_size=5,
            rng=random.Random(seed),
        )
        self.assertEqual(first, second)
        self.assertFalse({item["id"] for item in first} & {item["id"] for item in original})

    def test_prompt_plan_preserves_only_historical_biomedical_ids(self) -> None:
        plan = build_generation_plan(540_000)
        self.assertEqual(len(plan), 270_000)
        self.assertEqual(plan[0], {"id": "gen_1", "language": "English", "source": "biomedical"})
        self.assertEqual({item["source"] for item in plan}, {"biomedical"})
        self.assertTrue(all(int(item["id"].split("_")[1]) % 2 == 1 for item in plan))

    def test_unigram_excludes_both_protected_sets(self) -> None:
        kept, reports = exclude_protected_sentences(
            ["real sentence", "upper sentence", "learning sentence"],
            {"real_sample": ["real sentence"], "upper_bound": ["upper sentence"]},
            language="de",
            minimum_coverage=1.0,
        )
        self.assertEqual(kept, ["learning sentence"])
        self.assertEqual([report.coverage for report in reports], [1.0, 1.0])

    def test_language_and_unicode_token_normalization(self) -> None:
        self.assertEqual(normalize_language_key("biomedical_de"), "de")
        self.assertEqual(normalize_language_key("de_test"), "de")
        self.assertEqual(normalize_language_key("pt-BR"), "pt-br")
        self.assertEqual(tokenize_text("β-blocker l’insuffisance"), ["β-blocker", "l'insuffisance"])

    def test_equal_size_sampling_is_dictionary_order_independent(self) -> None:
        first, first_n = sample_maximum_equal_size(
            {"en": ["a", "b", "c"], "de": ["d", "e"]},
            13,
        )
        second, second_n = sample_maximum_equal_size(
            {"de": ["d", "e"], "en": ["a", "b", "c"]},
            13,
        )
        self.assertEqual((first, first_n), (second, second_n))

    def test_model_generators_reject_general_records(self) -> None:
        record = {
            "id": "x",
            "language": "German",
            "language_tag": "de",
            "source": "general",
            "original": "Original text.",
            "truncated": "Original",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            write_json(path, [record])
            with self.assertRaisesRegex(ValueError, "Only biomedical"):
                load_biogpt_items(path)
            with self.assertRaisesRegex(ValueError, "Only biomedical"):
                load_completion_items(path)


if __name__ == "__main__":
    unittest.main()
