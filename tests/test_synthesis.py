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
from src.synthesis.sample_corpora import (
    detect_language_code_from_filename,
    downsample_records_per_source,
    is_acceptable_text,
    read_biomedical_txt,
    read_general_corpus_txt,
)
from src.synthesis.truncate_samples import truncate_by_fraction


class SynthesisTests(unittest.TestCase):
    def test_trunction_keeps_30_percent_rounded_up(self) -> None:
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

    def test_prompt_plan_preserves_only_biomedical_ids(self) -> None:
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

    def test_is_acceptable_text_filters_short_and_multi_sentence(self) -> None:
        self.assertTrue(is_acceptable_text("This is fine."))
        self.assertFalse(is_acceptable_text("Too short"))
        self.assertFalse(is_acceptable_text("One sentence. Two sentences."))

    def test_detect_language_code_from_filename(self) -> None:
        self.assertEqual(detect_language_code_from_filename(Path("de.txt")), "de")
        self.assertEqual(detect_language_code_from_filename(Path("deu_biomedical_corpus.txt")), "deu")
        self.assertEqual(detect_language_code_from_filename(Path("en_full.txt")), "en")

    def test_downsample_caps_each_source_independently_and_is_reproducible(self) -> None:
        records = [{"id": f"web:{i}", "source": "web"} for i in range(20)] + [
            {"id": f"wiki:{i}", "source": "wikipedia"} for i in range(5)
        ]
        first, did_downsample, stats = downsample_records_per_source(records, max_per_source=10, seed=7)
        second, _, _ = downsample_records_per_source(records, max_per_source=10, seed=7)
        self.assertTrue(did_downsample)
        self.assertEqual(first, second)
        self.assertEqual(stats["web"], (20, 10))
        self.assertEqual(stats["wikipedia"], (5, 5))
        self.assertEqual(len([r for r in first if r["source"] == "web"]), 10)
        self.assertEqual(len([r for r in first if r["source"] == "wikipedia"]), 5)

    def test_downsample_collapses_biomedical_to_a_single_source(self) -> None:
        records = [{"id": f"de:{i}", "source": "biomedical"} for i in range(100)]
        selected, did_downsample, stats = downsample_records_per_source(records, max_per_source=30, seed=42)
        self.assertTrue(did_downsample)
        self.assertEqual(len(selected), 30)
        self.assertEqual(stats, {"biomedical": (100, 30)})

    def test_read_biomedical_txt_builds_stable_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "de.txt"
            path.write_text("Erster Satz hier.\nZweiter Satz hier auch.\n", encoding="utf8")
            records = list(read_biomedical_txt(path))
            self.assertEqual([record["id"] for record in records], ["de:1", "de:2"])
            self.assertEqual({record["source"] for record in records}, {"biomedical"})
            self.assertEqual(records[0]["language"], "German")

    def test_general_corpus_ids_stay_unique_across_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            web_path = directory_path / "deu-de_web_2011_10K-sentences.txt"
            news_path = directory_path / "deu-de_newscrawl_2011_10K-sentences.txt"
            web_path.write_text(
                "1\tDies ist ein Testsatz.\n2\tNoch ein Testsatz hier.\n", encoding="utf8"
            )
            news_path.write_text(
                "1\tDies ist eine Nachricht heute.\n2\tNoch eine Nachricht hier.\n", encoding="utf8"
            )

            records = list(read_general_corpus_txt(web_path)) + list(read_general_corpus_txt(news_path))
            ids = [record["id"] for record in records]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {"web:1", "web:2", "newscrawl:1", "newscrawl:2"})


if __name__ == "__main__":
    unittest.main()
