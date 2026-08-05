#!/usr/bin/env python3
"""Complete biomedical prefixes with Microsoft BioGPT-Large.

The historical generation settings are retained: seed 42, temperature 1.0,
top-p 0.9, 96 maximum new tokens, and right padding. The original submitted
script accepted the seed but did not apply it; this version initializes the
generation RNG and records all relevant settings in every output record.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

try:
    from .common import (
        collect_json_paths,
        model_short_name,
        read_json,
        require_biomedical,
        safe_name,
        write_json,
    )
except ImportError:  # Direct execution
    from common import (
        collect_json_paths,
        model_short_name,
        read_json,
        require_biomedical,
        safe_name,
        write_json,
    )


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "microsoft/BioGPT-Large"
MODEL_CONTEXT_LENGTH = 1024


def load_items(path: Path) -> list[dict[str, Any]]:
    """Load direct-child JSON arrays and reject non-biomedical inputs."""
    items: list[dict[str, Any]] = []
    for input_path in collect_json_paths(path):
        value = read_json(input_path)
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON array in {input_path}")
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"Entry {index} in {input_path} is not an object")
            require_biomedical(item, location=f"{input_path}, record {index}")
            if "truncated" not in item:
                raise ValueError(f"Missing 'truncated' at {input_path}, record {index}")
            if "language_tag" not in item and "language" not in item:
                raise ValueError(f"Missing language metadata at {input_path}, record {index}")
            copied = dict(item)
            copied.setdefault("source_file", input_path.name)
            copied.setdefault("doc_index", index)
            items.append(copied)
    if not items:
        raise ValueError(f"No biomedical evaluation records found in {path}")
    return items


def initialize_seed(seed: int, torch: Any, transformers: Any) -> None:
    """Initialize every RNG used by Hugging Face generation."""
    random.seed(seed)
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_batches(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    *,
    torch: Any,
    tqdm: Any,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Generate continuations while preserving each input prefix verbatim."""
    results: list[str] = []
    for start in tqdm(range(0, len(items), batch_size), desc="BioGPT generation"):
        batch = items[start : start + batch_size]
        prefixes = [str(item["truncated"]) for item in batch]
        encoded = tokenizer(
            prefixes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MODEL_CONTEXT_LENGTH - max_new_tokens,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        input_width = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        continuations = tokenizer.batch_decode(
            output_ids[:, input_width:],
            skip_special_tokens=True,
        )
        for prefix, continuation in zip(prefixes, continuations):
            continuation = continuation.strip()
            results.append(prefix.rstrip() + ((" " + continuation) if continuation else ""))
    return results


def output_path(base: Path, model_name: str, language: str) -> Path:
    return Path(f"{base}.{model_name}.{safe_name(language)}.biomedical.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output basename.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 0 < args.max_new_tokens < MODEL_CONTEXT_LENGTH:
        parser.error(f"--max-new-tokens must be between 1 and {MODEL_CONTEXT_LENGTH - 1}")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        import torch
        import transformers
        from tqdm import tqdm
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("BioGPT generation requires torch, transformers, and tqdm") from error

    initialize_seed(args.seed, torch, transformers)
    items = load_items(args.input)
    LOGGER.info("Loaded %d biomedical evaluation records", len(items))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"  # Historical BioGPT setting.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    generated = generate_batches(
        model,
        tokenizer,
        items,
        torch=torch,
        tqdm=tqdm,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    if len(generated) != len(items):
        raise AssertionError(f"Generated {len(generated)} outputs for {len(items)} inputs")

    name = model_short_name(args.model)
    groups: dict[str, list[dict[str, Any]]] = {}
    for item, text in zip(items, generated):
        language = str(item.get("language_tag") or item.get("language"))
        result = dict(item)
        result.update(
            synthesized=text,
            model=name,
            model_hf_id=args.model,
            forced_prefix_fallback=False,
            seed=args.seed,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        groups.setdefault(language, []).append(result)

    for language, group in sorted(groups.items()):
        path = output_path(args.output, name, language)
        write_json(path, group)
        LOGGER.info("Wrote %d records to %s", len(group), path)


if __name__ == "__main__":
    main()
