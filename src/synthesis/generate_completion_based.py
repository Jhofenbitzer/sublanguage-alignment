#!/usr/bin/env python3
"""Complete truncated biomedical sentences with a collection of vLLM models.

Inputs must be JSON arrays containing ``language``, ``language_tag``,
``source``, ``id``, ``original``, and ``truncated``. Only records whose source
is explicitly ``biomedical`` are accepted.

The historical prompt says that the prefix is "exactly one third" of the
original. The actual preprocessing retained 30% of whitespace-delimited words,
rounded up. The prompt text is intentionally preserved because changing it
would change model behavior; see the synthesis README for this discrepancy.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .common import (
        collect_json_paths,
        model_short_name,
        read_json,
        require_biomedical,
        safe_name,
        stable_seed,
        write_json,
    )
    from .generate_prompt_based import (
        PLLUM_COMPAT_MODELS,
        apply_pllum_compatibility,
        is_pllum,
    )
except ImportError:  # Direct execution
    from common import (
        collect_json_paths,
        model_short_name,
        read_json,
        require_biomedical,
        safe_name,
        stable_seed,
        write_json,
    )
    from generate_prompt_based import (
        PLLUM_COMPAT_MODELS,
        apply_pllum_compatibility,
        is_pllum,
    )


LOGGER = logging.getLogger(__name__)
PROMPT_VERSION = "historical-completion-v1"

DEFAULT_MODELS = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "meta-llama/Llama-3.3-70B-Instruct",
    "minilingua-ai/MiniLingua-1b-Instruct",
    "google/gemma-3-12b-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-27b-it",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "swiss-ai/Apertus-8B-Instruct-2509",
    "swiss-ai/Apertus-70B-Instruct-2509",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "mistralai/Ministral-3-14B-Instruct-2512",
    "google/medgemma-4b-it",
    "google/medgemma-27b-text-it",
    "BioMistral/BioMistral-7B",
    "m42-health/Llama3-Med42-70B",
    "LSX-UniWue/LLaMmlein_1B",
    "LSX-UniWue/LLaMmlein_7B",
    "croissantllm/CroissantLLMChat-v0.1",
    "sapienzanlp/Minerva-7B-instruct-v1.0",
    "BramVanroy/fietje-2-instruct",
    "CYFRAGOVPL/Llama-PLLuM-8B-instruct",
    "CYFRAGOVPL/PLLuM-12B-instruct",
    "CYFRAGOVPL/PLLuM-12B-nc-instruct",
    "CYFRAGOVPL/PLLuM-8x7B-instruct",
    "CYFRAGOVPL/Llama-PLLuM-70B-instruct",
    "OpenLLM-Ro/RoGemma-7b-Instruct",
    "OpenLLM-Ro/RoGemma2-9b-Instruct",
    "OpenLLM-Ro/RoLlama3-8b-Instruct",
    "OpenLLM-Ro/RoLlama3.1-8b-Instruct",
    "OpenLLM-Ro/RoLlama2-7b-Instruct",
    "OpenLLM-Ro/RoMistral-7b-Instruct",
    "BSC-LT/salamandra-2b-instruct",
    "BSC-LT/salamandra-7b-instruct",
    "AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct",
    "AI-Sweden-Models/gpt-sw3-1.3b-instruct",
    "AI-Sweden-Models/gpt-sw3-20b-instruct",
)

BATCH_SIZE = 64
DTYPE = "bfloat16"
GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN = 1024
MAX_NEW_TOKENS = 96
MIN_NEW_TOKENS = 1
TEMPERATURE = 1.0
TOP_P = 0.9
TENSOR_PARALLEL_SIZE = 4


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str


def build_prompt(item: dict[str, Any]) -> str:
    """Return the original completion prompt verbatim for a biomedical item."""
    language = item["language"]
    truncated = item["truncated"]
    return f"""You are doing a text completion task for a {language} biomedical sentence.  Given a truncated sentence prefix, produce one coherent full sentence that could plausibly be the original. The length of the truncated sentence is exactly one third of the original sentence length.

Please, follow these rules during completion:
- Output exactly ONE sentence.
- The output must start with the truncated text VERBATIM (character-for-character).
- Keep STYLE consistent with the domain (biomedical) and language ({language}).
- Keep the continuation REALISTIC.
- End with a single PERIOD.
- Output ONLY the completed sentence.

Now complete:
Truncated: {truncated}
Completed:"""


def clean_completion(
    text: str | None,
    truncated: str,
    *,
    item_id: str | None,
    model: ModelSpec,
) -> tuple[str, bool]:
    """Apply the historical cleanup and report forced-prefix fallbacks."""
    if text is None:
        LOGGER.warning("Received no generation for item %s", item_id)
        return "", False
    output = text.strip()
    if (
        (output.startswith('"') and output.endswith('"'))
        or (output.startswith("“") and output.endswith("”"))
    ):
        output = output[1:-1].strip()

    if "\n" in output:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        output = next(
            (line for line in reversed(lines) if truncated and line.startswith(truncated)),
            lines[-1] if lines else output,
        )
    output = re.sub(r"\s+", " ", output).strip()

    forced = False
    if truncated and not output.startswith(truncated):
        forced = True
        position = output.find(truncated)
        output = output[position:].strip() if position >= 0 else f"{truncated} {output}".strip()
        LOGGER.warning(
            "Forced prefix | item=%s | model=%s | prefix=%r | output=%r",
            item_id,
            model.hf_id,
            truncated[:80],
            output[:200],
        )
    return output, forced


def should_enforce_eager(model_id: str) -> bool:
    """Preserve the Qwen3-Next CUDA-graph stability workaround."""
    return "qwen3-next" in model_id.casefold()


def initialize_llm(LLM: Any, model_id: str, **kwargs: Any) -> tuple[ModelSpec, Any]:
    """Initialize vLLM while tolerating version-specific log-suppression flags."""
    apply_pllum_compatibility(model_id)
    kwargs.setdefault("tokenizer", model_id)
    kwargs.setdefault("tokenizer_mode", "hf" if is_pllum(model_id) else "auto")
    if "enforce_eager" not in kwargs and should_enforce_eager(model_id):
        kwargs["enforce_eager"] = True
        LOGGER.warning("Enabling eager mode for %s", model_id)
    for name in ("disable_log_stats", "disable_log_requests", "disable_tqdm"):
        kwargs.setdefault(name, True)

    removable = ["disable_tqdm", "disable_log_requests", "disable_log_stats"]
    while True:
        try:
            return ModelSpec(model_short_name(model_id), model_id), LLM(model=model_id, **kwargs)
        except TypeError as error:
            message = str(error)
            option = next(
                (name for name in removable if name in kwargs and (name in message or "unexpected keyword" in message)),
                None,
            )
            if option is None:
                raise
            kwargs.pop(option)


def record_sampling_params(
    SamplingParams: Any,
    items: list[dict[str, Any]],
    *,
    model_id: str,
    seed: int,
) -> list[Any]:
    """Create stable per-record sampling parameters independent of batching."""
    return [
        SamplingParams(
            min_tokens=MIN_NEW_TOKENS,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            # Historical behavior: terminal punctuation stops generation and is
            # not retained by vLLM's default stop-string behavior.
            stop=["\n", ".", "!", "?"],
            seed=stable_seed(seed, model_id, item["id"], PROMPT_VERSION) % (2**32),
        )
        for item in items
    ]


def run_model(
    llm: Any,
    SamplingParams: Any,
    items: list[dict[str, Any]],
    prompts: list[str],
    *,
    model_id: str,
    seed: int,
    batch_size: int,
    show_progress: bool,
    tqdm: Any,
) -> list[str]:
    """Generate in batches with stable record-level seeds."""
    iterator: Any = range(0, len(prompts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=(len(prompts) + batch_size - 1) // batch_size, desc=model_short_name(model_id))
    started = time.perf_counter()
    completions: list[str] = []

    for start in iterator:
        batch_prompts = prompts[start : start + batch_size]
        batch_items = items[start : start + batch_size]
        params = record_sampling_params(SamplingParams, batch_items, model_id=model_id, seed=seed)
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            outputs = llm.generate(batch_prompts, params)
        if len(outputs) != len(batch_items):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(batch_items)} prompts")
        completions.extend(output.outputs[0].text if output.outputs else "" for output in outputs)

    elapsed = time.perf_counter() - started
    LOGGER.info("Generated %d records with %s in %.2fs", len(items), model_id, elapsed)
    return completions


def load_items(path: Path) -> list[dict[str, Any]]:
    """Load biomedical completion inputs and preserve file provenance."""
    required = {"language", "language_tag", "source", "id", "original", "truncated"}
    items: list[dict[str, Any]] = []
    for input_path in collect_json_paths(path):
        value = read_json(input_path)
        if not isinstance(value, list):
            raise ValueError(f"Input JSON must be an array: {input_path}")
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object at {input_path}, record {index}")
            missing = sorted(required - item.keys())
            if missing:
                raise ValueError(f"Missing {missing} at {input_path}, record {index}")
            require_biomedical(item, location=f"{input_path}, record {index}")
            copied = dict(item)
            copied.setdefault("source_file", input_path.name)
            copied.setdefault("doc_index", index)
            items.append(copied)
    if not items:
        raise ValueError(f"No biomedical records found in {path}")
    return items


def model_output_path(base: Path, model: ModelSpec, language: str) -> Path:
    suffix = base.suffix + f".{model.name}.{safe_name(language)}.biomedical.json"
    return base.with_suffix(suffix)


def huggingface_model_cache(model_id: str) -> Path | None:
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    if "/" not in model_id:
        return None
    organization, name = model_id.split("/", 1)
    return hub / f"models--{organization}--{name}"


def cleanup_model_cache(model_id: str) -> None:
    """Delete one model cache only when the user explicitly requests it."""
    path = huggingface_model_cache(model_id)
    if path is None:
        LOGGER.warning("Cannot safely identify a cache directory for model %r; skipping cleanup", model_id)
        return
    if path.exists():
        shutil.rmtree(path)
        LOGGER.info("Deleted model cache %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output basename.")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Generate with one model; repeat for multiple models (default: historical model list).",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--tensor-parallel-size", type=int, default=TENSOR_PARALLEL_SIZE)
    parser.add_argument("--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--cleanup-model-cache", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.tensor_parallel_size <= 0:
        parser.error("--tensor-parallel-size must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    models = tuple(args.models) if args.models else DEFAULT_MODELS
    os.environ.setdefault("VLLM_DISABLE_TQDM", "1")
    os.environ.setdefault("VLLM_NO_TQDM", "1")
    os.environ.setdefault("VLLM_TQDM_DISABLE", "1")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if any(model.strip().casefold().rstrip("/") in PLLUM_COMPAT_MODELS for model in models):
        # Required before importing vLLM so the scoped tokenizer shim is also
        # visible to PLLuM-8x7B's engine core.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        import torch
        from tqdm import tqdm
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError("Completion generation requires torch, tqdm, and vLLM") from error

    for logger_name in ("vllm", "vllm.engine", "vllm.executor", "vllm.worker", "vllm.core"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_gpus < args.tensor_parallel_size:
        raise RuntimeError(
            f"Requested {args.tensor_parallel_size} GPUs, but only {visible_gpus} are visible. "
            "Set CUDA_VISIBLE_DEVICES or lower --tensor-parallel-size."
        )

    items = load_items(args.input)
    prompts = [build_prompt(item) for item in items]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Loaded %d biomedical records; models=%d", len(items), len(models))

    for model_id in models:
        spec, llm = initialize_llm(
            LLM,
            model_id,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=DTYPE,
            max_model_len=MAX_MODEL_LEN,
        )
        raw_completions = run_model(
            llm,
            SamplingParams,
            items,
            prompts,
            model_id=model_id,
            seed=args.seed,
            batch_size=args.batch_size,
            show_progress=not args.no_progress,
            tqdm=tqdm,
        )

        groups: dict[str, list[dict[str, Any]]] = {}
        for item, raw in zip(items, raw_completions):
            synthesized, forced = clean_completion(
                raw,
                str(item["truncated"]),
                item_id=str(item["id"]),
                model=spec,
            )
            result = dict(item)
            result.update(
                synthesized=synthesized,
                raw_synthesized=raw,
                model=spec.name,
                model_hf_id=spec.hf_id,
                forced_prefix_fallback=forced,
                seed=args.seed,
                record_seed=stable_seed(args.seed, model_id, item["id"], PROMPT_VERSION) % (2**32),
                prompt_version=PROMPT_VERSION,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            groups.setdefault(str(item["language_tag"]), []).append(result)

        for language, group in sorted(groups.items()):
            path = model_output_path(args.output, spec, language)
            write_json(path, group)
            LOGGER.info("Wrote %d records to %s", len(group), path)

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
        if args.cleanup_model_cache:
            cleanup_model_cache(spec.hf_id)


if __name__ == "__main__":
    main()
