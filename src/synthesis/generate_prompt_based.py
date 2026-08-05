#!/usr/bin/env python3
"""
Generate prompt-based biomedical sentences with vLLM.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

try:
    from .common import model_short_name, stable_seed
except ImportError:  # Direct execution
    from common import model_short_name, stable_seed


LOGGER = logging.getLogger(__name__)
DEFAULT_TOTAL_CANDIDATES = 540_000
DEFAULT_TEST_CANDIDATES = 10
DEFAULT_CHUNK_SIZE = 512
PROMPT_VERSION = "biomedical-prompt-v1"

GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 1024
MAX_NEW_TOKENS = 96
MIN_NEW_TOKENS = 1
TEMPERATURE = 1.0
TOP_P = 0.9

LANGUAGES = (
    "English",
    "German",
    "Spanish",
    "French",
    "Italian",
    "Dutch",
    "Romanian",
    "Polish",
    "Swedish",
)
ORIGINAL_DOMAINS = ("general", "biomedical")
PLLUM_COMPAT_MODELS = {
    "cyfragovpl/pllum-8x7b-instruct",
    "cyfragovpl/pllum-8x7b-instruct-2412",
}


def configure_vllm_environment(model_id: str) -> None:
    """Set vLLM flags before importing vLLM."""
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if model_id.strip().casefold().rstrip("/") in PLLUM_COMPAT_MODELS:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def log_runtime_versions() -> None:
    LOGGER.info(
        "Runtime | torch=%s | vllm=%s | transformers=%s | tokenizers=%s | mistral-common=%s",
        *(package_version(name) for name in ("torch", "vllm", "transformers", "tokenizers", "mistral-common")),
    )


def build_prompt(item: dict[str, Any]) -> str:
    """Construct the biomedical-only prompt used for every language."""
    language = item["language"]
    return f"""Generate one realistic {language} sentence from the biomedical domain.
The sentence should resemble formal biomedical or regulatory medical prose found in official documents from the European Medicines Agency (EMA).
Rules:
- Output exactly one sentence.
- Use {language}.
- Keep the style realistic for the biomedical domain.
- Do not add explanations, labels, translations, bullet points, or quotation marks.
- End with sentence-final punctuation.
Sentence:"""


def clean_completion(text: str) -> str:
    """Apply one-sentence cleanup."""
    if not text:
        return ""
    output = text.strip()
    if (
        (output.startswith('"') and output.endswith('"'))
        or (output.startswith("“") and output.endswith("”"))
        or (output.startswith("'") and output.endswith("'"))
    ):
        output = output[1:-1].strip()
    output = re.sub(r"\s+", " ", output).strip()
    output = re.sub(r"^[-*•]\s*", "", output).strip()
    output = re.sub(r"^\d+[\).\s]+", "", output).strip()
    output = re.split(
        r"(?<=[.!?。！？])\s+(?=[A-ZÄÖÜÀ-Ý\"“'(\[])",
        output,
        maxsplit=1,
    )[0].strip()
    if output.endswith(("!", "?")):
        output = output[:-1].rstrip() + "."
    if output and not output.endswith((".", "。", "!", "?", "！", "？")):
        output += "."
    return output


def build_generation_plan(total_candidates: int) -> list[dict[str, Any]]:
    """Filter alternating-domain plan to biomedical records."""
    combinations = list(itertools.product(LANGUAGES, ORIGINAL_DOMAINS))
    plan: list[dict[str, Any]] = []
    for index in range(total_candidates):
        language, source = combinations[index % len(combinations)]
        if source == "biomedical":
            plan.append({"id": f"gen_{index}", "language": language, "source": source})
    return plan


def load_processed_ids(path: Path) -> set[str]:
    """Read valid IDs from an existing JSONL checkpoint."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Ignoring malformed checkpoint line %s:%d", path, line_number)
                continue
            if isinstance(record, dict) and isinstance(record.get("id"), str):
                ids.add(record["id"])
    return ids


def is_pllum(model_id: str) -> bool:
    return model_id.strip().casefold().startswith("cyfragovpl/pllum-")


def apply_pllum_compatibility(model_id: str) -> None:
    """Apply the narrowly scoped Transformers 5/vLLM tokenizer shim."""
    if model_id.strip().casefold().rstrip("/") not in PLLUM_COMPAT_MODELS:
        return
    try:
        from transformers.tokenization_mistral_common import MistralCommonBackend
    except (ImportError, AttributeError) as error:
        LOGGER.warning("PLLuM tokenizer shim unavailable: %r", error)
        return

    applied: list[str] = []
    if not hasattr(MistralCommonBackend, "is_fast"):
        setattr(MistralCommonBackend, "is_fast", True)
        applied.append("is_fast")
    if "convert_tokens_to_string" not in MistralCommonBackend.__dict__:
        def convert_tokens_to_string(_self: Any, tokens: list[str]) -> str:
            return "".join(tokens)
        setattr(MistralCommonBackend, "convert_tokens_to_string", convert_tokens_to_string)
        applied.append("convert_tokens_to_string")
    if applied:
        LOGGER.info("Applied PLLuM tokenizer compatibility: %s", ", ".join(applied))


def adjusted_resources(model_id: str, base_memory: float, base_chunk: int) -> tuple[float, int]:
    """Adjust resources for large and MoE models."""
    markers = ("420b", "119b", "80b", "70b", "8x7b", "14b", "12b")
    if any(marker in model_id.casefold() for marker in markers):
        return min(0.98, base_memory + 0.05), min(base_chunk, 64)
    return base_memory, base_chunk


def build_llm(
    LLM: Any,
    *,
    model_id: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
) -> Any:
    tokenizer_mode = "hf" if is_pllum(model_id) else "auto"
    apply_pllum_compatibility(model_id)
    return LLM(
        model=model_id,
        tokenizer=model_id,
        tokenizer_mode=tokenizer_mode,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="auto",
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        disable_custom_all_reduce=True,
        limit_mm_per_prompt={"image": 0},
    )


def sampling_params_for_chunk(
    SamplingParams: Any,
    items: list[dict[str, Any]],
    *,
    model_id: str,
    base_seed: int,
) -> list[Any]:
    """Give every record a stable seed independent of chunking and restarts."""
    return [
        SamplingParams(
            min_tokens=MIN_NEW_TOKENS,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stop=["\n"],
            seed=stable_seed(base_seed, model_id, item["id"], PROMPT_VERSION) % (2**32),
        )
        for item in items
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Hugging Face model ID.")
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--total-records",
        type=int,
        default=DEFAULT_TOTAL_CANDIDATES,
    )
    parser.add_argument("--test-records", type=int, default=DEFAULT_TEST_CANDIDATES)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.total_records <= 0 or args.test_records <= 0:
        parser.error("record counts must be positive")
    if args.tensor_parallel_size is not None and args.tensor_parallel_size <= 0:
        parser.error("--tensor-parallel-size must be positive")
    if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    configure_vllm_environment(args.model)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError("Prompt-based generation requires torch and vLLM") from error

    for logger_name in ("vllm", "vllm.engine", "vllm.executor", "vllm.worker", "vllm.core"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    log_runtime_versions()

    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_gpus <= 0:
        raise RuntimeError("No CUDA GPUs are visible")
    tensor_parallel_size = args.tensor_parallel_size or visible_gpus
    if tensor_parallel_size > visible_gpus:
        raise RuntimeError(
            f"tensor_parallel_size={tensor_parallel_size}, but only {visible_gpus} GPUs are visible"
        )

    candidate_count = args.test_records if args.test else args.total_records
    plan = build_generation_plan(candidate_count)
    if not plan:
        raise ValueError("Candidate count produced no biomedical records")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{model_short_name(args.model)}.jsonl"
    processed_ids = load_processed_ids(output_path)
    remaining = [item for item in plan if item["id"] not in processed_ids]
    if not remaining:
        LOGGER.info("All %d biomedical records are already present in %s", len(plan), output_path)
        return

    base_memory = args.gpu_memory_utilization or GPU_MEMORY_UTILIZATION
    memory, chunk_size = adjusted_resources(args.model, base_memory, args.chunk_size)
    LOGGER.info(
        "Starting | model=%s | biomedical_remaining=%d/%d | GPUs=%d | tensor_parallel=%d | "
        "chunk=%d | memory=%.3f | seed=%d",
        args.model,
        len(remaining),
        len(plan),
        visible_gpus,
        tensor_parallel_size,
        chunk_size,
        memory,
        args.seed,
    )
    llm = build_llm(
        LLM,
        model_id=args.model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=memory,
        enforce_eager=args.enforce_eager,
    )

    with output_path.open("a", encoding="utf-8") as output_file:
        for start in range(0, len(remaining), chunk_size):
            chunk = remaining[start : start + chunk_size]
            prompts = [build_prompt(item) for item in chunk]
            params = sampling_params_for_chunk(
                SamplingParams,
                chunk,
                model_id=args.model,
                base_seed=args.seed,
            )
            outputs = llm.generate(prompts, params, use_tqdm=False)
            if len(outputs) != len(chunk):
                raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(chunk)} prompts")

            for item, output in zip(chunk, outputs):
                raw_text = output.outputs[0].text if output.outputs else ""
                row = dict(item)
                row.update(
                    synthesized=clean_completion(raw_text),
                    raw_synthesized=raw_text,
                    model=model_short_name(args.model),
                    model_hf_id=args.model,
                    seed=args.seed,
                    record_seed=stable_seed(args.seed, args.model, item["id"], PROMPT_VERSION) % (2**32),
                    prompt_version=PROMPT_VERSION,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
            LOGGER.info("Checkpoint | processed_remaining=%d/%d", start + len(chunk), len(remaining))

    LOGGER.info("Done | output=%s", output_path)


if __name__ == "__main__":
    main()
