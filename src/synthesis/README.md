# Biomedical synthesis pipeline

These scripts reproduce the biomedical side of the synthesis workflow. They
deliberately reject records marked `source: general`; general-domain files such
as `data/general/deu_test.json` are outside this pipeline.

The repository may contain short sample data, while the full language corpora
remain external. Paths below are repository-relative examples. Do not write
`/data/...` unless an absolute directory named `data` really exists at the
filesystem root.

## Scripts and order

1. `create_upper_bound.py` draws the disjoint upper-bound sample from the
   external full biomedical corpus.
2. `truncate_samples.py` creates evaluation prefixes.
3. `generate_unigram.py` learns language-specific unigram distributions and
   generates lexical continuations.
4. `generate_biogpt.py` completes prefixes with BioGPT-Large.
5. `generate_prompt_based.py` generates new biomedical sentences from prompts.
6. `generate_completion_based.py` completes prefixes with the historical vLLM
   model collection.

`common.py` contains dependency-free validation, naming, JSON, and seed helpers.

## Outcome-critical historical choices

- Truncation keeps `ceil(n_words * 0.30)` leading whitespace-delimited words,
  with at least one word. Whitespace in `original` is normalized. The default
  must remain `--fraction 0.30`.
- The historical completion prompt says the prefix is "exactly one third" of
  the original even though preprocessing used 30%. The prompt wording is kept
  unchanged because changing a model prompt changes generated outcomes.
- Upper-bound sampling uses seed 42 and the original per-filename SHA-256 seed
  derivation.
- Unigram learning uses seed 13 by default. It excludes both the real sample
  and upper-bound sample before equalizing training-corpus sizes. Each
  protected set must match the external learning corpus at 100% coverage by
  default.
- The original BioGPT script accepted seed 42 but did not apply it. The two
  original vLLM scripts were also unseeded. Therefore, previously generated
  neural outputs cannot be reconstructed exactly from those scripts and seed
  values alone. Preserve the experimental output files as authoritative
  artifacts. The harmonized scripts seed new runs and record their settings.
- The prompt-based generator retains the biomedical half of the historical
  alternating-domain plan. `--total-records 540000` therefore emits 270,000
  biomedical records (30,000 per language) with their original odd IDs such as
  `gen_1`, `gen_3`, and so on.
- Completion-based vLLM generation still stops at `.`, `!`, and `?`. Under
  vLLM's default behavior the stop character is not included in output. This is
  retained as historical behavior even though the prompt requests a period.

## Example commands

Create the upper bound from an external corpus:

```bash
python src/synthesis/create_upper_bound.py \
  --input-dir /path/to/biomedical_per_language_0226 \
  --original-sample-dir /path/to/biomedical_per_language_0226/downsampled_30k \
  --output-dir /path/to/biomedical_per_language_0226/downsampled_30k_second \
  --sample-size 30000 \
  --seed 42
```

Create 30% prefixes from repository sample files:

```bash
python src/synthesis/truncate_samples.py \
  --input-dir data/biomedical \
  --output-dir data/biomedical/truncated \
  --fraction 0.30
```

Run the unigram baseline while protecting both experimental sets:

```bash
python src/synthesis/generate_unigram.py \
  --input data/biomedical/truncated \
  --train-dir /path/to/full/biomedical/corpora \
  --test-dir data/biomedical \
  --upper-bound-dir data/biomedical/upperbound \
  --output outputs/unigram \
  --alpha 0.75 \
  --seed 13
```

Run BioGPT:

```bash
python src/synthesis/generate_biogpt.py \
  --input data/biomedical/truncated \
  --output outputs/biogpt \
  --seed 42
```

Generate prompt-based biomedical sentences with one model:

```bash
python src/synthesis/generate_prompt_based.py \
  --model MODEL_ORGANIZATION/MODEL_NAME \
  --output-dir outputs/prompt_based \
  --seed 42
```

Run completion-based generation with one or more selected models:

```bash
python src/synthesis/generate_completion_based.py \
  --input data/biomedical/truncated \
  --output outputs/completion \
  --model MODEL_ORGANIZATION/FIRST_MODEL \
  --model MODEL_ORGANIZATION/SECOND_MODEL \
  --tensor-parallel-size 4 \
  --seed 42
```

Omitting `--model` from the completion command uses the full historical model
list. GPU assignment should normally be controlled by the scheduler through
`CUDA_VISIBLE_DEVICES`.

## Environment and verification

The preprocessing and unigram scripts use only Python's standard library. The
neural generators additionally require the packages used by the corresponding
runtime:

- BioGPT: `torch`, `transformers`, `tqdm`, and the model files.
- vLLM generators: `torch`, `vllm`, `transformers`, and the model files.

Exact package versions, CUDA libraries, model revisions, and hardware should
be archived with the experimental run. Model IDs alone do not pin a Hugging
Face repository revision.

Run dependency-free checks with:

```bash
python -m unittest tests/test_synthesis.py
```
