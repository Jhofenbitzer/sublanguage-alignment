# Biomedical synthesis pipeline

These scripts synthesize the biomedical texts. This repository includes
short sample data only; the full language corpora used in the paper are
external and not redistributed here.

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

## Example commands

Create the upper bound from an external corpus:

```bash
python src/synthesis/create_upper_bound.py \
  --input-dir /path/to/large_biomedical_corpus \
  --original-sample-dir /path/to/authentic_biomedical_downsampled_corpus \
  --output-dir /path/to/biomedical_upperbound \
  --sample-size 30000 \
  --seed 42
```

Create prefixes from repository sample files:

```bash
python src/synthesis/truncate_samples.py \
  --input-dir data/biomedical \
  --output-dir data/biomedical/truncated \
  --fraction 0.30
```

`data/biomedical/truncated/` is already checked in as the output of this
exact command, so `generate_biogpt.py` and `generate_completion_based.py`
have a valid `--input` to run against out of the box.

`create_upper_bound.py`'s `--input-dir`/`--original-sample-dir` and
`generate_unigram.py`'s `--train-dir` point at the full external corpus,
which has no fixed location in this repository (see "Full language corpora
remain external" above) — supply your own path for those.

Run the unigram baseline:

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

Omitting `--model` from the completion command uses the full model list. GPU assignment should normally be controlled by the scheduler through `CUDA_VISIBLE_DEVICES`.

## Environment and verification

The preprocessing and unigram scripts use only Python's standard library. The
neural generators additionally require the packages used by the corresponding
runtime:

- BioGPT: `torch`, `transformers`, `tqdm`, and the model files.
- vLLM generators: `torch`, `vllm`, `transformers`, and the model files.


Run dependency-free checks with:

```bash
python -m unittest tests/test_synthesis.py
```
