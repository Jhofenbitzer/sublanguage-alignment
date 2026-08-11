# Synthetic smoke-test fixtures

The CSVs below are **not real LFTK output and are not used for any result in
the paper**. They are small, seeded, synthetic numeric feature matrices whose
only purpose is to let `src/alignment/common.R`, `univariate.R`,
`univariate_stability.R`, and `multivariate.R` run end to end without any
external data, so the alignment pipeline can be exercised and reviewed
locally.

Real feature matrices are distributed via Zenodo; see the root README for the
DOI and download instructions. Extracting that deposit over this directory
replaces these fixtures.

## Layout

```text
data/features/
├── real/
│   ├── features_real_general.csv
│   └── features_real_biomedical.csv
├── upper_bound_features/features_upper_bound_biomedical.csv
├── biogpt_features/features_biogpt-large_biomedical.csv
├── unigram_features/features_biounigram_biomedical.csv
├── vllm_features/20260708/features_gemma-3-1b-it_biomedical.csv
└── completions20260216_bio/features_gemma-3-1b-it_biomedical.csv
```

This mirrors the layout documented in `src/alignment/README.md`: one combined,
all-language file per model/domain, with language identified by the `country`
column.

## How they were generated

For each of 9 languages and 12 invented LFTK-style numeric columns (word/
sentence/syllable counts, POS counts, `flesch_reading_ease`, type-token
ratio), 30 rows were drawn per language per file from
`N(language_level * general_mean, general_sd)`, with a fixed per-feature
general-vs-biomedical shift added for biomedical rows. Each non-reference
model reproduces that shift with a different "recovery factor" and noise
multiplier so the models aren't statistically identical:

| Model file | Recovery of the true shift | Noise multiplier |
| --- | --- | --- |
| `real_biomedical` | 1.00 (defines the true shift) | 1.00 |
| `upper_bound` | 1.00 (independent authentic draw) | 1.00 |
| `biogpt-large` | 0.70 | 1.20 |
| `gemma-3-1b-it` (completion) | 0.65 | 1.15 |
| `gemma-3-1b-it` (prompt, vLLM) | 0.55 | 1.30 |
| `biounigram` | 0.15 | 1.60 |

All draws use `random.Random` seeded deterministically from
`(purpose, label, language)` via a SHA-256-derived integer, so regenerating
with the same parameters reproduces the same files.

## Verified runnable

Every stage was run against these fixtures directly (`ALIGNMENT_MODEL_ID=real
Rscript src/alignment/common.R`, then `univariate.R`,
`univariate_stability.R`, and `multivariate.R`, repeated for `upper_bound`,
`biogpt-large`, `biounigram`, `gemma-3-1b-it_prompt`, and
`gemma-3-1b-it_completion`) and completed successfully.

One pre-existing script issue surfaced during this: running `univariate.R`
with `ALIGNMENT_MODEL_ID=real` and the default
`ALIGNMENT_REAL_BIOMEDICAL_CONTRAST=true` always fails with `Selected model
'real' was not discovered as a biomedical target for the second contrast`,
because that optional contrast looks up the selected model in the
family-directory manifest, which never assigns a model a `real` id (the
`real` reference is read through a separate code path). This reproduces with
any input data, not just these fixtures. Set
`ALIGNMENT_REAL_BIOMEDICAL_CONTRAST=false` when running the `real` model
until this is addressed.

At this fixture's scale (30 rows per domain, 12 independent candidate
features), `univariate_stability.R`'s stability thresholds are hard to
satisfy — expect very few or zero stable-and-significant features for some
models (`real` returned zero with `ALIGNMENT_BALANCED_REPS=15`). That's a
statistical-power artifact of the tiny synthetic sample, not a script bug.
