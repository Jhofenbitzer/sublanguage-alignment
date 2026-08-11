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
└── models/
    ├── upper_bound_features/features_upper_bound_biomedical_<lang>.csv       (9 files)
    ├── biogpt_features/features_biogpt-large_biomedical_<lang>.csv           (9 files)
    ├── unigram_features/features_biounigram_biomedical_<lang>.csv            (9 files)
    ├── vllm_features/20260708/features_gemma-3-1b-it_biomedical_<lang>.csv   (9 files)
    └── completions20260216_bio/features_gemma-3-1b-it_biomedical_<lang>.csv  (9 files)
```

This mirrors the real Zenodo deposit's structure after flattening (see below):
`real/` holds combined, all-language files; everything under `models/` is one
file per model per language, matching the layout documented in
`src/alignment/README.md`. `country` values follow the real convention too:
`general_<iso3>` in `real/features_real_general.csv` (e.g. `general_deu`) and
`biomedical_<iso2>` everywhere else (e.g. `biomedical_de`).

## Why this layout, and what changed

These fixtures were regenerated after inspecting the actual ~66 GB Zenodo
staging data. The original data was not directly usable as shipped:

- `real/features_general.csv` / `features_biomedical.csv` didn't match the
  filename pattern `read_real_general_files()`/`read_real_biomedical_files()`
  require (`features_..._general.csv` / `features_..._biomedical.csv` — note
  the required infix); renamed to `features_real_general.csv` /
  `features_real_biomedical.csv`, which do match.
- Per-model families lived under `features/models/<family>/`, one directory
  layer deeper than `src/alignment/univariate.R`'s `resolve_family_dir()`
  looked for by default (`data/features/<family>` or
  `data/features/outputs/<family>`, never `data/features/models/<family>`).
  Fixed in code: `resolve_family_dir()` now checks `models/<family>` first
  (see `src/alignment/univariate.R`), so this layout works with zero
  environment-variable overrides.
- Per-model, per-language CSVs were nested inside SLURM array-job wrapper
  directories (`_SUCCESS` markers, `no_date/` placeholders, directories
  literally named e.g. `biomedical_de.biogpt-large.de.biomedical.json` that
  are folders, not JSON files) with no language suffix in the leaf filename —
  discoverable only because recursive search plus the `country` column
  happened to still work, and only by accident: the duplicate-language safety
  check in `discover_biomedical_manifest()` is silently skipped whenever the
  filename carries no language suffix. Flattened to
  `features_<model>_biomedical_<lang>.csv` directly under
  `models/<family>/`, so every family now gets that safety check and the
  directory is something a person can actually browse.

Real feature CSVs downloaded from Zenodo should already reflect this
flattened structure — this was fixed at the source before upload, not by
adding a translation step here or in the R scripts.

## How the fixtures were generated

For each of 9 languages and a representative subset of real LFTK column names
(`t_word, t_stopword, t_punct, t_syll, t_uword, t_sent, t_char, a_word_ps,
a_char_pw, a_syll_pw, n_noun, n_verb, n_adj, n_adv, n_pron, simp_ttr,
root_ttr, fkre, fkgl, smog` — 20 of the real ~156, chosen to cover count,
average, POS, type-token-ratio, and readability families), 30 rows were drawn
per language per file from `N(language_level * general_mean, general_sd)`,
with a fixed per-feature general-vs-biomedical shift added for biomedical
rows. Each non-reference model reproduces that shift with a different
"recovery factor" and noise multiplier so the models aren't statistically
identical:

| Model | Recovery of the true shift | Noise multiplier |
| --- | --- | --- |
| `real_biomedical` | 1.00 (defines the true shift) | 1.00 |
| `upper_bound` | 1.00 (independent authentic draw) | 1.00 |
| `biogpt-large` | 0.70 | 1.20 |
| `gemma-3-1b-it` (completion) | 0.65 | 1.15 |
| `gemma-3-1b-it` (prompt, vLLM) | 0.55 | 1.30 |
| `biounigram` | 0.15 | 1.60 |

All draws use `random.Random` seeded deterministically from
`(purpose, label, language, is_biomedical)` via a SHA-256-derived integer, so
regenerating with the same parameters reproduces the same files.

## Verified runnable

Every stage was run against both these fixtures and, separately, against the
real (large-scale) Zenodo data after restructuring: `ALIGNMENT_MODEL_ID=<model>
Rscript src/alignment/common.R`, then `univariate.R`, `univariate_stability.R`,
and `multivariate.R`, for `real`, `upper_bound`, `biogpt-large`, `biounigram`,
`gemma-3-1b-it_prompt`, and `gemma-3-1b-it_completion` — all completed
successfully, and `Rscript src/alignment/common.R --list-models` correctly
discovered all ~70 model/variant ids in the real data.

One pre-existing script issue surfaced during this: running `univariate.R`
with `ALIGNMENT_MODEL_ID=real` and the default
`ALIGNMENT_REAL_BIOMEDICAL_CONTRAST=true` always fails with `Selected model
'real' was not discovered as a biomedical target for the second contrast`,
because that optional contrast looks up the selected model in the
family-directory manifest, which never assigns a model a `real` id (the
`real` reference is read through a separate code path). This reproduces with
any input data. Set `ALIGNMENT_REAL_BIOMEDICAL_CONTRAST=false` when running
the `real` model until this is addressed.

At this fixture's scale (30 rows per domain, 20 independent candidate
features), `univariate_stability.R`'s stability thresholds are hard to
satisfy — expect very few or zero stable-and-significant features for some
models (`real` returned zero with `ALIGNMENT_BALANCED_REPS=15`). That's a
statistical-power artifact of the tiny synthetic sample, not a script bug.
