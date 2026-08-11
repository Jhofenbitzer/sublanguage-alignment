# Synthetic smoke-test fixtures

The CSVs below are synthetic, seeded numeric feature matrices, not real LFTK
output. They let `src/alignment/common.R`, `univariate.R`,
`univariate_stability.R`, and `multivariate.R` run end to end without
external data, and are not used for any result in the paper.

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

This mirrors the layout of the real Zenodo deposit: `real/` holds combined,
all-language files, and everything under `models/` is one file per model per
language, matching `src/alignment/README.md`. `resolve_family_dir()` in
`src/alignment/univariate.R` resolves each family as
`data/features/models/<family>/` first. Real files must match
`features_..._general.csv` / `features_..._biomedical.csv` for
`read_real_general_files()` / `read_real_biomedical_files()` to find them.
`country` values follow the real convention: `general_<iso3>` in
`real/features_real_general.csv` (e.g. `general_deu`) and
`biomedical_<iso2>` everywhere else (e.g. `biomedical_de`). Feature CSVs
downloaded from the Zenodo deposit already follow this layout; no reshaping
is needed.

## How the fixtures were generated

For each of 9 languages and a representative subset of real LFTK column names
(`t_word, t_stopword, t_punct, t_syll, t_uword, t_sent, t_char, a_word_ps,
a_char_pw, a_syll_pw, n_noun, n_verb, n_adj, n_adv, n_pron, simp_ttr,
root_ttr, fkre, fkgl, smog` — 20 of the real ~156, chosen to cover count,
average, POS, type-token-ratio, and readability families), 30 rows were drawn
per language per file from `N(language_level * general_mean, general_sd)`,
with a fixed per-feature general-vs-biomedical shift added for biomedical
rows. Each non-reference model reproduces that shift with a different
recovery factor and noise multiplier, so the models are not statistically
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

## Known limitations

Running `univariate.R` with `ALIGNMENT_MODEL_ID=real` and the default
`ALIGNMENT_REAL_BIOMEDICAL_CONTRAST=true` fails with `Selected model 'real'
was not discovered as a biomedical target for the second contrast`: the
optional contrast looks up the selected model in the family-directory
manifest, which never assigns the `real` id, since the `real` reference is
read through a separate code path. This reproduces with any input data. Set
`ALIGNMENT_REAL_BIOMEDICAL_CONTRAST=false` when running the `real` model
until this is fixed.

At this fixture's scale (30 rows per domain, 20 independent candidate
features), `univariate_stability.R`'s stability thresholds are hard to
satisfy; expect few or no stable-and-significant features for some models.
This reflects the small synthetic sample size, not a defect in the script.
