# Alignment analyses

`common.R` prepares the model-language tables shared by the univariate and multivariate analyses. Internal path handling, environment parsing, model-id validation, required-column checks, and connected-component construction live in `utils.R`, which is sourced automatically and should not be run directly.

The intended order is:

```text
                         ┌─> univariate.R ─> univariate_stability.R
common.R (preparation) ──┤
                         └─> multivariate.R
```


## Common data preparation

Run the common stage once for each model that will be analyzed:

```bash
ALIGNMENT_MODEL_ID=<model> Rscript src/alignment/common.R
```

This stage reads the feature CSVs, validates their language and model metadata, filters explicit generation-fallback rows, identifies features estimable in both domains, and writes:

- `df_clean.rds`: filtered feature values and analysis identifiers;
- `df_z.rds`: features z-scored within model and language; and
- `feats_ok.rds`: estimable model-language-feature combinations.

These files are stored below `outputs/alignment/univariate/<model>/` by default because they are also the starting point for the univariate results. Both downstream scripts reuse them without loading the raw feature CSVs.

## Univariate analysis

The univariate workflow compares the distribution of each LFTK feature between real general-language samples and biomedical samples, separately for every model and language. It consists of two scripts so the computationally expensive scoring and resampling do not need to be repeated when adjusting the final stability-supported clustering.

### 1. Score features and estimate stability

```bash
ALIGNMENT_MODEL_ID=<model> Rscript src/alignment/univariate.R
```

For each prepared model-language contrast, `univariate.R`:

1. computes Cohen's d, one-dimensional Wasserstein distance, and Jensen-Shannon
   divergence;
2. combines metric ranks into a Borda score;
3. estimates permutation p-values with 500 label permutations and applies the
   Benjamini-Hochberg correction within each model-language contrast;
4. performs 100 balanced resamples to quantify the stability of effects and
   feature ranks; and
5. writes the statistical results and significant-feature clustering outputs.

The full-data permutation results provide statistical inference. The balanced resamples are a sensitivity analysis: they repeatedly draw the same number of general and biomedical observations in every cell and recompute the observed metrics and ranks without rerunning permutation tests. Unless `ALIGNMENT_BALANCED_TARGET_N` is set, the common sample size is the smallest available model-language-domain cell. Within a repetition, a language's real general rows are shared across models, while biomedical rows are sampled independently for each model and language.

The random seed is 123. Balanced repetition `b` uses seed `123 + 100000 + b`. The configuration and R session information are saved as `analysis_config.rds` and `session_info.rds`.

### 2. Cluster stable and significant features

```bash
ALIGNMENT_MODEL_ID=<model> Rscript src/alignment/univariate_stability.R
```

This stage reads `df_z.rds`, `per_lang_scores.rds`, and `balanced_stability_feature_summary.rds` from the first stage. A feature is selected when all of the following hold:

- full-data permutation q-value is at most 0.05;
- it is estimable in at least 95% of balanced repetitions;
- its effect direction matches the full-data direction in at least 95% of repetitions; and
- it occurs among the top 20 features in at least 80% of repetitions.

Selected features are clustered within languages using absolute Spearman correlation of at least 0.80 after domain centering. Cross-language meta-clusters use Jaccard thresholds of 0.20, 0.30, 0.40, 0.60, and 0.80. The results are written below `stable_significant_clusters/` so they remain separate from the significant-only clustering produced by the first stage.

## Multivariate analysis

```bash
ALIGNMENT_MODEL_ID=<model> Rscript src/alignment/multivariate.R
```

`multivariate.R` reads `df_z.rds`, `feats_ok.rds`, and `df_clean.rds` from the selected model's common-preparation directory. Therefore, each multivariate task must be preceded by a model-specific common run:

```bash
ALIGNMENT_MODEL_ID=<model> Rscript src/alignment/common.R
ALIGNMENT_MODEL_ID=<model> Rscript src/alignment/multivariate.R
```

The default real reference model id is `real`. Prepare that reference once before running non-reference targets:

```bash
ALIGNMENT_MODEL_ID=real Rscript src/alignment/common.R
```

The multivariate workflow uses random forests with seed 123, five stratified folds, and 500 trees. For a synthetic target, the primary analyses train forests on authentic biomedical/general data and evaluate the same fitted forest on matched authentic and generated test data. They cover:

- per-language stratified cross-validation;
- pooled language-agnostic cross-validation;
- leave-one-language-out evaluation; and
- directed cross-language transfer.

The same shared feature panel and authentic-data scaling are applied to both evaluations. Authentic forests may be cached, but a cache is reused only when the authentic-data signature, exact feature panel, language coverage, model settings, package version, R version, and seeds match.

Two complementary sensitivity analyses are enabled by default:

- Balanced-refit stability repeats the per-language authentic-versus-generated endpoint 20 times using a fresh balanced authentic sample, a fresh 80/20 train/held-out split, and a 200-tree forest. Stability requires at least 95% estimable iterations and at least 95% agreement with the full-data effect direction. Its subsampling intervals are descriptive quantiles rather than formal confidence intervals.
- Biomedical-to-biomedical analysis directly distinguishes authentic biomedical samples from target biomedical samples using per-language, pooled, leave-one-language-out, and transfer settings. Each language/class is balanced to at most 3,000 observations by default.

Model-specific biomedical/general discrimination runs automatically for the `real` reference task. It is disabled for synthetic targets unless `RF_MODEL_SPECIFIC_ENABLED=true`. Training-only downsampling and shallow CART diagnostics are also optional and disabled by default.

Use the univariate checkpoint root to list the available model tasks:

```bash
Rscript src/alignment/multivariate.R --list-models
```

## Input layout

Feature CSVs matching this layout are distributed via Zenodo rather than
checked into this repository; see the root README for the DOI and download
instructions.

The default paths are relative to the repository root:

```text
data/features/real/                         real general and biomedical CSVs
data/features/<feature-family>/             biomedical feature CSVs
outputs/alignment/univariate/               analysis results and checkpoints
outputs/alignment/multivariate/             random-forest results
```

Biomedical files may be language-specific, for example `features_model_biomedical_de.csv`, or may contain multiple languages in `features_model_biomedical.csv`. For combined files, every row must contain a supported language in the `country` column. Real general files follow the same combined or language-specific convention.

The expected biomedical feature-family names are:

```text
biogpt_features
completions20260216_bio
unigram_features
upper_bound_features
vllm_features
```

If none of these directories exists, biomedical CSVs are discovered recursively below `data/features`. This supports the date/model directories written by `src/features/extract_lftk.py`.

## Configuration

Paths and scheduler selections can be supplied without editing either script:

```bash
ALIGNMENT_REAL_GENERAL_DIR=/path/to/real/general \
ALIGNMENT_REAL_BIOMEDICAL_DIR=/path/to/real/biomedical \
ALIGNMENT_BIOMEDICAL_FEATURE_ROOT=/path/to/feature/root \
ALIGNMENT_UNIVARIATE_OUTPUT_DIR=/path/to/results \
ALIGNMENT_MODEL_ID=<model> \
Rscript src/alignment/common.R
```

Available variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ALIGNMENT_PROJECT_ROOT` | repository root | Base for relative paths |
| `ALIGNMENT_FEATURE_ROOT` | `data/features` | Shared feature root |
| `ALIGNMENT_REAL_GENERAL_DIR` | `data/features/real` | Real general features |
| `ALIGNMENT_REAL_BIOMEDICAL_DIR` | `data/features/real` | Real biomedical features |
| `ALIGNMENT_BIOMEDICAL_FEATURE_ROOT` | `data/features` | Biomedical feature discovery root |
| `ALIGNMENT_BIOMEDICAL_FAMILIES` | family list above | Comma-separated family directories |
| `ALIGNMENT_ADDITIONAL_BIOMEDICAL_DIRS` | empty | Comma-separated additional directories |
| `ALIGNMENT_VLLM_RUN_DATE` | `20260708` | vLLM run date; use `all` for every date |
| `ALIGNMENT_UNIVARIATE_OUTPUT_DIR` | `outputs/alignment/univariate` | Checkpoint and result root |
| `ALIGNMENT_MULTIVARIATE_OUTPUT_DIR` | `outputs/alignment/multivariate` | Random-forest result root |
| `ALIGNMENT_MODEL_ID` | empty | Run or post-process one model |
| `ALIGNMENT_REAL_MODEL_ID` | `real` | Authentic reference model for multivariate tasks |
| `ALIGNMENT_REAL_INPUT_DIR` | selected reference directory | Optional authentic checkpoint directory |
| `ALIGNMENT_RESUME` | `true` | Reuse complete scoring checkpoints |
| `ALIGNMENT_REAL_BIOMEDICAL_CONTRAST` | `true` | Also compare real biomedical references with biomedical targets |
| `ALIGNMENT_BALANCED_STABILITY` | `true` | Run balanced stability analysis |
| `ALIGNMENT_BALANCED_REPS` | `100` | Number of balanced repetitions |
| `ALIGNMENT_BALANCED_TARGET_N` | smallest cell | Rows drawn per domain cell |

Use `--list-models` with `common.R` or `univariate.R` to inspect discovered model IDs. With `univariate_stability.R` or `multivariate.R`, the option lists model-specific checkpoint directories.

The multivariate method settings retain `RF_*` names because they configure random-forest-specific analyses. Important controls include:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RF_REAL_TO_GENERATED_ENABLED` | `true` | Paired authentic-versus-generated analyses |
| `RF_REAL_TO_GENERATED_R` | `5` | Repeated paired evaluations |
| `RF_SAMPLING_STABILITY_ENABLED` | `true` | Balanced-refit stability analysis |
| `RF_SAMPLING_STABILITY_ITERATIONS` | `20` | Balanced-refit iterations |
| `RF_SAMPLING_STABILITY_TREES` | `200` | Trees per stability forest |
| `RF_BIOBIO_ENABLED` | `true` | Direct biomedical-to-biomedical analyses |
| `RF_BIOBIO_MAX_N_PER_CLASS` | `3000` | Maximum balanced observations per language/class |
| `RF_MODEL_SPECIFIC_ENABLED` | `false` | Model-specific discrimination for synthetic targets |
| `RF_DOWNSAMPLE_ENABLED` | `false` | Training-only downsampling diagnostic |
| `RF_DOWNSAMPLE_MODEL_IDS` | empty | Optional comma-separated diagnostic allowlist |
| `RF_AUTHENTIC_CACHE_ENABLED` | `true` | Reuse compatible authentic forests |
| `RF_DECISION_TREE_ENABLED` | `false` | Fit shallow CART diagnostics |
| `SLURM_CPUS_PER_TASK` | `1` | Ranger threads allocated to each forest |

## Dependencies

The univariate scripts require R packages `dplyr`, `tidyr`, `stringr`, `purrr`, `tibble`, and `readr`. The multivariate script additionally requires `ranger` and `rpart`.

Lightweight statistical regression checks (`tests/test_common.R`, `tests/test_univariate.R`, `tests/test_multivariate.R`) are planned but not yet added.
