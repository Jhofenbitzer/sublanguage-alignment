# Reproducible analysis reports

The Quarto reports consume the saved outputs produced by the scripts in
`src/alignment/`. Run the reports from the repository root so that all paths
resolve consistently.

The default project render order is:

1. `01_data_quality.qmd`
2. `02_univariate_alignment.qmd`
3. `03_multivariate_alignment.qmd`
4. `04_upper_bound_calibration.qmd`

Each report has one responsibility:

- data quality checks input coverage and feature availability;
- univariate alignment reports feature recovery and ranking agreement for
  models and ordinary baselines;
- multivariate alignment reports authentic-trained RF contrasts and refit
  stability for models and ordinary baselines; and
- upper-bound calibration contains all independent-human recovery,
  biomedical-control, and empirical-margin diagnostics.

The univariate and multivariate reports write compact model summaries to
`outputs/analysis/tables/`. Figures and other derived tables are written below
`outputs/analysis/`.

The source feature matrices belong in `data/features/`; see
`src/alignment/README.md` for the exact real/model layout.

By default, the reports read computed results from:

- `data/processed/alignment/univariate/`
- `data/processed/alignment/multivariate/`

These locations can be overridden with `ANALYSIS_RESULT_ROOT` and
`RF_RESULT_ROOT`, respectively. `ANALYSIS_OUTPUT_ROOT` overrides the report
artifact directory.

Both the feature matrices and the saved R outputs are distributed via Zenodo
rather than checked into this repository; see the root README for the DOI and
download instructions. Download and extract them to the default paths above
so the reports can be rendered without recomputing the statistical analyses.
To render newly computed results instead, set the two result-root variables to
`outputs/alignment/univariate` and `outputs/alignment/multivariate`.

`diagnostics/real_reference.qmd` contains optional exploratory diagnostics for
the authentic reference analysis.
