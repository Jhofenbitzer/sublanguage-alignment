# Sublanguage Alignment

Code and data-processing pipeline for our paper *Sublanguage Alignment: A Linguistically-Grounded Evaluation Method for Domain-Specific Text Generation*. Whether
biomedical text produced by different generation methods reproduces the
linguistic shifts that authentic biomedical text shows relative to general-
domain language, across multiple languages and models.

## Pipeline

The project runs in four stages, each documented in its own README:

1. **[Synthesis](src/synthesis/README.md)** (`src/synthesis/`) — generates
   biomedical text with several methods (unigram baseline, BioGPT, prompt-
   based and completion-based LLM generation) and draws the authentic
   upper-bound sample.
2. **[Feature extraction](src/features/README.md)** (`src/features/`) —
   extracts a shared LFTK linguistic feature set from real and generated text
   in both domains.
3. **[Alignment analysis](src/alignment/README.md)** (`src/alignment/`, R) —
   univariate distributional tests (Cohen's d, Wasserstein, Jensen-Shannon,
   permutation testing, stability resampling) and multivariate random-forest
   discriminability analyses comparing authentic and generated text.
4. **[Reports](analysis/README.md)** (`analysis/`, Quarto) — renders the
   saved analysis outputs into data-quality, univariate-alignment,
   multivariate-alignment, and upper-bound-calibration reports.

```text
synthesis ──> feature extraction ──> alignment analysis (R) ──> Quarto reports
```

## Data

`data/general/` and `data/biomedical/` contain small illustrative samples
that document the expected input JSON layout only — they are not the corpora
used in the paper.

The processed feature matrices and saved R analysis outputs (the inputs
consumed by `analysis/`) are hosted on Zenodo rather than checked into this
repository:

- DOI: [10.5281/zenodo.21875590](https://doi.org/10.5281/zenodo.21875590)

To reproduce the reports without recomputing the statistical analyses,
download the Zenodo deposit and extract its contents to:

- `data/features/` — feature CSVs (see `src/alignment/README.md` for the
  exact real/model layout)
- `data/processed/alignment/univariate/` and
  `data/processed/alignment/multivariate/` — saved R outputs

The full external text corpora used for synthesis are not redistributed; see
`src/synthesis/README.md`.

## Setup

- Python 3.x with the packages listed in `src/features/README.md` and
  `src/synthesis/README.md` (`lftk`, `spacy`, `tqdm`, plus `torch`,
  `transformers`, and `vllm` for the neural generators).
- R with `dplyr`, `tidyr`, `stringr`, `purrr`, `tibble`, `readr`, `ranger`,
  and `rpart` (see `src/alignment/README.md`).
- [Quarto](https://quarto.org/) to render the reports in `analysis/`.

Dependency versions are not yet pinned; see the per-stage READMEs for the
exact package lists.

## Reproducing the reports

After downloading and extracting the Zenodo artifacts as described above, run
the alignment scripts per model as documented in
[src/alignment/README.md](src/alignment/README.md), then render the reports
from the repository root:

```bash
quarto render
```

or render an individual notebook, e.g. `quarto render analysis/01_data_quality.qmd`.

## Citation

TODO: add the paper citation once accepted.

## License

Code is released under the [MIT License](LICENSE).
