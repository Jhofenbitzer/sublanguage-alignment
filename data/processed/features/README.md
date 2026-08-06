# Analysis feature matrices

This directory contains the processed CSV files used as direct inputs to the
alignment analyses. These files are distinct from the short raw-text samples in
`data/general/` and `data/biomedical/`.

Expected layout:

```text
data/processed/features/
├── real/
│   ├── features_real_general.csv
│   └── features_real_biomedical.csv
└── models/
    ├── features_<model-id>_biomedical.csv
    └── ...
```

Each model variant has its own stable `<model-id>`. Prompt and completion
variants must remain distinguishable, for example with `_prompt` and
`_completion` suffixes. Synthetic general-domain feature files are not part of
the analysis.

Files may contain all languages together. In that case, every row must identify
its language in the `country` column. Language-specific files named
`features_<model-id>_biomedical_<language>.csv` are also supported.

The feature columns should be supplied exactly as analyzed. When the files are
added, record their SHA-256 checksums in `checksums.sha256` so users can verify
that the public inputs match the analysis inputs.

From the repository root, the manifest can be created with:

```bash
find data/processed/features -type f -name '*.csv' \
  -exec shasum -a 256 {} + \
  | sort \
  > data/processed/features/checksums.sha256
```
