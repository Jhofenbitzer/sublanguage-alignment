# LFTK feature extraction

`extract_lftk.py` extracts the same LFTK feature set from prompt-based
generations, completion-based generations, and real reference samples. It
processes both general and biomedical domains and writes them to separate CSVs.

## Supported inputs

The input directory may contain JSON arrays, individual JSON objects, or JSONL
files. Supported filename layouts include:

- `completions_20260216.<model>.<language>.biomedical.json`
- `generations_20260216.<model>.<language>.biomedical.jsonl`
- `<model>.jsonl` for prompt-based generation, with language and source stored
  in each record
- `biomedical_de.json`
- `data/general/deu_test.json`
- `data/biomedical/upperbound/de_test.json`

For model-only files, the `source` field determines the domain record by record.

## Text selection

The default field priority covers generated and real records in one run:

```text
synthesized,text,original,sentence
```

The first non-empty field is used. A narrower field set can be selected when
processing one type of input, for example:

```bash
--text-fields synthesized,text
--text-fields original,text,sentence
```

## Example

```bash
python src/features/extract_lftk.py \
  --input-dir outputs/completions \
  --output-root data/features/completion_features \
  --n-process 8 \
  --batch-size 512 \
  --spacy-model-size lg \
  --language-filter all \
  --text-fields synthesized,text
```

Outputs follow this structure:

```text
<output-root>/<date>/<model>/features_<model>_<domain>.csv
```

When `--language-filter` selects one language, its normalized two-letter code is
appended to the CSV filename.

Malformed JSONL records are skipped by default and counted in
`<output-root>/error_count.csv`. Use `--malformed-record-policy error` to stop
at the first malformed record.

## Dependencies

The extractor requires `lftk`, `spacy`, and `tqdm`, plus the spaCy language
models requested by the input records. Named-entity recognition remains enabled
because some LFTK features depend on it.
