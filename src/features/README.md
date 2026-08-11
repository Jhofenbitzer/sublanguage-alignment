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

By default, output is grouped into `<output-root>/<date>/<model>/features_<model>_<domain>.csv`
— `<date>` is parsed from the input filename (`no_date` when none is found),
so one invocation covering several dates or models writes each combination to
its own subdirectory rather than overwriting a shared file. `src/alignment/*.R`
discovers CSVs recursively, so this nested form is always safe to feed it.

To land output directly in the flat layout `src/alignment/README.md`
documents (`data/features/models/<feature-family>/features_<model>_<domain>_<lang>.csv`,
no `<date>/<model>/` subdirectories), add `--flat-output` and run one
invocation per language, with `--output-root` set to the family directory:

```bash
for lang in de en es fr it nl pl ro sv; do
  python src/features/extract_lftk.py \
    --input-dir outputs/completions \
    --output-root data/features/models/completions20260216_bio \
    --n-process 8 \
    --batch-size 512 \
    --spacy-model-size lg \
    --language-filter "$lang" \
    --flat-output \
    --text-fields synthesized,text
done
```

`--flat-output` raises instead of silently overwriting a file if the input
directory actually spans more than one date/model/domain combination for the
same language filter — drop it (or invoke once per combination) if that
happens.

When `--language-filter` selects one language, its normalized two-letter code
is appended to the CSV filename (`features_<model>_<domain>_<lang>.csv`); this
is required to avoid collisions when `--flat-output` is set, and is what
`completions20260216_bio` and `vllm_features` already use. Passing
`--language-filter all` never appends a language suffix, so combine it with
`--flat-output` only when the whole call covers a single language already
(for example, single-language input files), or rows across languages have to
be told apart via the `country` column instead.

Malformed JSONL records are skipped by default and counted in
`<output-root>/error_count.csv`. Use `--malformed-record-policy error` to stop
at the first malformed record.

## Dependencies

The extractor requires `lftk`, `spacy`, and `tqdm`, plus the spaCy language
models requested by the input records. Named-entity recognition remains enabled
because some LFTK features depend on it.
