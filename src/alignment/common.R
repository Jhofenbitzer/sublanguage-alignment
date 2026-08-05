#!/usr/bin/env Rscript

# Prepare the shared model-language tables used by both alignment analyses.
#
# The script reads feature CSVs, validates language and model metadata, filters
# invalid generation fallbacks, removes non-estimable features, and writes
# df_clean.rds, df_z.rds, and feats_ok.rds. Statistical scoring and
# random-forest fitting are performed by their respective analysis scripts.

alignment_source_dir <- trimws(Sys.getenv("ALIGNMENT_SOURCE_DIR", unset = ""))
if (!nzchar(alignment_source_dir)) {
  alignment_file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  alignment_source_dir <- dirname(sub("^--file=", "", alignment_file_arg[[1]]))
}
alignment_source_dir <- normalizePath(alignment_source_dir, mustWork = FALSE)

previous_prepare_only <- Sys.getenv("ALIGNMENT_PREPARE_ONLY", unset = NA_character_)
Sys.setenv(
  ALIGNMENT_SOURCE_DIR = alignment_source_dir,
  ALIGNMENT_PREPARE_ONLY = "true"
)
on.exit({
  if (is.na(previous_prepare_only)) {
    Sys.unsetenv("ALIGNMENT_PREPARE_ONLY")
  } else {
    Sys.setenv(ALIGNMENT_PREPARE_ONLY = previous_prepare_only)
  }
}, add = TRUE)

sys.source(
  file.path(alignment_source_dir, "univariate.R"),
  envir = environment()
)

print_common_usage <- function() {
  cat(
    "Usage: ALIGNMENT_MODEL_ID=<model> Rscript src/alignment/common.R\n",
    "       Rscript src/alignment/common.R --list-models\n",
    "\nThe output checkpoints are consumed by univariate.R and multivariate.R.\n",
    "See src/alignment/README.md for configuration.\n",
    sep = ""
  )
}

if (sys.nframe() == 0L) {
  cli_args <- commandArgs(trailingOnly = TRUE)
  if (any(cli_args %in% c("--help", "-h"))) {
    print_common_usage()
  } else if ("--list-models" %in% cli_args) {
    cat(paste(discover_available_model_ids(), collapse = "\n"), "\n", sep = "")
  } else {
    main()
  }
}
