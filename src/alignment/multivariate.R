#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(tibble)
  library(readr)
  library(ranger)
  library(rpart)
})

alignment_source_dir <- trimws(Sys.getenv("ALIGNMENT_SOURCE_DIR", unset = ""))
if (!nzchar(alignment_source_dir)) {
  alignment_file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  alignment_source_dir <- dirname(sub("^--file=", "", alignment_file_arg[[1]]))
}
source(file.path(alignment_source_dir, "utils.R"))

# =========================================================
# CONFIGURATION
# =========================================================
# One upstream model directory must contain both df_z.rds and feats_ok.rds.
Z_OUTPUT_ROOT <- alignment_env_value(
  "ALIGNMENT_UNIVARIATE_OUTPUT_DIR",
  file.path(ALIGNMENT_PROJECT_ROOT, "outputs", "alignment", "univariate"),
  "RF_Z_OUTPUT_ROOT"
)
RF_OUTPUT_ROOT <- alignment_env_value(
  "ALIGNMENT_MULTIVARIATE_OUTPUT_DIR",
  file.path(ALIGNMENT_PROJECT_ROOT, "outputs", "alignment", "multivariate"),
  "RF_OUTPUT_ROOT"
)

SELECTED_MODEL_ID <- alignment_env_value(
  "ALIGNMENT_MODEL_ID",
  "",
  "DOMAIN_SHIFT_MODEL_ID"
)
alignment_validate_model_id(SELECTED_MODEL_ID)

MODEL_INPUT_DIR <- if (nzchar(SELECTED_MODEL_ID)) {
  file.path(Z_OUTPUT_ROOT, SELECTED_MODEL_ID)
} else {
  NA_character_
}
INPUT_DF_Z     <- file.path(MODEL_INPUT_DIR, "df_z.rds")
INPUT_FEATS_OK <- file.path(MODEL_INPUT_DIR, "feats_ok.rds")
INPUT_DF_CLEAN <- file.path(MODEL_INPUT_DIR, "df_clean.rds")
OUTPUT_DIR     <- if (nzchar(SELECTED_MODEL_ID)) {
  file.path(RF_OUTPUT_ROOT, SELECTED_MODEL_ID)
} else {
  NA_character_
}

DF_Z_IS_CSV     <- FALSE
FEATS_OK_IS_CSV <- FALSE

BASE_SEED             <- 123
SCRIPT_REVISION       <- "optimized-biobio-stability-20260729"
# This cache contract identifies the authentic-forest fitting algorithm and
# permits reuse only when its data, features, settings, and seeds also match.
AUTHENTIC_CACHE_REVISION <- "optimized-20260726"
K_FOLDS               <- 5
TREES_FINAL           <- 500
TREES_FEATURE_EXTR    <- 100  
MIN_N_PER_DOMAIN      <- 5
SLURM_CPUS_PER_TASK   <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", unset = "1")))
if (!is.finite(SLURM_CPUS_PER_TASK) || SLURM_CPUS_PER_TASK < 1) SLURM_CPUS_PER_TASK <- 1L

# Let ranger parallelize each forest across the CPUs assigned by SLURM. The
# outer R loops remain sequential to prevent nested CPU oversubscription.
WORKERS               <- 1L
RANGER_NUM_THREADS    <- SLURM_CPUS_PER_TASK
PROBABILITY_FOREST    <- FALSE

TRANSFER_R            <- 5
TRANSFER_CONF         <- 0.95

# Primary paired validation: train only on real biomedical/general data and
# evaluate each identical forest on matched real and generated test data.
REAL_REFERENCE_MODEL_ID <- alignment_env_value(
  "ALIGNMENT_REAL_MODEL_ID",
  "real",
  "RF_REAL_MODEL_ID"
)
alignment_validate_model_id(
  REAL_REFERENCE_MODEL_ID,
  "ALIGNMENT_REAL_MODEL_ID",
  allow_empty = FALSE
)
REAL_INPUT_DIR <- alignment_env_value(
  "ALIGNMENT_REAL_INPUT_DIR",
  file.path(Z_OUTPUT_ROOT, REAL_REFERENCE_MODEL_ID),
  "RF_REAL_INPUT_DIR"
)
REAL_INPUT_DF_Z <- file.path(REAL_INPUT_DIR, "df_z.rds")
REAL_INPUT_FEATS_OK <- file.path(REAL_INPUT_DIR, "feats_ok.rds")
REAL_INPUT_DF_CLEAN <- file.path(REAL_INPUT_DIR, "df_clean.rds")
RF_REAL_TO_GENERATED_ENABLED <- tolower(trimws(Sys.getenv(
  "RF_REAL_TO_GENERATED_ENABLED", unset = "true"
))) %in% c("1", "true", "t", "yes", "y")
RF_REAL_TO_GENERATED_R <- suppressWarnings(as.integer(Sys.getenv(
  "RF_REAL_TO_GENERATED_R", unset = as.character(TRANSFER_R)
)))
RF_REAL_TO_GENERATED_CONF <- suppressWarnings(as.numeric(Sys.getenv(
  "RF_REAL_TO_GENERATED_CONF", unset = as.character(TRANSFER_CONF)
)))

# Repeated balanced-refit sensitivity analysis for the primary per-language
# authentic-vs-generated endpoint. Each iteration draws a fresh balanced
# authentic sample, creates a new train/held-out split, and fits a smaller RF.
RF_SAMPLING_STABILITY_ENABLED <- tolower(trimws(Sys.getenv(
  "RF_SAMPLING_STABILITY_ENABLED", unset = "true"
))) %in% c("1", "true", "t", "yes", "y")
RF_SAMPLING_STABILITY_ITERATIONS <- suppressWarnings(as.integer(Sys.getenv(
  "RF_SAMPLING_STABILITY_ITERATIONS", unset = "20"
)))
RF_SAMPLING_STABILITY_TREES <- suppressWarnings(as.integer(Sys.getenv(
  "RF_SAMPLING_STABILITY_TREES", unset = "200"
)))
RF_SAMPLING_STABILITY_MAX_N_PER_DOMAIN <-
  suppressWarnings(as.integer(Sys.getenv(
    "RF_SAMPLING_STABILITY_MAX_N_PER_DOMAIN", unset = "3000"
  )))
RF_SAMPLING_STABILITY_TRAIN_FRACTION <-
  suppressWarnings(as.numeric(Sys.getenv(
    "RF_SAMPLING_STABILITY_TRAIN_FRACTION", unset = "0.80"
  )))
RF_SAMPLING_STABILITY_MIN_COVERAGE <-
  suppressWarnings(as.numeric(Sys.getenv(
    "RF_SAMPLING_STABILITY_MIN_COVERAGE", unset = "0.95"
  )))
RF_SAMPLING_STABILITY_MIN_SIGN_AGREEMENT <-
  suppressWarnings(as.numeric(Sys.getenv(
    "RF_SAMPLING_STABILITY_MIN_SIGN_AGREEMENT", unset = "0.95"
  )))
RF_SAMPLING_STABILITY_DELTA_TOLERANCE <-
  suppressWarnings(as.numeric(Sys.getenv(
    "RF_SAMPLING_STABILITY_DELTA_TOLERANCE", unset = "0.05"
  )))

# Direct biomedical-to-biomedical discriminability complements the primary
# contrast-preservation analysis. It is run for every non-reference target by
# default and can be disabled independently.
RF_BIOBIO_ENABLED <- tolower(trimws(Sys.getenv(
  "RF_BIOBIO_ENABLED", unset = "true"
))) %in% c("1", "true", "t", "yes", "y")

# The authentic reference task runs the model-specific discriminability
# analyses. Synthetic tasks skip them by default and retain
# only the paired authentic-to-generated analyses below. Set this to true for a
# selected synthetic model when its own domain discriminability is required.
RF_MODEL_SPECIFIC_ENABLED <- tolower(trimws(Sys.getenv(
  "RF_MODEL_SPECIFIC_ENABLED", unset = "false"
))) %in% c("1", "true", "t", "yes", "y")

RF_SHARED_FEATURE_MODEL_IDS_RAW <- trimws(Sys.getenv(
  "RF_SHARED_FEATURE_MODEL_IDS", unset = ""
))
RF_SHARED_FEATURE_MODEL_IDS <- if (nzchar(RF_SHARED_FEATURE_MODEL_IDS_RAW)) {
  unique(trimws(strsplit(RF_SHARED_FEATURE_MODEL_IDS_RAW, ",", fixed = TRUE)[[1]]))
} else {
  character()
}
if (any(!grepl("^[A-Za-z0-9._-]+$", RF_SHARED_FEATURE_MODEL_IDS))) {
  stop("RF_SHARED_FEATURE_MODEL_IDS contains an invalid model identifier")
}
if (!is.finite(RF_REAL_TO_GENERATED_R) || RF_REAL_TO_GENERATED_R < 1) {
  stop("RF_REAL_TO_GENERATED_R must be an integer >= 1")
}
if (!is.finite(RF_REAL_TO_GENERATED_CONF) ||
    RF_REAL_TO_GENERATED_CONF <= 0 || RF_REAL_TO_GENERATED_CONF >= 1) {
  stop("RF_REAL_TO_GENERATED_CONF must be strictly between 0 and 1")
}
if (!is.finite(RF_SAMPLING_STABILITY_ITERATIONS) ||
    RF_SAMPLING_STABILITY_ITERATIONS < 2L) {
  stop("RF_SAMPLING_STABILITY_ITERATIONS must be an integer >= 2")
}
if (!is.finite(RF_SAMPLING_STABILITY_TREES) ||
    RF_SAMPLING_STABILITY_TREES < 1L) {
  stop("RF_SAMPLING_STABILITY_TREES must be an integer >= 1")
}
if (!is.finite(RF_SAMPLING_STABILITY_MAX_N_PER_DOMAIN) ||
    RF_SAMPLING_STABILITY_MAX_N_PER_DOMAIN < MIN_N_PER_DOMAIN + 2L) {
  stop(
    "RF_SAMPLING_STABILITY_MAX_N_PER_DOMAIN must be >= ",
    "MIN_N_PER_DOMAIN + 2"
  )
}
if (!is.finite(RF_SAMPLING_STABILITY_TRAIN_FRACTION) ||
    RF_SAMPLING_STABILITY_TRAIN_FRACTION <= 0 ||
    RF_SAMPLING_STABILITY_TRAIN_FRACTION >= 1) {
  stop("RF_SAMPLING_STABILITY_TRAIN_FRACTION must be between 0 and 1")
}
if (!is.finite(RF_SAMPLING_STABILITY_MIN_COVERAGE) ||
    RF_SAMPLING_STABILITY_MIN_COVERAGE < 0 ||
    RF_SAMPLING_STABILITY_MIN_COVERAGE > 1) {
  stop("RF_SAMPLING_STABILITY_MIN_COVERAGE must be between 0 and 1")
}
if (!is.finite(RF_SAMPLING_STABILITY_MIN_SIGN_AGREEMENT) ||
    RF_SAMPLING_STABILITY_MIN_SIGN_AGREEMENT < 0 ||
    RF_SAMPLING_STABILITY_MIN_SIGN_AGREEMENT > 1) {
  stop("RF_SAMPLING_STABILITY_MIN_SIGN_AGREEMENT must be between 0 and 1")
}
if (!is.finite(RF_SAMPLING_STABILITY_DELTA_TOLERANCE) ||
    RF_SAMPLING_STABILITY_DELTA_TOLERANCE < 0) {
  stop("RF_SAMPLING_STABILITY_DELTA_TOLERANCE must be nonnegative")
}
if (RF_SAMPLING_STABILITY_ENABLED && !RF_REAL_TO_GENERATED_ENABLED) {
  stop(
    "RF_SAMPLING_STABILITY_ENABLED requires ",
    "RF_REAL_TO_GENERATED_ENABLED"
  )
}

# Repeated training-only downsampling is diagnostic and off by default.
# For a single post-hoc task, enable RF_DOWNSAMPLE_ENABLED. For an array run,
# additionally set RF_DOWNSAMPLE_MODEL_IDS to a comma-separated allowlist so
# that only the requested model tasks execute the diagnostic.
RF_DOWNSAMPLE_ENABLED <- tolower(trimws(Sys.getenv(
  "RF_DOWNSAMPLE_ENABLED", unset = "false"
))) %in% c("1", "true", "t", "yes", "y")
RF_DOWNSAMPLE_MODEL_IDS_RAW <- trimws(Sys.getenv(
  "RF_DOWNSAMPLE_MODEL_IDS", unset = ""
))
RF_DOWNSAMPLE_MODEL_IDS <- if (nzchar(RF_DOWNSAMPLE_MODEL_IDS_RAW)) {
  unique(trimws(strsplit(RF_DOWNSAMPLE_MODEL_IDS_RAW, ",", fixed = TRUE)[[1]]))
} else {
  character()
}
if (any(!grepl("^[A-Za-z0-9._-]+$", RF_DOWNSAMPLE_MODEL_IDS))) {
  stop("RF_DOWNSAMPLE_MODEL_IDS contains an invalid model identifier")
}
RF_DOWNSAMPLE_RUN_THIS_MODEL <- RF_DOWNSAMPLE_ENABLED &&
  (length(RF_DOWNSAMPLE_MODEL_IDS) == 0 ||
     SELECTED_MODEL_ID %in% RF_DOWNSAMPLE_MODEL_IDS)
RF_DOWNSAMPLE_REPS <- suppressWarnings(as.integer(Sys.getenv(
  "RF_DOWNSAMPLE_REPS", unset = "10"
)))
RF_DOWNSAMPLE_TARGET_RAW <- trimws(Sys.getenv(
  "RF_DOWNSAMPLE_TARGET_N", unset = ""
))
RF_DOWNSAMPLE_TARGET_N <- if (nzchar(RF_DOWNSAMPLE_TARGET_RAW)) {
  suppressWarnings(as.integer(RF_DOWNSAMPLE_TARGET_RAW))
} else {
  NA_integer_
}
RF_DOWNSAMPLE_CONF <- 0.95

# Authentic forests are reused only when the exact RF predictor panel and
# language coverage match. The cache namespace also contains the authentic-data
# fingerprint, preprocessing configuration, setting, fold/source and seed.
RF_AUTHENTIC_CACHE_ENABLED <- tolower(trimws(Sys.getenv(
  "RF_AUTHENTIC_CACHE_ENABLED", unset = "true"
))) %in% c("1", "true", "t", "yes", "y")
RF_AUTHENTIC_CACHE_DIR <- trimws(Sys.getenv(
  "RF_AUTHENTIC_CACHE_DIR",
  unset = file.path(RF_OUTPUT_ROOT, "_authentic_forest_cache")
))
RF_AUTHENTIC_CACHE_WAIT_SECONDS <- suppressWarnings(as.numeric(Sys.getenv(
  "RF_AUTHENTIC_CACHE_WAIT_SECONDS", unset = "21600"
)))
if (!is.finite(RF_AUTHENTIC_CACHE_WAIT_SECONDS) ||
    RF_AUTHENTIC_CACHE_WAIT_SECONDS < 0) {
  stop("RF_AUTHENTIC_CACHE_WAIT_SECONDS must be >= 0")
}

# Shallow CART models mirror the train/test structure of every RF setting.
# They are deliberately approximate diagnostic models rather than exact
# explanations of the forests. Set RF_DECISION_TREE_ENABLED=false to skip them.
RF_DECISION_TREE_ENABLED <- tolower(trimws(Sys.getenv(
  "RF_DECISION_TREE_ENABLED", unset = "false"
))) %in% c("1", "true", "t", "yes", "y")
DT_CP <- suppressWarnings(as.numeric(Sys.getenv("RF_DT_CP", unset = "0.01")))
DT_MAXDEPTH <- suppressWarnings(as.integer(Sys.getenv(
  "RF_DT_MAXDEPTH", unset = "4"
)))
DT_MINBUCKET <- suppressWarnings(as.integer(Sys.getenv(
  "RF_DT_MINBUCKET", unset = "10"
)))
DT_MINSPLIT <- suppressWarnings(as.integer(Sys.getenv(
  "RF_DT_MINSPLIT", unset = "20"
)))

if (!is.finite(DT_CP) || DT_CP < 0) stop("RF_DT_CP must be a number >= 0")
if (!is.finite(DT_MAXDEPTH) || DT_MAXDEPTH < 1 || DT_MAXDEPTH > 30) {
  stop("RF_DT_MAXDEPTH must be an integer from 1 through 30")
}
if (!is.finite(DT_MINBUCKET) || DT_MINBUCKET < 1) {
  stop("RF_DT_MINBUCKET must be an integer >= 1")
}
if (!is.finite(DT_MINSPLIT) || DT_MINSPLIT < 2) {
  stop("RF_DT_MINSPLIT must be an integer >= 2")
}

if (RF_DOWNSAMPLE_RUN_THIS_MODEL &&
    (!is.finite(RF_DOWNSAMPLE_REPS) || RF_DOWNSAMPLE_REPS < 2)) {
  stop("RF_DOWNSAMPLE_REPS must be an integer >= 2")
}
if (nzchar(RF_DOWNSAMPLE_TARGET_RAW) && is.na(RF_DOWNSAMPLE_TARGET_N)) {
  stop("RF_DOWNSAMPLE_TARGET_N must be empty or a valid integer")
}
if (!is.na(RF_DOWNSAMPLE_TARGET_N) &&
    RF_DOWNSAMPLE_TARGET_N < MIN_N_PER_DOMAIN) {
  stop("RF_DOWNSAMPLE_TARGET_N must be >= MIN_N_PER_DOMAIN")
}

RF_RELEVANCE_B        <- 10
RF_RELEVANCE_ALPHA    <- 0.05
RF_RELEVANCE_ADJUST   <- "BH"
RF_IMPORTANCE_MODE    <- "permutation"

CLUSTER_Q_MAX         <- 0.05
CLUSTER_CUTOFF_CORR   <- 0.8
COR_METHOD            <- "spearman"
USE_DOMAIN_CENTERING  <- TRUE
MAX_N_PER_DOMAIN      <- 3000
META_JACCS            <- c(0.2, 0.3, 0.4, 0.6, 0.8)

RF_BIOBIO_MAX_N_PER_CLASS <- suppressWarnings(as.integer(Sys.getenv(
  "RF_BIOBIO_MAX_N_PER_CLASS", unset = as.character(MAX_N_PER_DOMAIN)
)))
RF_BIOBIO_TRANSFER_R <- suppressWarnings(as.integer(Sys.getenv(
  "RF_BIOBIO_TRANSFER_R", unset = as.character(TRANSFER_R)
)))
if (!is.finite(RF_BIOBIO_MAX_N_PER_CLASS) ||
    RF_BIOBIO_MAX_N_PER_CLASS < MIN_N_PER_DOMAIN) {
  stop("RF_BIOBIO_MAX_N_PER_CLASS must be an integer >= MIN_N_PER_DOMAIN")
}
if (!is.finite(RF_BIOBIO_TRANSFER_R) || RF_BIOBIO_TRANSFER_R < 1) {
  stop("RF_BIOBIO_TRANSFER_R must be an integer >= 1")
}

FILE_RF_LANG_CV       <- file.path(OUTPUT_DIR, "rf_lang_cv.rds")
FILE_RF_AGNOSTIC      <- file.path(OUTPUT_DIR, "rf_agnostic.rds")
FILE_RF_LOLO          <- file.path(OUTPUT_DIR, "rf_lolo.rds")
FILE_RF_TRANSFER      <- file.path(OUTPUT_DIR, "rf_transfer_ci.rds")
FILE_RF_BALANCE_AUDIT <- file.path(OUTPUT_DIR, "rf_downsample_balance_audit.rds")
FILE_RF_LANG_CV_DS    <- file.path(OUTPUT_DIR, "rf_lang_cv_downsampled.rds")
FILE_RF_LANG_CV_DS_RAW <- file.path(OUTPUT_DIR, "rf_lang_cv_downsampled_raw.rds")
FILE_RF_AGNOSTIC_DS   <- file.path(OUTPUT_DIR, "rf_agnostic_downsampled.rds")
FILE_RF_AGNOSTIC_DS_RAW <- file.path(OUTPUT_DIR, "rf_agnostic_downsampled_raw.rds")
FILE_RF_LOLO_DS       <- file.path(OUTPUT_DIR, "rf_lolo_downsampled.rds")
FILE_RF_LOLO_DS_RAW   <- file.path(OUTPUT_DIR, "rf_lolo_downsampled_raw.rds")
FILE_RF_TRANSFER_DS   <- file.path(OUTPUT_DIR, "rf_transfer_downsampled.rds")
FILE_RF_TRANSFER_DS_RAW <- file.path(OUTPUT_DIR, "rf_transfer_downsampled_raw.rds")
FILE_RF_LANG_FEAT     <- file.path(OUTPUT_DIR, "rf_lang_feat.rds")
FILE_RF_LANG_CLUSTERS <- file.path(OUTPUT_DIR, "rf_lang_clusters.rds")
FILE_RF_META          <- file.path(OUTPUT_DIR, "rf_meta_clusters.rds")
FILE_RF_META_SUMMARY  <- file.path(OUTPUT_DIR, "rf_meta_summary.rds")
FILE_RF_RUN_MANIFEST  <- file.path(OUTPUT_DIR, "rf_run_manifest.rds")
FILE_DT_LANG_CV       <- file.path(OUTPUT_DIR, "rf_tree_lang_cv.rds")
FILE_DT_AGNOSTIC      <- file.path(OUTPUT_DIR, "rf_tree_agnostic.rds")
FILE_DT_LOLO          <- file.path(OUTPUT_DIR, "rf_tree_lolo.rds")
FILE_DT_TRANSFER      <- file.path(OUTPUT_DIR, "rf_tree_transfer.rds")
FILE_DT_LANG_CV_DS    <- file.path(OUTPUT_DIR, "rf_tree_lang_cv_downsampled.rds")
FILE_DT_AGNOSTIC_DS   <- file.path(OUTPUT_DIR, "rf_tree_agnostic_downsampled.rds")
FILE_DT_LOLO_DS       <- file.path(OUTPUT_DIR, "rf_tree_lolo_downsampled.rds")
FILE_DT_TRANSFER_DS   <- file.path(OUTPUT_DIR, "rf_tree_transfer_downsampled.rds")
FILE_RF_SHARED_PANEL <- file.path(OUTPUT_DIR, "rf_paired_shared_feature_panel.rds")
FILE_RF_REAL_GEN_LANG <- file.path(OUTPUT_DIR, "rf_paired_real_generated_lang.rds")
FILE_RF_REAL_GEN_LANG_RAW <- file.path(OUTPUT_DIR, "rf_paired_real_generated_lang_raw.rds")
FILE_RF_SAMPLING_STABILITY <- file.path(
  OUTPUT_DIR, "rf_paired_lang_balanced_refit_stability.rds"
)
FILE_RF_SAMPLING_STABILITY_RAW <- file.path(
  OUTPUT_DIR, "rf_paired_lang_balanced_refit_stability_raw.rds"
)
FILE_RF_REAL_GEN_AGNOSTIC <- file.path(OUTPUT_DIR, "rf_paired_real_generated_agnostic.rds")
FILE_RF_REAL_GEN_AGNOSTIC_RAW <- file.path(OUTPUT_DIR, "rf_paired_real_generated_agnostic_raw.rds")
FILE_RF_REAL_GEN_LOLO <- file.path(OUTPUT_DIR, "rf_paired_real_generated_lolo.rds")
FILE_RF_REAL_GEN_LOLO_RAW <- file.path(OUTPUT_DIR, "rf_paired_real_generated_lolo_raw.rds")
FILE_RF_REAL_GEN_TRANSFER <- file.path(OUTPUT_DIR, "rf_paired_real_generated_transfer.rds")
FILE_RF_REAL_GEN_TRANSFER_RAW <- file.path(OUTPUT_DIR, "rf_paired_real_generated_transfer_raw.rds")
FILE_RF_AUTHENTIC_CACHE_AUDIT <- file.path(
  OUTPUT_DIR, "rf_authentic_forest_cache_audit.rds"
)
FILE_RF_BIOBIO_BALANCE_AUDIT <- file.path(
  OUTPUT_DIR, "rf_biobio_balance_audit.rds"
)
FILE_RF_BIOBIO_LANG <- file.path(OUTPUT_DIR, "rf_biobio_lang_cv.rds")
FILE_RF_BIOBIO_AGNOSTIC <- file.path(OUTPUT_DIR, "rf_biobio_agnostic.rds")
FILE_RF_BIOBIO_LOLO <- file.path(OUTPUT_DIR, "rf_biobio_lolo.rds")
FILE_RF_BIOBIO_TRANSFER <- file.path(
  OUTPUT_DIR, "rf_biobio_transfer_ci.rds"
)

# =========================================================
# LOGGING / PARALLEL
# =========================================================
format_elapsed <- function(start_time) {
  secs <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  sprintf("%.1f sec", secs)
}

log_step_start <- function(label) {
  message(sprintf("[%s] START: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), label))
  flush.console()
}

log_step_end <- function(label, start_time) {
  message(sprintf("[%s] END: %s | elapsed: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), label, format_elapsed(start_time)))
  flush.console()
}

log_item_start <- function(step_label, item_label, i = NA_integer_, n = NA_integer_) {
  if (is.finite(i) && is.finite(n)) {
    message(sprintf("[%s] %s | item %d/%d START: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), step_label, i, n, item_label))
  } else {
    message(sprintf("[%s] %s | START: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), step_label, item_label))
  }
  flush.console()
}

log_item_end <- function(step_label, item_label, start_time, i = NA_integer_, n = NA_integer_) {
  if (is.finite(i) && is.finite(n)) {
    message(sprintf("[%s] %s | item %d/%d END: %s | elapsed: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), step_label, i, n, item_label, format_elapsed(start_time)))
  } else {
    message(sprintf("[%s] %s | END: %s | elapsed: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), step_label, item_label, format_elapsed(start_time)))
  }
  flush.console()
}

set_future_plan <- function(workers = WORKERS) {
  invisible(workers)
}

# =========================================================
# IO HELPERS
# =========================================================
discover_completed_models <- function(root = Z_OUTPUT_ROOT) {
  if (!dir.exists(root)) stop("Z-score output root does not exist: ", root)

  model_dirs <- list.dirs(root, full.names = TRUE, recursive = FALSE)
  complete <- model_dirs[
    file.exists(file.path(model_dirs, "df_z.rds")) &
      file.exists(file.path(model_dirs, "feats_ok.rds")) &
      file.exists(file.path(model_dirs, "df_clean.rds"))
  ]
  sort(basename(complete))
}

load_tbl <- function(path, is_csv = FALSE) {
  if (!file.exists(path)) stop("File not found: ", path)
  if (is_csv) {
    readr::read_csv(path, show_col_types = FALSE)
  } else {
    readRDS(path)
  }
}

tag_model_id <- function(x, model_id = SELECTED_MODEL_ID) {
  if (is.data.frame(x)) {
    if ("model_id" %in% names(x)) {
      x %>% mutate(model_id = .env$model_id)
    } else {
      x %>% mutate(model_id = .env$model_id, .before = 1)
    }
  } else if (is.list(x) && all(c("observed", "null", "tested") %in% names(x))) {
    x[c("observed", "null", "tested")] <- lapply(
      x[c("observed", "null", "tested")],
      tag_model_id,
      model_id = model_id
    )
    x
  } else {
    x
  }
}

save_outputs <- function(x, path_base) {
  x <- tag_model_id(x)
  saveRDS(x, path_base)
  csv_path <- sub("\\.rds$", ".csv", path_base)
  
  if (is.data.frame(x)) {
    readr::write_csv(x, csv_path)
  } else if (is.list(x) && all(c("observed", "null", "tested") %in% names(x))) {
    readr::write_csv(x$observed, sub("\\.rds$", "_observed.csv", path_base))
    readr::write_csv(x$null,     sub("\\.rds$", "_null.csv", path_base))
    readr::write_csv(x$tested,   sub("\\.rds$", "_tested.csv", path_base))
  }
}

# Resume support is intentionally based only on file existence.  No success
# marker, input fingerprint, timestamp, or configuration comparison is used.
outputs_exist <- function(paths) {
  paths <- as.character(paths)
  length(paths) > 0 && all(file.exists(paths))
}

log_existing_outputs <- function(label, paths) {
  message(
    "Skipping ", label, "; output file",
    if (length(paths) == 1L) " exists: " else "s exist: ",
    paste(paths, collapse = ", ")
  )
}

# =========================================================
# HELPERS
# =========================================================
bal_acc2 <- function(truth, pred) {
  truth <- factor(truth, levels = c("biomedical", "general"))
  pred  <- factor(pred,  levels = c("biomedical", "general"))
  tab <- table(truth, pred)
  if (nrow(tab) < 2 || ncol(tab) < 2) return(NA_real_)
  sens <- tab["biomedical", "biomedical"] / sum(tab["biomedical", ])
  spec <- tab["general", "general"] / sum(tab["general", ])
  mean(c(sens, spec), na.rm = TRUE)
}

make_stratified_folds <- function(df, k = 5, strata_cols = c("domain"), seed = 123) {
  set.seed(seed)
  n <- nrow(df)
  fold <- rep(NA_integer_, n)
  
  strata <- df %>%
    mutate(.row_id = row_number()) %>%
    unite(".stratum", all_of(strata_cols), remove = FALSE) %>%
    group_by(.stratum) %>%
    summarise(rows = list(.row_id), .groups = "drop")
  
  for (i in seq_len(nrow(strata))) {
    r <- strata$rows[[i]]
    r <- sample(r, length(r), replace = FALSE)
    fold[r] <- rep(seq_len(k), length.out = length(r))
  }
  fold
}

empirical_p_greater <- function(obs, null_vec) {
  if (!is.finite(obs)) return(NA_real_)
  null_vec <- null_vec[is.finite(null_vec)]
  if (length(null_vec) == 0) return(NA_real_)
  (1 + sum(null_vec >= obs)) / (length(null_vec) + 1)
}

prep_global_feats_intersection <- function(feats_ok) {
  n_lang_total <- n_distinct(feats_ok$language)
  feats_ok %>%
    group_by(feature) %>%
    summarise(n_langs = n_distinct(language), .groups = "drop") %>%
    filter(n_langs == n_lang_total) %>%
    pull(feature)
}

md5_object <- function(x) {
  path <- tempfile("rf-cache-signature-", fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(x, path, version = 2)
  unname(tools::md5sum(path))
}

file_signature <- function(path) {
  info <- file.info(path)
  if (nrow(info) != 1 || is.na(info$size)) stop("Cannot fingerprint file: ", path)
  list(
    path = normalizePath(path, winslash = "/", mustWork = TRUE),
    size = unname(info$size),
    modified = format(info$mtime, "%Y-%m-%dT%H:%M:%OS6%z")
  )
}

sanitize_cache_key <- function(x) {
  gsub("[^A-Za-z0-9._-]+", "_", as.character(x))
}

new_authentic_cache <- function(enabled, root, namespace, wait_seconds) {
  list(
    enabled = isTRUE(enabled),
    root = root,
    namespace = namespace,
    wait_seconds = wait_seconds
  )
}

authentic_cache_file <- function(cache, key) {
  file.path(
    cache$root,
    cache$namespace,
    paste0(sanitize_cache_key(key), ".rds")
  )
}

fit_ranger_cached <- function(train, feats, trees, probability, seed,
                              num_threads, cache = NULL, cache_key = NULL) {
  fit_once <- function() {
    ranger(
      dependent.variable.name = "domain",
      data = train %>% select(domain, all_of(feats)),
      num.trees = trees,
      probability = probability,
      seed = seed,
      num.threads = num_threads,
      classification = TRUE,
      write.forest = TRUE
    )
  }

  if (is.null(cache) || !isTRUE(cache$enabled) ||
      is.null(cache_key) || !nzchar(cache_key)) {
    return(fit_once())
  }

  cache_file <- authentic_cache_file(cache, cache_key)
  dir.create(dirname(cache_file), recursive = TRUE, showWarnings = FALSE)
  if (file.exists(cache_file)) return(readRDS(cache_file))

  lock_dir <- paste0(cache_file, ".lock")
  acquired <- dir.create(lock_dir, showWarnings = FALSE)
  wait_started <- Sys.time()

  while (!acquired && !file.exists(cache_file)) {
    lock_info <- file.info(lock_dir)
    lock_age <- if (nrow(lock_info) == 1 && !is.na(lock_info$mtime)) {
      as.numeric(difftime(Sys.time(), lock_info$mtime, units = "secs"))
    } else {
      0
    }
    stale_after <- max(cache$wait_seconds, 43200)
    if (is.finite(lock_age) && lock_age > stale_after) {
      unlink(lock_dir, recursive = TRUE, force = TRUE)
      acquired <- dir.create(lock_dir, showWarnings = FALSE)
      next
    }

    waited <- as.numeric(difftime(Sys.time(), wait_started, units = "secs"))
    if (waited >= cache$wait_seconds) {
      warning("Timed out waiting for authentic RF cache key; fitting without cache: ",
              cache_key)
      return(fit_once())
    }
    Sys.sleep(2)
    acquired <- dir.create(lock_dir, showWarnings = FALSE)
  }

  if (!acquired && file.exists(cache_file)) return(readRDS(cache_file))
  on.exit(unlink(lock_dir, recursive = TRUE, force = TRUE), add = TRUE)
  if (file.exists(cache_file)) return(readRDS(cache_file))

  fit <- fit_once()
  temp_file <- tempfile(
    paste0(basename(cache_file), "-"),
    tmpdir = dirname(cache_file),
    fileext = ".tmp"
  )
  saveRDS(fit, temp_file, compress = FALSE)
  if (!file.rename(temp_file, cache_file)) {
    unlink(temp_file)
    if (!file.exists(cache_file)) {
      warning("Could not save authentic RF cache file: ", cache_file)
    }
  }
  fit
}

build_shared_feature_panel <- function(
    root = Z_OUTPUT_ROOT, requested_model_ids = character()) {
  model_ids <- if (length(requested_model_ids) > 0) {
    sort(unique(requested_model_ids))
  } else {
    discover_completed_models(root)
  }
  if (length(model_ids) == 0) {
    stop("No completed model directories are available for the shared panel")
  }

  required_ids <- unique(c(REAL_REFERENCE_MODEL_ID, SELECTED_MODEL_ID))
  missing_required <- setdiff(required_ids, model_ids)
  if (length(missing_required) > 0) {
    stop(
      "The shared feature panel model list omits required model(s): ",
      paste(missing_required, collapse = ", ")
    )
  }

  per_model <- lapply(model_ids, function(model_id) {
    path <- file.path(root, model_id, "feats_ok.rds")
    x <- load_tbl(path, is_csv = FALSE)
    if (!all(c("language", "feature") %in% names(x))) {
      stop("Invalid feats_ok for shared-panel model: ", model_id)
    }
    if ("model_id" %in% names(x)) {
      available <- unique(as.character(x$model_id))
      if (!model_id %in% available) {
        stop("feats_ok does not contain expected model '", model_id, "': ", path)
      }
      x <- x %>% filter(as.character(.data$model_id) == .env$model_id)
    }
    x %>%
      transmute(
        panel_model_id = model_id,
        language = as.character(language),
        feature = as.character(feature)
      ) %>%
      distinct()
  })
  availability <- bind_rows(per_model)
  cells <- availability %>% distinct(panel_model_id, language)
  n_cells <- nrow(cells)

  audit <- availability %>%
    count(feature, name = "n_model_language_cells") %>%
    mutate(
      total_model_language_cells = n_cells,
      coverage = n_model_language_cells / total_model_language_cells,
      is_shared = n_model_language_cells == total_model_language_cells
    ) %>%
    arrange(desc(is_shared), desc(coverage), feature)
  shared_features <- audit %>% filter(is_shared) %>% pull(feature)

  list(
    model_ids = model_ids,
    n_models = length(model_ids),
    n_model_language_cells = n_cells,
    features = shared_features,
    audit = audit,
    definition = paste(
      "Feature is present in feats_ok for every available model-language cell",
      "listed in model_ids."
    )
  )
}

prepare_biobio_balanced <- function(
    real_df, target_df, feats, max_n_per_class = 3000,
    min_per_class = 5, seed = 123) {
  prep_one <- function(x, source_class, internal_domain) {
    x %>%
      filter(as.character(.data$domain) == "biomedical") %>%
      transmute(
        language = as.character(language),
        source_class = source_class,
        domain = internal_domain,
        across(all_of(feats), as.numeric)
      ) %>%
      filter(if_all(all_of(feats), ~ is.finite(.x)))
  }

  combined <- bind_rows(
    prep_one(
      real_df,
      source_class = "real_biomedical",
      internal_domain = "general"
    ),
    prep_one(
      target_df,
      source_class = "target_biomedical",
      internal_domain = "biomedical"
    )
  )
  if (nrow(combined) == 0) {
    stop("No finite biomedical observations are available for bio-bio RFs")
  }

  class_levels <- c("real_biomedical", "target_biomedical")
  available <- combined %>%
    count(language, source_class, name = "n_available") %>%
    complete(
      language = sort(unique(combined$language)),
      source_class = class_levels,
      fill = list(n_available = 0L)
    )

  targets <- available %>%
    group_by(language) %>%
    summarise(
      n_real_available = n_available[source_class == "real_biomedical"],
      n_target_available = n_available[source_class == "target_biomedical"],
      balanced_n_per_class = min(n_available),
      .groups = "drop"
    ) %>%
    mutate(
      balanced_n_per_class = pmin(
        balanced_n_per_class, as.integer(max_n_per_class)
      ),
      included = balanced_n_per_class >= min_per_class,
      exclusion_reason = case_when(
        n_real_available == 0L ~ "no finite authentic biomedical observations",
        n_target_available == 0L ~ "no finite target biomedical observations",
        balanced_n_per_class < min_per_class ~
          "fewer than minimum observations per class",
        TRUE ~ NA_character_
      )
    )

  eligible <- targets %>% filter(included)
  if (nrow(eligible) == 0) {
    stop(
      "No language has at least ", min_per_class,
      " finite observations in both biomedical classes"
    )
  }

  sampled <- map_dfr(seq_len(nrow(eligible)), function(i) {
    lang <- eligible$language[[i]]
    target_n <- eligible$balanced_n_per_class[[i]]
    map_dfr(seq_along(class_levels), function(j) {
      cls <- class_levels[[j]]
      cell <- combined %>%
        filter(
          as.character(.data$language) == .env$lang,
          .data$source_class == .env$cls
        )
      set.seed(seed + i * 1000L + j)
      if (nrow(cell) > target_n) {
        cell <- cell[sample.int(nrow(cell), target_n), , drop = FALSE]
      }
      cell
    })
  }) %>%
    mutate(
      domain = factor(domain, levels = c("biomedical", "general"))
    )

  sampled_counts <- sampled %>%
    count(language, source_class, name = "n_sampled") %>%
    pivot_wider(
      names_from = source_class,
      values_from = n_sampled,
      names_prefix = "n_sampled_",
      values_fill = 0L
    )
  audit <- targets %>%
    left_join(sampled_counts, by = "language") %>%
    mutate(
      n_sampled_real_biomedical =
        replace_na(n_sampled_real_biomedical, 0L),
      n_sampled_target_biomedical =
        replace_na(n_sampled_target_biomedical, 0L),
      selected_model_id = SELECTED_MODEL_ID,
      real_reference_model_id = REAL_REFERENCE_MODEL_ID,
      feature_panel_hash = md5_object(sort(unique(as.character(feats)))),
      n_features = length(unique(feats)),
      max_n_per_class = as.integer(max_n_per_class),
      balance_seed = as.integer(seed),
      internal_general_label = "real_biomedical",
      internal_biomedical_label = "target_biomedical",
      .before = 1
    )

  list(data = sampled, audit = audit)
}

add_biobio_metadata <- function(x, setting) {
  x %>%
    mutate(
      analysis = "direct_biomedical_vs_biomedical_discriminability",
      setting = setting,
      real_reference_model_id = REAL_REFERENCE_MODEL_ID,
      target_model_id = SELECTED_MODEL_ID,
      internal_general_label = "real_biomedical",
      internal_biomedical_label = "target_biomedical",
      chance_balanced_accuracy = 0.5,
      feature_source = "global_shared_panel",
      sample_balance = "equal classes within language",
      .before = 1
    )
}

compute_rf_usable_cell_counts <- function(df_z, feats_by_lang, global_feats) {
  langs <- sort(unique(df_z$language))

  per_language <- map_dfr(langs, function(lang) {
    feats <- feats_by_lang %>%
      filter(.data$language == .env$lang) %>%
      pull(feature) %>%
      unique()
    if (length(feats) < 2) return(tibble())

    counts <- df_z %>%
      filter(.data$language == .env$lang, domain %in% c("biomedical", "general")) %>%
      filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
      mutate(domain = as.character(domain)) %>%
      count(domain, name = "n_usable")

    tibble(
      language = as.character(lang),
      domain = c("biomedical", "general")
    ) %>%
      left_join(counts, by = "domain") %>%
      mutate(n_usable = replace_na(n_usable, 0L)) %>%
      mutate(analysis = "per_language", n_features = length(feats), .before = 1)
  })

  pooled_counts <- df_z %>%
    filter(domain %in% c("biomedical", "general")) %>%
    filter(if_all(all_of(global_feats), ~ is.finite(.x))) %>%
    transmute(language = as.character(language), domain = as.character(domain)) %>%
    count(language, domain, name = "n_usable")

  pooled <- crossing(
    language = as.character(langs),
    domain = c("biomedical", "general")
  ) %>%
    left_join(pooled_counts, by = c("language", "domain")) %>%
    mutate(n_usable = replace_na(n_usable, 0L)) %>%
    mutate(analysis = "pooled_lolo_transfer", n_features = length(global_feats), .before = 1)

  bind_rows(per_language, pooled) %>%
    mutate(
      language = as.character(language),
      domain = as.character(domain)
    ) %>%
    arrange(analysis, language, domain)
}

resolve_rf_downsample_target <- function(usable_counts, requested_n = NA_integer_,
                                         k = 5, min_per_domain = 5) {
  if (nrow(usable_counts) == 0) stop("No complete RF rows are available for downsampling")

  minimum_usable <- min(usable_counts$n_usable)
  target_n <- if (is.na(requested_n)) minimum_usable else requested_n
  if (target_n > minimum_usable) {
    limiting <- usable_counts %>% filter(n_usable == minimum_usable)
    stop(
      "RF downsampling target n=", target_n,
      " exceeds the smallest complete model/language/domain cell (n=", minimum_usable, "): ",
      paste0(limiting$analysis, "/", limiting$language, "/", limiting$domain,
             collapse = ", ")
    )
  }
  if (target_n < min_per_domain) {
    stop("RF downsampling target n=", target_n,
         " is below MIN_N_PER_DOMAIN=", min_per_domain)
  }

  # With stratified folds, this many observations from a limiting full cell
  # are guaranteed to remain in every training fold.
  cv_train_target_n <- target_n - ceiling(target_n / k)
  if (cv_train_target_n < 2) {
    stop("RF CV training target is too small after reserving a test fold: ",
         cv_train_target_n)
  }

  list(
    target_n = target_n,
    cv_train_target_n = cv_train_target_n,
    audit = usable_counts %>%
      mutate(
        requested_target_n = requested_n,
        target_n_used = target_n,
        cv_train_target_n = cv_train_target_n,
        is_limiting_cell = n_usable == minimum_usable
      )
  )
}

downsample_training_groups <- function(df, group_cols, target_n, seed) {
  counts <- df %>% count(across(all_of(group_cols)), name = "n")
  if (nrow(counts) == 0 || any(counts$n < target_n)) {
    stop("A training group contains fewer than the requested downsampling target n=",
         target_n)
  }

  set.seed(seed)
  df %>%
    group_by(across(all_of(group_cols))) %>%
    slice_sample(n = target_n, replace = FALSE) %>%
    ungroup()
}

summarize_repeated_metric <- function(raw, group_cols, metric,
                                      conf = 0.95, output_name = metric) {
  if (nrow(raw) == 0) return(tibble())
  alpha <- (1 - conf) / 2

  per_rep <- raw %>%
    group_by(across(all_of(c(group_cols, "rep")))) %>%
    summarise(value = mean(.data[[metric]], na.rm = TRUE), .groups = "drop")

  per_rep %>%
    group_by(across(all_of(group_cols))) %>%
    summarise(
      repetitions = sum(is.finite(value)),
      mean = mean(value, na.rm = TRUE),
      median = median(value, na.rm = TRUE),
      sd = sd(value, na.rm = TRUE),
      lo = quantile(value, alpha, na.rm = TRUE, names = FALSE),
      hi = quantile(value, 1 - alpha, na.rm = TRUE, names = FALSE),
      .groups = "drop"
    ) %>%
    rename_with(~ paste0(output_name, "_", .x), c("mean", "median", "sd", "lo", "hi"))
}

# =========================================================
# MODEL HELPERS
# =========================================================
rf_predict_labels <- function(fit, newdata, probability = TRUE) {
  pr <- predict(fit, data = newdata)$predictions
  if (probability) {
    colnames(pr)[max.col(pr, ties.method = "first")]
  } else {
    as.character(pr)
  }
}

rf_kfold_cv <- function(df, feats, k = 5, trees = 500, seed = 123,
                        strata_cols = c("domain"), probability = TRUE,
                        num_threads = 1) {
  d <- df %>%
    select(all_of(unique(c(strata_cols, "domain", feats)))) %>%
    filter(domain %in% c("biomedical", "general")) %>%
    mutate(domain = factor(domain, levels = c("biomedical", "general"))) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x)))
  
  if (nrow(d) < 10) return(tibble())
  if (any(table(d$domain) < 2)) return(tibble())
  
  fold_id <- make_stratified_folds(d, k = k, strata_cols = strata_cols, seed = seed)
  out_cv <- vector("list", k)
  
  for (f in seq_len(k)) {
    tr <- d[fold_id != f, , drop = FALSE]
    te <- d[fold_id == f, , drop = FALSE]
    if (nrow(te) == 0) next
    
    fit <- ranger(
      dependent.variable.name = "domain",
      data = tr %>% select(domain, all_of(feats)),
      num.trees = trees,
      probability = probability,
      seed = seed + f,
      num.threads = num_threads,
      classification = TRUE
    )
    
    pred_lab <- rf_predict_labels(fit, te %>% select(all_of(feats)), probability = probability)
    out_cv[[f]] <- tibble(fold = f, bal_acc = bal_acc2(te$domain, pred_lab))
  }
  
  bind_rows(out_cv)
}

rf_kfold_cv_training_downsampled <- function(
    df, feats, k = 5, trees = 500, seed = 123, reps = 10,
    strata_cols = c("domain"), balance_cols = c("domain"),
    train_target_n, eval_group_col = NULL,
    probability = TRUE, num_threads = 1) {
  keep_cols <- unique(c(strata_cols, balance_cols, eval_group_col, "domain", feats))
  d <- df %>%
    select(all_of(keep_cols)) %>%
    filter(domain %in% c("biomedical", "general")) %>%
    mutate(domain = factor(domain, levels = c("biomedical", "general"))) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x)))

  if (nrow(d) < 10 || any(table(d$domain) < 2)) return(tibble())
  fold_id <- make_stratified_folds(d, k = k, strata_cols = strata_cols, seed = seed)
  out <- vector("list", reps * k)
  idx <- 0L

  for (r in seq_len(reps)) {
    for (f in seq_len(k)) {
      tr <- d[fold_id != f, , drop = FALSE]
      te <- d[fold_id == f, , drop = FALSE]
      if (nrow(te) == 0) next

      tr_bal <- downsample_training_groups(
        tr,
        group_cols = balance_cols,
        target_n = train_target_n,
        seed = seed + r * 10000L + f
      )

      fit <- ranger(
        dependent.variable.name = "domain",
        data = tr_bal %>% select(domain, all_of(feats)),
        num.trees = trees,
        probability = probability,
        seed = seed + r * 10000L + f + 1000000L,
        num.threads = num_threads,
        classification = TRUE
      )

      pred_lab <- rf_predict_labels(
        fit,
        te %>% select(all_of(feats)),
        probability = probability
      )

      macro_bal_acc <- NA_real_
      if (!is.null(eval_group_col)) {
        eval_tbl <- tibble(
          eval_group = as.character(te[[eval_group_col]]),
          truth = te$domain,
          pred = pred_lab
        ) %>%
          group_by(eval_group) %>%
          summarise(group_bal_acc = bal_acc2(truth, pred), .groups = "drop")
        macro_bal_acc <- mean(eval_tbl$group_bal_acc, na.rm = TRUE)
      }

      idx <- idx + 1L
      out[[idx]] <- tibble(
        rep = r,
        fold = f,
        bal_acc = bal_acc2(te$domain, pred_lab),
        macro_bal_acc = macro_bal_acc,
        train_target_n = train_target_n,
        n_train = nrow(tr_bal),
        n_test = nrow(te)
      )
    }
  }

  bind_rows(out)
}

rf_fit_importance_once <- function(d, feats, trees = 500, seed = 123,
                                   importance_mode = "permutation",
                                   probability = TRUE,
                                   num_threads = 1) {
  fit <- ranger(
    dependent.variable.name = "domain",
    data = d %>% select(domain, all_of(feats)),
    num.trees = trees,
    importance = importance_mode,
    probability = probability,
    seed = seed,
    num.threads = num_threads,
    classification = TRUE
  )
  
  imp <- fit$variable.importance
  tibble(feature = names(imp), importance = as.numeric(imp))
}

# =========================================================
# DIAGNOSTIC DECISION TREES
# =========================================================
# These deliberately shallow CART models use the same inputs and held-out
# evaluation structure as the RF analyses. They expose rough decision rules;
# they are not surrogate explanations of individual ranger predictions.
dt_fit_once <- function(d, feats, seed = 123,
                        cp = DT_CP, maxdepth = DT_MAXDEPTH,
                        minsplit = DT_MINSPLIT,
                        minbucket = DT_MINBUCKET) {
  train <- d %>%
    select(domain, all_of(feats)) %>%
    mutate(domain = factor(domain, levels = c("biomedical", "general")))

  set.seed(seed)
  rpart::rpart(
    domain ~ .,
    data = train,
    method = "class",
    parms = list(split = "gini"),
    control = rpart::rpart.control(
      cp = cp,
      maxdepth = maxdepth,
      minsplit = minsplit,
      minbucket = minbucket,
      xval = 0
    ),
    model = FALSE,
    x = FALSE,
    y = FALSE
  )
}

dt_predict_labels <- function(fit, newdata) {
  as.character(predict(fit, newdata = newdata, type = "class"))
}

dt_diagnostics <- function(fit) {
  frame <- fit$frame
  node_ids <- suppressWarnings(as.integer(row.names(frame)))
  split_vars <- frame$var[frame$var != "<leaf>"]
  list(
    root_feature = if (length(split_vars) > 0) as.character(frame$var[[1]]) else NA_character_,
    n_splits = length(split_vars),
    n_terminal_nodes = sum(frame$var == "<leaf>"),
    tree_depth = if (length(node_ids) > 0 && all(is.finite(node_ids))) {
      max(floor(log(node_ids, base = 2)))
    } else {
      NA_real_
    }
  )
}

new_dt_bundle <- function(setting, models, performance) {
  if (nrow(performance) == 0 && !"model_id" %in% names(performance)) {
    performance$model_id <- character()
  } else {
    performance <- performance %>%
      mutate(model_id = SELECTED_MODEL_ID, .before = 1)
  }

  list(
    model_id = SELECTED_MODEL_ID,
    setting = setting,
    interpretation = paste(
      "Shallow standalone CART classifiers evaluated like the corresponding RF setting.",
      "Use them to inspect approximate discriminative rules, not as exact explanations",
      "of ranger predictions or feature effects."
    ),
    control = list(
      cp = DT_CP,
      maxdepth = DT_MAXDEPTH,
      minsplit = DT_MINSPLIT,
      minbucket = DT_MINBUCKET,
      split = "gini"
    ),
    performance = performance,
    models = models,
    created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
  )
}

save_dt_bundle <- function(bundle, path) {
  saveRDS(bundle, path)
  invisible(path)
}

dt_kfold_models <- function(
    df, feats, setting, k = 5, seed = 123,
    strata_cols = "domain", balance_cols = "domain",
    reps = 1, train_target_n = NULL, eval_group_col = NULL) {
  keep_cols <- unique(c(strata_cols, balance_cols, eval_group_col, "domain", feats))
  d <- df %>%
    select(all_of(keep_cols)) %>%
    filter(domain %in% c("biomedical", "general")) %>%
    mutate(domain = factor(domain, levels = c("biomedical", "general"))) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x)))

  if (nrow(d) < 10 || any(table(d$domain) < 2)) {
    return(new_dt_bundle(setting, list(), tibble()))
  }

  fold_id <- make_stratified_folds(d, k = k, strata_cols = strata_cols, seed = seed)
  reps_used <- if (is.null(train_target_n)) 1L else reps
  models <- list()
  performance <- list()
  idx <- 0L

  for (r in seq_len(reps_used)) {
    for (f in seq_len(k)) {
      tr <- d[fold_id != f, , drop = FALSE]
      te <- d[fold_id == f, , drop = FALSE]
      if (nrow(te) == 0) next

      if (is.null(train_target_n)) {
        tr_fit <- tr
        fit_seed <- seed + f
        model_key <- sprintf("fold_%02d", f)
      } else {
        tr_fit <- downsample_training_groups(
          tr,
          group_cols = balance_cols,
          target_n = train_target_n,
          seed = seed + r * 10000L + f
        )
        fit_seed <- seed + r * 10000L + f + 1000000L
        model_key <- sprintf("rep_%03d__fold_%02d", r, f)
      }

      fit <- dt_fit_once(tr_fit, feats = feats, seed = fit_seed)
      pred_lab <- dt_predict_labels(fit, te %>% select(all_of(feats)))

      tree_macro_bal_acc <- NA_real_
      if (!is.null(eval_group_col)) {
        eval_tbl <- tibble(
          eval_group = as.character(te[[eval_group_col]]),
          truth = te$domain,
          pred = pred_lab
        ) %>%
          group_by(eval_group) %>%
          summarise(group_bal_acc = bal_acc2(truth, pred), .groups = "drop")
        tree_macro_bal_acc <- mean(eval_tbl$group_bal_acc, na.rm = TRUE)
      }

      diag <- dt_diagnostics(fit)
      idx <- idx + 1L
      models[[model_key]] <- fit
      performance[[idx]] <- tibble(
        model_key = model_key,
        rep = if (is.null(train_target_n)) NA_integer_ else r,
        fold = f,
        tree_bal_acc = bal_acc2(te$domain, pred_lab),
        tree_macro_bal_acc = tree_macro_bal_acc,
        train_target_n = if (is.null(train_target_n)) NA_integer_ else train_target_n,
        n_train = nrow(tr_fit),
        n_test = nrow(te),
        n_features = length(feats),
        root_feature = diag$root_feature,
        n_splits = diag$n_splits,
        n_terminal_nodes = diag$n_terminal_nodes,
        tree_depth = diag$tree_depth
      )
    }
  }

  new_dt_bundle(setting, models, bind_rows(performance))
}

dt_per_language_cv <- function(
    df_z, feats_by_lang, setting, k = 5, seed = 123,
    min_per_domain = 5, reps = 1, train_target_n = NULL) {
  langs <- sort(unique(as.character(df_z$language)))
  models <- list()
  performance <- list()

  for (i in seq_along(langs)) {
    lang <- langs[[i]]
    feats <- feats_by_lang %>%
      filter(as.character(.data$language) == .env$lang) %>%
      pull(feature) %>%
      unique()
    if (length(feats) < 2) next

    d <- df_z %>%
      filter(as.character(.data$language) == .env$lang) %>%
      select(domain, all_of(feats)) %>%
      filter(domain %in% c("biomedical", "general")) %>%
      filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
      mutate(domain = factor(domain, levels = c("biomedical", "general")))
    if (nrow(d) == 0 || any(table(d$domain) < min_per_domain)) next

    lang_seed <- if (is.null(train_target_n)) seed + i else seed + i * 100000L
    one <- dt_kfold_models(
      df = d,
      feats = feats,
      setting = setting,
      k = k,
      seed = lang_seed,
      strata_cols = "domain",
      balance_cols = "domain",
      reps = reps,
      train_target_n = train_target_n
    )

    if (length(one$models) > 0) {
      new_names <- paste0(lang, "__", names(one$models))
      models <- c(models, setNames(one$models, new_names))
      performance[[lang]] <- one$performance %>%
        select(-model_id) %>%
        mutate(
          language = lang,
          model_key = paste0(lang, "__", model_key),
          .before = 1
        )
    }
  }

  new_dt_bundle(setting, models, bind_rows(performance))
}

dt_pooled_cv <- function(
    df_z, feats, setting, k = 5, seed = 123,
    min_per_lang_domain = 5, reps = 1, train_target_n = NULL) {
  d <- df_z %>%
    select(language, domain, all_of(feats)) %>%
    filter(domain %in% c("biomedical", "general")) %>%
    mutate(
      language = factor(language),
      domain = factor(domain, levels = c("biomedical", "general"))
    ) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x)))

  ok_langs <- d %>%
    count(language, domain, .drop = FALSE) %>%
    group_by(language) %>%
    summarise(min_n = min(n), .groups = "drop") %>%
    filter(min_n >= min_per_lang_domain) %>%
    pull(language) %>%
    as.character()
  d <- d %>% filter(as.character(language) %in% ok_langs)

  dt_kfold_models(
    df = d,
    feats = feats,
    setting = setting,
    k = k,
    seed = seed,
    strata_cols = c("language", "domain"),
    balance_cols = c("language", "domain"),
    reps = reps,
    train_target_n = train_target_n,
    eval_group_col = "language"
  )
}

dt_lolo_models <- function(
    df_z, feats, setting, seed = 123, min_per_domain = 5,
    reps = 1, target_n = NULL) {
  d <- df_z %>%
    select(language, domain, all_of(feats)) %>%
    filter(domain %in% c("biomedical", "general")) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
    mutate(
      language = factor(language),
      domain = factor(domain, levels = c("biomedical", "general"))
    )

  langs <- sort(unique(as.character(d$language)))
  reps_used <- if (is.null(target_n)) 1L else reps
  models <- list()
  performance <- list()
  idx <- 0L

  for (i in seq_along(langs)) {
    lang_out <- langs[[i]]
    tr_full <- d %>% filter(as.character(.data$language) != .env$lang_out)
    te <- d %>%
      filter(as.character(.data$language) == .env$lang_out) %>%
      select(domain, all_of(feats))
    if (any(table(tr_full$domain) < min_per_domain) ||
        any(table(te$domain) < min_per_domain)) next

    for (r in seq_len(reps_used)) {
      if (is.null(target_n)) {
        tr_fit <- tr_full
        fit_seed <- seed + i
        model_key <- paste0("heldout_", lang_out)
      } else {
        tr_fit <- downsample_training_groups(
          tr_full,
          group_cols = c("language", "domain"),
          target_n = target_n,
          seed = seed + i * 100000L + r
        )
        fit_seed <- seed + i * 100000L + r + 1000000L
        model_key <- sprintf("heldout_%s__rep_%03d", lang_out, r)
      }

      fit <- dt_fit_once(tr_fit, feats = feats, seed = fit_seed)
      pred_lab <- dt_predict_labels(fit, te %>% select(all_of(feats)))
      diag <- dt_diagnostics(fit)
      idx <- idx + 1L
      models[[model_key]] <- fit
      performance[[idx]] <- tibble(
        model_key = model_key,
        heldout_language = lang_out,
        rep = if (is.null(target_n)) NA_integer_ else r,
        tree_bal_acc = bal_acc2(te$domain, pred_lab),
        train_target_n = if (is.null(target_n)) NA_integer_ else target_n,
        n_train = nrow(tr_fit),
        n_test = nrow(te),
        n_features = length(feats),
        root_feature = diag$root_feature,
        n_splits = diag$n_splits,
        n_terminal_nodes = diag$n_terminal_nodes,
        tree_depth = diag$tree_depth
      )
    }
  }

  new_dt_bundle(setting, models, bind_rows(performance))
}

dt_transfer_models <- function(
    df_z, feats, setting, seed = 123, min_per_domain = 5,
    reps = 1, target_n = NULL, max_n_per_domain = NULL,
    rf_repetitions_represented = 1L) {
  set.seed(seed)
  data_by_lang <- df_z %>%
    select(language, domain, all_of(feats)) %>%
    filter(domain %in% c("general", "biomedical")) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
    mutate(domain = factor(domain, levels = c("biomedical", "general"))) %>%
    group_split(language)

  names(data_by_lang) <- vapply(
    data_by_lang, function(x) as.character(x$language[[1]]), character(1)
  )
  data_by_lang <- lapply(data_by_lang, function(x) {
    x <- x %>% select(-language)
    if (!is.null(max_n_per_domain)) {
      x <- x %>%
        group_by(domain) %>%
        group_modify(~ {
          if (nrow(.x) <= max_n_per_domain) {
            .x
          } else {
            .x[sample.int(nrow(.x), max_n_per_domain), , drop = FALSE]
          }
        }) %>%
        ungroup()
    }
    x
  })

  langs <- sort(names(data_by_lang))
  reps_used <- if (is.null(target_n)) 1L else reps
  models <- list()
  performance <- list()
  idx <- 0L

  for (src in langs) {
    dsrc_full <- data_by_lang[[src]]
    if (any(table(dsrc_full$domain) < min_per_domain)) next

    for (tgt in setdiff(langs, src)) {
      dtgt <- data_by_lang[[tgt]]
      if (any(table(dtgt$domain) < min_per_domain)) next

      for (r in seq_len(reps_used)) {
        idx <- idx + 1L
        if (is.null(target_n)) {
          dsrc <- dsrc_full
          fit_seed <- seed + idx
          model_key <- paste0(src, "__to__", tgt)
        } else {
          dsrc <- downsample_training_groups(
            dsrc_full,
            group_cols = "domain",
            target_n = target_n,
            seed = seed + idx
          )
          fit_seed <- seed + idx + 1000000L
          model_key <- sprintf("%s__to__%s__rep_%03d", src, tgt, r)
        }

        fit <- dt_fit_once(dsrc, feats = feats, seed = fit_seed)
        pred_lab <- dt_predict_labels(fit, dtgt %>% select(all_of(feats)))
        diag <- dt_diagnostics(fit)
        models[[model_key]] <- fit
        performance[[idx]] <- tibble(
          model_key = model_key,
          src_language = src,
          tgt_language = tgt,
          rep = if (is.null(target_n)) NA_integer_ else r,
          tree_bal_acc = bal_acc2(dtgt$domain, pred_lab),
          rf_repetitions_represented = if (is.null(target_n)) {
            rf_repetitions_represented
          } else {
            1L
          },
          train_target_n = if (is.null(target_n)) NA_integer_ else target_n,
          n_train = nrow(dsrc),
          n_test = nrow(dtgt),
          n_features = length(feats),
          root_feature = diag$root_feature,
          n_splits = diag$n_splits,
          n_terminal_nodes = diag$n_terminal_nodes,
          tree_depth = diag$tree_depth
        )
      }
    }
  }

  new_dt_bundle(setting, models, bind_rows(performance))
}

# =========================================================
# PRIMARY: PERFORMANCE
# =========================================================
rf_per_language_cv_parallel <- function(df_z, feats_by_lang,
                                        k = 5, trees = 500, seed = 123,
                                        min_per_domain = 5,
                                        probability = TRUE,
                                        workers = 1,
                                        num_threads = 1) {
  set_future_plan(workers = workers)
  langs <- sort(unique(df_z$language))
  
  res <- lapply(langs, function(lang) {
    lang_idx <- match(lang, langs)
    t_lang <- Sys.time()
    log_item_start("per-language CV", as.character(lang), lang_idx, length(langs))
    
    feats <- feats_by_lang %>%
      filter(.data$language == .env$lang) %>%
      pull(feature) %>%
      unique()
    
    if (length(feats) < 2) {
      log_item_end("per-language CV", paste0(lang, " [skipped: <2 feats]"), t_lang, lang_idx, length(langs))
      return(NULL)
    }
    
    d <- df_z %>%
      filter(.data$language == .env$lang) %>%
      select(domain, all_of(feats)) %>%
      filter(domain %in% c("biomedical", "general")) %>%
      filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
      mutate(domain = factor(domain, levels = c("biomedical", "general")))
    
    if (any(table(d$domain) < min_per_domain)) {
      log_item_end("per-language CV", paste0(lang, " [skipped: min_per_domain]"), t_lang, lang_idx, length(langs))
      return(NULL)
    }
    
    cv <- rf_kfold_cv(
      df = d,
      feats = feats,
      k = k,
      trees = trees,
      seed = seed + lang_idx,
      strata_cols = c("domain"),
      probability = probability,
      num_threads = num_threads
    )
    
    if (nrow(cv) == 0) {
      log_item_end("per-language CV", paste0(lang, " [skipped: empty cv]"), t_lang, lang_idx, length(langs))
      return(NULL)
    }
    
    out <- tibble(
      language = as.character(lang),
      mean_bal_acc = mean(cv$bal_acc, na.rm = TRUE),
      sd_bal_acc   = sd(cv$bal_acc, na.rm = TRUE),
      n_folds      = sum(is.finite(cv$bal_acc)),
      n_features   = length(feats)
    )
    
    log_item_end("per-language CV", as.character(lang), t_lang, lang_idx, length(langs))
    out
  })
  
  bind_rows(res) %>% arrange(desc(mean_bal_acc))
}

rf_per_language_cv_downsampled <- function(
    df_z, feats_by_lang, k = 5, trees = 500, seed = 123, reps = 10,
    train_target_n, conf = 0.95, probability = TRUE, num_threads = 1) {
  langs <- sort(unique(df_z$language))
  raw_all <- vector("list", length(langs))

  for (i in seq_along(langs)) {
    lang <- langs[[i]]
    t_lang <- Sys.time()
    log_item_start("per-language downsampled CV", as.character(lang), i, length(langs))

    feats <- feats_by_lang %>%
      filter(.data$language == .env$lang) %>%
      pull(feature) %>%
      unique()
    if (length(feats) < 2) next

    raw_all[[i]] <- rf_kfold_cv_training_downsampled(
      df = df_z %>% filter(.data$language == .env$lang),
      feats = feats,
      k = k,
      trees = trees,
      seed = seed + i * 100000L,
      reps = reps,
      strata_cols = "domain",
      balance_cols = "domain",
      train_target_n = train_target_n,
      probability = probability,
      num_threads = num_threads
    ) %>%
      mutate(language = as.character(lang), n_features = length(feats), .before = 1)

    log_item_end("per-language downsampled CV", as.character(lang),
                 t_lang, i, length(langs))
  }

  raw <- bind_rows(raw_all)
  summary <- summarize_repeated_metric(
    raw = raw,
    group_cols = "language",
    metric = "bal_acc",
    conf = conf,
    output_name = "bal_acc"
  ) %>%
    left_join(
      raw %>% distinct(language, train_target_n, n_features),
      by = "language"
    ) %>%
    arrange(desc(bal_acc_median))

  list(raw = raw, summary = summary)
}

rf_language_agnostic_cv <- function(df_z, feats, k = 5, trees = 500, seed = 123,
                                    min_per_lang_domain = 5,
                                    probability = TRUE,
                                    num_threads = 1) {
  d <- df_z %>%
    select(language, domain, all_of(feats)) %>%
    filter(domain %in% c("biomedical", "general")) %>%
    mutate(language = factor(language), domain = factor(domain, levels = c("biomedical", "general"))) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x)))
  
  ok_langs <- d %>%
    count(language, domain) %>%
    group_by(language) %>%
    summarise(min_n = min(n), .groups = "drop") %>%
    filter(min_n >= min_per_lang_domain) %>%
    pull(language)
  
  d <- d %>% filter(language %in% ok_langs)
  if (nrow(d) == 0) return(tibble())
  
  cv <- rf_kfold_cv(
    df = d,
    feats = feats,
    k = k,
    trees = trees,
    seed = seed,
    strata_cols = c("language", "domain"),
    probability = probability,
    num_threads = num_threads
  )
  
  if (nrow(cv) == 0) return(tibble())
  
  tibble(
    setting = "pooled_stratified_kfold",
    mean_bal_acc = mean(cv$bal_acc, na.rm = TRUE),
    sd_bal_acc   = sd(cv$bal_acc, na.rm = TRUE),
    n_folds      = sum(is.finite(cv$bal_acc)),
    n_languages  = n_distinct(d$language),
    n_features   = length(feats)
  )
}

rf_language_agnostic_cv_downsampled <- function(
    df_z, feats, k = 5, trees = 500, seed = 123, reps = 10,
    train_target_n, conf = 0.95, probability = TRUE, num_threads = 1) {
  raw <- rf_kfold_cv_training_downsampled(
    df = df_z,
    feats = feats,
    k = k,
    trees = trees,
    seed = seed,
    reps = reps,
    strata_cols = c("language", "domain"),
    balance_cols = c("language", "domain"),
    train_target_n = train_target_n,
    eval_group_col = "language",
    probability = probability,
    num_threads = num_threads
  )

  overall <- summarize_repeated_metric(
    raw, group_cols = character(), metric = "bal_acc",
    conf = conf, output_name = "bal_acc"
  )
  macro <- summarize_repeated_metric(
    raw, group_cols = character(), metric = "macro_bal_acc",
    conf = conf, output_name = "macro_bal_acc"
  )

  summary <- bind_cols(overall, macro %>% select(-repetitions)) %>%
    mutate(
      setting = "pooled_training_downsampled",
      train_target_n = train_target_n,
      n_languages = n_distinct(df_z$language),
      n_features = length(feats),
      .before = 1
    )

  list(raw = raw, summary = summary)
}

rf_language_agnostic_lolo <- function(df_z, feats, trees = 500, seed = 123,
                                      min_per_domain = 5,
                                      probability = TRUE,
                                      num_threads = 1) {
  d <- df_z %>%
    select(language, domain, all_of(feats)) %>%
    filter(domain %in% c("biomedical", "general")) %>%
    mutate(language = factor(language), domain = factor(domain, levels = c("biomedical", "general"))) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x)))
  
  langs <- sort(unique(d$language))
  out_perf <- vector("list", length(langs))
  
  for (i in seq_along(langs)) {
    lang_out <- langs[[i]]
    t_lang <- Sys.time()
    log_item_start("LOLO", as.character(lang_out), i, length(langs))
    
    tr <- d %>% filter(.data$language != .env$lang_out) %>% select(domain, all_of(feats))
    te <- d %>% filter(.data$language == .env$lang_out) %>% select(domain, all_of(feats))
    
    if (any(table(tr$domain) < min_per_domain) || any(table(te$domain) < min_per_domain)) {
      log_item_end("LOLO", paste0(lang_out, " [skipped]"), t_lang, i, length(langs))
      next
    }
    
    fit <- ranger(
      dependent.variable.name = "domain",
      data = tr,
      num.trees = trees,
      probability = probability,
      seed = seed + i,
      num.threads = num_threads,
      classification = TRUE
    )
    
    pred_lab <- rf_predict_labels(fit, te %>% select(all_of(feats)), probability = probability)
    out_perf[[i]] <- tibble(heldout_language = as.character(lang_out), bal_acc = bal_acc2(te$domain, pred_lab), n_test = nrow(te))
    
    log_item_end("LOLO", as.character(lang_out), t_lang, i, length(langs))
  }
  
  bind_rows(out_perf) %>% arrange(desc(bal_acc))
}

rf_language_agnostic_lolo_downsampled <- function(
    df_z, feats, trees = 500, seed = 123, reps = 10,
    target_n, conf = 0.95, probability = TRUE, num_threads = 1) {
  d <- df_z %>%
    select(language, domain, all_of(feats)) %>%
    filter(domain %in% c("biomedical", "general")) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
    mutate(
      language = factor(language),
      domain = factor(domain, levels = c("biomedical", "general"))
    )

  langs <- sort(unique(d$language))
  raw_all <- vector("list", length(langs) * reps)
  idx <- 0L

  for (i in seq_along(langs)) {
    lang_out <- langs[[i]]
    te <- d %>%
      filter(.data$language == .env$lang_out) %>%
      select(domain, all_of(feats))
    tr_full <- d %>% filter(.data$language != .env$lang_out)

    for (r in seq_len(reps)) {
      tr <- downsample_training_groups(
        tr_full,
        group_cols = c("language", "domain"),
        target_n = target_n,
        seed = seed + i * 100000L + r
      )

      fit <- ranger(
        dependent.variable.name = "domain",
        data = tr %>% select(domain, all_of(feats)),
        num.trees = trees,
        probability = probability,
        seed = seed + i * 100000L + r + 1000000L,
        num.threads = num_threads,
        classification = TRUE
      )
      pred_lab <- rf_predict_labels(
        fit, te %>% select(all_of(feats)), probability = probability
      )

      idx <- idx + 1L
      raw_all[[idx]] <- tibble(
        heldout_language = as.character(lang_out),
        rep = r,
        bal_acc = bal_acc2(te$domain, pred_lab),
        train_target_n = target_n,
        n_train = nrow(tr),
        n_test = nrow(te)
      )
    }
  }

  raw <- bind_rows(raw_all)
  summary <- summarize_repeated_metric(
    raw, group_cols = "heldout_language", metric = "bal_acc",
    conf = conf, output_name = "bal_acc"
  ) %>%
    left_join(
      raw %>% distinct(heldout_language, train_target_n, n_train, n_test),
      by = "heldout_language"
    ) %>%
    arrange(desc(bal_acc_median))

  list(raw = raw, summary = summary)
}

rf_transfer_with_ci_lowmem <- function(df_z, feats,
                                       trees = 500, seed = 123,
                                       R = 30,
                                       conf = 0.95,
                                       min_per_domain = 5,
                                       probability = FALSE,
                                       num_threads = 1,
                                       max_n_per_domain = NULL) {
  alpha <- (1 - conf) / 2
  set.seed(seed)
  
  data_by_lang <- df_z %>%
    select(language, domain, all_of(feats)) %>%
    filter(domain %in% c("general", "biomedical")) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
    mutate(domain = factor(domain, levels = c("biomedical", "general"))) %>%
    group_split(language)
  
  names(data_by_lang) <- vapply(data_by_lang, function(x) as.character(x$language[1]), character(1))
  data_by_lang <- lapply(data_by_lang, function(x) {
    x <- x %>% select(-language)
    
    if (!is.null(max_n_per_domain)) {
      x <- x %>%
        group_by(domain) %>%
        group_modify(~ {
          if (nrow(.x) <= max_n_per_domain) .x else .x[sample.int(nrow(.x), max_n_per_domain), , drop = FALSE]
        }) %>%
        ungroup()
    }
    
    x
  })
  
  langs <- sort(names(data_by_lang))
  n_fits <- length(langs) * R
  n_evaluations <- length(langs) * max(length(langs) - 1L, 0L) * R
  out <- vector("list", n_evaluations)
  idx_out <- 0L
  idx_fit <- 0L
  
  for (src_idx in seq_along(langs)) {
    src <- langs[[src_idx]]
    dsrc <- data_by_lang[[src]]
    if (any(table(dsrc$domain) < min_per_domain)) next

    for (r in seq_len(R)) {
      idx_fit <- idx_fit + 1L
      fit_label <- sprintf("%s (rep %d)", src, r)
      t_fit <- Sys.time()
      log_item_start("transfer source fit", fit_label, idx_fit, n_fits)

      # One source forest is reused for every target. This removes redundant
      # refitting and makes target comparisons use the same learned boundary.
      fit <- ranger(
        dependent.variable.name = "domain",
        data = dsrc,
        num.trees = trees,
        probability = probability,
        seed = seed + src_idx * 100000L + r,
        num.threads = num_threads,
        classification = TRUE
      )

      # Same-language transfer is excluded; within-language performance is
      # handled by cross-validation above.
      for (tgt in setdiff(langs, src)) {
        dtgt <- data_by_lang[[tgt]]
        if (any(table(dtgt$domain) < min_per_domain)) next

        idx_out <- idx_out + 1L
        label <- sprintf("%s -> %s (rep %d)", src, tgt, r)
        pred_lab <- rf_predict_labels(
          fit,
          dtgt %>% select(all_of(feats)),
          probability = probability
        )
        
        out[[idx_out]] <- tibble(
          src_language = src,
          tgt_language = tgt,
          rep = r,
          bal_acc = bal_acc2(dtgt$domain, pred_lab),
          n_train = nrow(dsrc),
          n_test  = nrow(dtgt)
        )
        rm(pred_lab)
      }

      rm(fit)
      gc(verbose = FALSE)
      log_item_end("transfer source fit", fit_label, t_fit, idx_fit, n_fits)
    }
  }
  
  raw <- bind_rows(out)
  
  if (nrow(raw) == 0) return(tibble())
  
  raw %>%
    group_by(src_language, tgt_language) %>%
    summarise(
      mean_bal_acc = mean(bal_acc, na.rm = TRUE),
      lo = quantile(bal_acc, probs = alpha, na.rm = TRUE),
      hi = quantile(bal_acc, probs = 1 - alpha, na.rm = TRUE),
      n = sum(is.finite(bal_acc)),
      med_n_train = median(n_train, na.rm = TRUE),
      med_n_test  = median(n_test,  na.rm = TRUE),
      .groups = "drop"
    )
}

rf_transfer_training_downsampled <- function(
    df_z, feats, trees = 500, seed = 123, reps = 10,
    target_n, conf = 0.95, probability = FALSE, num_threads = 1) {
  data_by_lang <- df_z %>%
    select(language, domain, all_of(feats)) %>%
    filter(domain %in% c("general", "biomedical")) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
    mutate(domain = factor(domain, levels = c("biomedical", "general"))) %>%
    group_split(language)

  names(data_by_lang) <- vapply(
    data_by_lang, function(x) as.character(x$language[1]), character(1)
  )
  data_by_lang <- lapply(data_by_lang, function(x) x %>% select(-language))
  langs <- sort(names(data_by_lang))
  n_fits <- length(langs) * reps
  n_evaluations <- length(langs) * max(length(langs) - 1L, 0L) * reps
  raw_all <- vector("list", n_evaluations)
  idx <- 0L
  fit_idx <- 0L

  for (src_idx in seq_along(langs)) {
    src <- langs[[src_idx]]
    dsrc_full <- data_by_lang[[src]]

    for (r in seq_len(reps)) {
      fit_idx <- fit_idx + 1L
      label <- sprintf("%s (downsample rep %d)", src, r)
      t_job <- Sys.time()
      log_item_start("transfer downsampled source fit", label, fit_idx, n_fits)

      dsrc <- downsample_training_groups(
        dsrc_full,
        group_cols = "domain",
        target_n = target_n,
        seed = seed + src_idx * 100000L + r
      )

      fit <- ranger(
        dependent.variable.name = "domain",
        data = dsrc,
        num.trees = trees,
        probability = probability,
        seed = seed + 1000000L + src_idx * 100000L + r,
        num.threads = num_threads,
        classification = TRUE
      )

      for (tgt in setdiff(langs, src)) {
        dtgt <- data_by_lang[[tgt]]
        idx <- idx + 1L
        pred_lab <- rf_predict_labels(
          fit,
          dtgt %>% select(all_of(feats)),
          probability = probability
        )

        raw_all[[idx]] <- tibble(
          src_language = src,
          tgt_language = tgt,
          rep = r,
          bal_acc = bal_acc2(dtgt$domain, pred_lab),
          train_target_n = target_n,
          n_train = nrow(dsrc),
          n_test = nrow(dtgt)
        )
        rm(pred_lab)
      }

      rm(fit)
      gc(verbose = FALSE)
      log_item_end("transfer downsampled source fit",
                   label, t_job, fit_idx, n_fits)
    }
  }

  raw <- bind_rows(raw_all)
  summary <- summarize_repeated_metric(
    raw,
    group_cols = c("src_language", "tgt_language"),
    metric = "bal_acc",
    conf = conf,
    output_name = "bal_acc"
  ) %>%
    left_join(
      raw %>%
        distinct(src_language, tgt_language, train_target_n, n_train, n_test),
      by = c("src_language", "tgt_language")
    ) %>%
    arrange(src_language, desc(bal_acc_median), tgt_language)

  list(raw = raw, summary = summary)
}

# =========================================================
# EXTERNAL VALIDATION: REAL-TRAINED RF -> GENERATED TEST DATA
# =========================================================
safe_finite_mean <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) NA_real_ else mean(x)
}

safe_finite_sd <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 2) NA_real_ else stats::sd(x)
}

safe_finite_quantile <- function(x, prob) {
  x <- x[is.finite(x)]
  if (length(x) == 0) NA_real_ else stats::quantile(
    x, probs = prob, names = FALSE, na.rm = TRUE
  )
}

safe_finite_min <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else min(x)
}

safe_finite_max <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else max(x)
}

rf_external_eval_repeated <- function(
    train_df, test_df, feats, trees = 500, seed = 123, R = 5,
    probability = FALSE, num_threads = 1, eval_group_col = NULL) {
  train <- train_df %>%
    select(domain, all_of(feats)) %>%
    filter(domain %in% c("biomedical", "general")) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
    mutate(domain = factor(domain, levels = c("biomedical", "general")))

  test_keep <- unique(c(eval_group_col, "domain", feats))
  test <- test_df %>%
    select(all_of(test_keep)) %>%
    filter(domain %in% c("biomedical", "general")) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
    mutate(domain = factor(domain, levels = c("biomedical", "general")))

  train_counts <- table(train$domain)
  test_counts <- table(test$domain)
  if (nrow(train) == 0 || nrow(test) == 0 ||
      any(train_counts < MIN_N_PER_DOMAIN) ||
      any(test_counts < MIN_N_PER_DOMAIN)) {
    return(tibble())
  }

  out <- vector("list", R)
  for (r in seq_len(R)) {
    fit <- ranger(
      dependent.variable.name = "domain",
      data = train,
      num.trees = trees,
      probability = probability,
      seed = seed + r,
      num.threads = num_threads,
      classification = TRUE
    )
    pred_lab <- rf_predict_labels(
      fit,
      test %>% select(all_of(feats)),
      probability = probability
    )

    macro_bal_acc <- NA_real_
    if (!is.null(eval_group_col)) {
      macro_tbl <- tibble(
        eval_group = as.character(test[[eval_group_col]]),
        truth = test$domain,
        pred = pred_lab
      ) %>%
        group_by(eval_group) %>%
        summarise(group_bal_acc = bal_acc2(truth, pred), .groups = "drop")
      macro_bal_acc <- safe_finite_mean(macro_tbl$group_bal_acc)
    }

    out[[r]] <- tibble(
      rep = r,
      bal_acc = bal_acc2(test$domain, pred_lab),
      macro_bal_acc = macro_bal_acc,
      n_train = nrow(train),
      n_test = nrow(test),
      n_train_biomedical = unname(train_counts[["biomedical"]]),
      n_train_general = unname(train_counts[["general"]]),
      n_test_biomedical = unname(test_counts[["biomedical"]]),
      n_test_general = unname(test_counts[["general"]]),
      n_features = length(feats)
    )

    rm(fit, pred_lab)
    gc(verbose = FALSE)
  }

  bind_rows(out)
}

summarize_external_rf <- function(raw, group_cols = character(), conf = 0.95) {
  if (nrow(raw) == 0) return(tibble())
  alpha <- (1 - conf) / 2

  raw %>%
    group_by(across(all_of(group_cols))) %>%
    summarise(
      repetitions = sum(is.finite(bal_acc)),
      mean_bal_acc = safe_finite_mean(bal_acc),
      sd_bal_acc = safe_finite_sd(bal_acc),
      lo = safe_finite_quantile(bal_acc, alpha),
      hi = safe_finite_quantile(bal_acc, 1 - alpha),
      mean_macro_bal_acc = safe_finite_mean(macro_bal_acc),
      sd_macro_bal_acc = safe_finite_sd(macro_bal_acc),
      macro_lo = safe_finite_quantile(macro_bal_acc, alpha),
      macro_hi = safe_finite_quantile(macro_bal_acc, 1 - alpha),
      n_train = first(n_train),
      n_test = first(n_test),
      n_train_biomedical = first(n_train_biomedical),
      n_train_general = first(n_train_general),
      n_test_biomedical = first(n_test_biomedical),
      n_test_general = first(n_test_general),
      n_features = first(n_features),
      .groups = "drop"
    )
}

rf_real_to_generated_per_language <- function(
    real_df_z, generated_df_z, real_feats_by_lang,
    trees = 500, seed = 123, R = 5, conf = 0.95,
    probability = FALSE, num_threads = 1) {
  langs <- intersect(
    sort(unique(as.character(real_df_z$language))),
    sort(unique(as.character(generated_df_z$language)))
  )
  raw_all <- vector("list", length(langs))

  for (i in seq_along(langs)) {
    lang <- langs[[i]]
    t_lang <- Sys.time()
    log_item_start("real -> generated per-language", lang, i, length(langs))
    feats <- real_feats_by_lang %>%
      filter(as.character(.data$language) == .env$lang) %>%
      pull(feature) %>%
      unique()
    if (length(feats) < 2) {
      log_item_end("real -> generated per-language",
                   paste0(lang, " [skipped: <2 real features]"),
                   t_lang, i, length(langs))
      next
    }

    one <- rf_external_eval_repeated(
      train_df = real_df_z %>%
        filter(as.character(.data$language) == .env$lang),
      test_df = generated_df_z %>%
        filter(as.character(.data$language) == .env$lang),
      feats = feats,
      trees = trees,
      seed = seed + i * 100000L,
      R = R,
      probability = probability,
      num_threads = num_threads
    )
    if (nrow(one) > 0) {
      raw_all[[i]] <- one %>% mutate(language = lang, .before = 1)
    }
    log_item_end("real -> generated per-language", lang,
                 t_lang, i, length(langs))
  }

  raw <- bind_rows(raw_all) %>%
    mutate(
      train_model_id = REAL_REFERENCE_MODEL_ID,
      test_model_id = SELECTED_MODEL_ID,
      feature_source = "real_feats_ok"
    )
  summary <- summarize_external_rf(raw, "language", conf) %>%
    mutate(
      train_model_id = REAL_REFERENCE_MODEL_ID,
      test_model_id = SELECTED_MODEL_ID,
      feature_source = "real_feats_ok"
    ) %>%
    arrange(desc(mean_bal_acc), language)

  list(raw = raw, summary = summary)
}

rf_real_to_generated_pooled <- function(
    real_df_z, generated_df_z, feats,
    trees = 500, seed = 123, R = 5, conf = 0.95,
    min_per_lang_domain = 5,
    probability = FALSE, num_threads = 1) {
  train <- real_df_z %>%
    select(language, domain, all_of(feats)) %>%
    filter(domain %in% c("biomedical", "general")) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
    mutate(
      language = factor(language),
      domain = factor(domain, levels = c("biomedical", "general"))
    )
  train_langs <- train %>%
    count(language, domain, .drop = FALSE) %>%
    group_by(language) %>%
    summarise(min_n = min(n), .groups = "drop") %>%
    filter(min_n >= min_per_lang_domain) %>%
    pull(language) %>%
    as.character()
  train <- train %>% filter(as.character(language) %in% train_langs)

  test <- generated_df_z %>%
    select(language, domain, all_of(feats)) %>%
    filter(domain %in% c("biomedical", "general")) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
    mutate(
      language = factor(language),
      domain = factor(domain, levels = c("biomedical", "general"))
    )
  test_langs <- test %>%
    count(language, domain, .drop = FALSE) %>%
    group_by(language) %>%
    summarise(min_n = min(n), .groups = "drop") %>%
    filter(min_n >= min_per_lang_domain) %>%
    pull(language) %>%
    as.character()
  test <- test %>% filter(as.character(language) %in% test_langs)

  raw <- rf_external_eval_repeated(
    train_df = train,
    test_df = test,
    feats = feats,
    trees = trees,
    seed = seed,
    R = R,
    probability = probability,
    num_threads = num_threads,
    eval_group_col = "language"
  ) %>%
    mutate(
      setting = "pooled_real_to_generated",
      train_model_id = REAL_REFERENCE_MODEL_ID,
      test_model_id = SELECTED_MODEL_ID,
      n_train_languages = length(train_langs),
      n_test_languages = length(test_langs),
      feature_source = "real_global_intersection",
      .before = 1
    )

  summary <- summarize_external_rf(raw, character(), conf) %>%
    mutate(
      setting = "pooled_real_to_generated",
      train_model_id = REAL_REFERENCE_MODEL_ID,
      test_model_id = SELECTED_MODEL_ID,
      n_train_languages = length(train_langs),
      n_test_languages = length(test_langs),
      feature_source = "real_global_intersection",
      .before = 1
    )

  list(raw = raw, summary = summary)
}

rf_real_to_generated_lolo <- function(
    real_df_z, generated_df_z, feats,
    trees = 500, seed = 123, R = 5, conf = 0.95,
    probability = FALSE, num_threads = 1) {
  heldout_langs <- intersect(
    sort(unique(as.character(real_df_z$language))),
    sort(unique(as.character(generated_df_z$language)))
  )
  raw_all <- vector("list", length(heldout_langs))

  for (i in seq_along(heldout_langs)) {
    lang_out <- heldout_langs[[i]]
    t_lang <- Sys.time()
    log_item_start("real -> generated LOLO", lang_out, i, length(heldout_langs))
    one <- rf_external_eval_repeated(
      train_df = real_df_z %>%
        filter(as.character(.data$language) != .env$lang_out),
      test_df = generated_df_z %>%
        filter(as.character(.data$language) == .env$lang_out),
      feats = feats,
      trees = trees,
      seed = seed + i * 100000L,
      R = R,
      probability = probability,
      num_threads = num_threads
    )
    if (nrow(one) > 0) {
      raw_all[[i]] <- one %>%
        mutate(heldout_language = lang_out, .before = 1)
    }
    log_item_end("real -> generated LOLO", lang_out,
                 t_lang, i, length(heldout_langs))
  }

  raw <- bind_rows(raw_all) %>%
    mutate(
      train_model_id = REAL_REFERENCE_MODEL_ID,
      test_model_id = SELECTED_MODEL_ID,
      feature_source = "real_global_intersection"
    )
  summary <- summarize_external_rf(raw, "heldout_language", conf) %>%
    mutate(
      train_model_id = REAL_REFERENCE_MODEL_ID,
      test_model_id = SELECTED_MODEL_ID,
      feature_source = "real_global_intersection"
    ) %>%
    arrange(desc(mean_bal_acc), heldout_language)

  list(raw = raw, summary = summary)
}

rf_real_to_generated_transfer <- function(
    real_df_z, generated_df_z, feats,
    trees = 500, seed = 123, R = 5, conf = 0.95,
    min_per_domain = 5, probability = FALSE, num_threads = 1,
    max_n_per_domain = NULL) {
  common_langs <- intersect(
    sort(unique(as.character(real_df_z$language))),
    sort(unique(as.character(generated_df_z$language)))
  )
  raw_all <- list()
  idx <- 0L
  n_jobs <- length(common_langs) * max(length(common_langs) - 1L, 0L)

  for (i in seq_along(common_langs)) {
    src <- common_langs[[i]]
    dsrc <- real_df_z %>%
      filter(as.character(.data$language) == .env$src) %>%
      select(domain, all_of(feats)) %>%
      filter(domain %in% c("biomedical", "general")) %>%
      filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
      mutate(domain = factor(domain, levels = c("biomedical", "general")))

    if (!is.null(max_n_per_domain)) {
      set.seed(seed + i)
      dsrc <- dsrc %>%
        group_by(domain) %>%
        group_modify(~ {
          if (nrow(.x) <= max_n_per_domain) {
            .x
          } else {
            .x[sample.int(nrow(.x), max_n_per_domain), , drop = FALSE]
          }
        }) %>%
        ungroup()
    }
    if (nrow(dsrc) == 0 || any(table(dsrc$domain) < min_per_domain)) next

    for (tgt in setdiff(common_langs, src)) {
      idx <- idx + 1L
      label <- paste0(src, " -> ", tgt)
      t_job <- Sys.time()
      log_item_start("real -> generated transfer", label, idx, n_jobs)
      one <- rf_external_eval_repeated(
        train_df = dsrc,
        test_df = generated_df_z %>%
          filter(as.character(.data$language) == .env$tgt),
        feats = feats,
        trees = trees,
        seed = seed + idx * 100000L,
        R = R,
        probability = probability,
        num_threads = num_threads
      )
      if (nrow(one) > 0) {
        raw_all[[idx]] <- one %>%
          mutate(src_language = src, tgt_language = tgt, .before = 1)
      }
      log_item_end("real -> generated transfer", label, t_job, idx, n_jobs)
    }
  }

  raw <- bind_rows(raw_all) %>%
    mutate(
      train_model_id = REAL_REFERENCE_MODEL_ID,
      test_model_id = SELECTED_MODEL_ID,
      feature_source = "real_global_intersection"
    )
  summary <- summarize_external_rf(
    raw, c("src_language", "tgt_language"), conf
  ) %>%
    mutate(
      train_model_id = REAL_REFERENCE_MODEL_ID,
      test_model_id = SELECTED_MODEL_ID,
      feature_source = "real_global_intersection"
    ) %>%
    arrange(src_language, desc(mean_bal_acc), tgt_language)

  list(raw = raw, summary = summary)
}

# =========================================================
# PRIMARY PAIRED ANALYSIS
# =========================================================
# The upstream df_z files are standardized separately by model and language.
# For paired real-vs-generated evaluation, use raw df_clean features, estimate
# centers/scales from real reference data, and apply those parameters unchanged
# to both evaluation sets.
fit_real_reference_scaler <- function(df, feats) {
  langs <- sort(unique(as.character(df$language)))
  centers <- matrix(
    NA_real_, nrow = length(langs), ncol = length(feats),
    dimnames = list(langs, feats)
  )
  scales <- centers

  for (lang in langs) {
    x <- df %>%
      filter(as.character(.data$language) == .env$lang) %>%
      select(all_of(feats)) %>%
      as.data.frame()
    centers[lang, ] <- vapply(x, function(v) mean(v, na.rm = TRUE), numeric(1))
    scales[lang, ] <- vapply(x, function(v) stats::sd(v, na.rm = TRUE), numeric(1))
  }

  bad <- which(!is.finite(centers) | !is.finite(scales) | scales <= 0,
               arr.ind = TRUE)
  if (nrow(bad) > 0) {
    examples <- paste0(
      rownames(centers)[bad[, 1]], "/", colnames(centers)[bad[, 2]]
    )
    stop(
      "Real reference scaling is undefined for: ",
      paste(head(examples, 20), collapse = ", "),
      if (length(examples) > 20) " ..." else ""
    )
  }

  list(languages = langs, features = feats, centers = centers, scales = scales)
}

apply_real_reference_scaler <- function(df, scaler) {
  out <- df
  lang_idx <- match(as.character(out$language), scaler$languages)
  for (feature in scaler$features) {
    center <- scaler$centers[, feature][lang_idx]
    scale <- scaler$scales[, feature][lang_idx]
    out[[feature]] <- (out[[feature]] - center) / scale
  }
  out
}

macro_bal_acc_by_group <- function(truth, pred, group) {
  tibble(
    eval_group = as.character(group),
    truth = truth,
    pred = pred
  ) %>%
    group_by(eval_group) %>%
    summarise(group_bal_acc = bal_acc2(truth, pred), .groups = "drop") %>%
    summarise(value = safe_finite_mean(group_bal_acc)) %>%
    pull(value)
}

rf_paired_eval_repeated <- function(
    train_df, real_test_df, generated_test_df, feats,
    trees = 500, seed = 123, R = 5,
    min_train_per_domain = 5, min_real_test_per_domain = 1,
    min_generated_test_per_domain = 5,
    probability = FALSE, num_threads = 1, eval_group_col = NULL,
    cache = NULL, cache_key_prefix = NULL) {
  prep_train <- function(x) {
    x %>%
      select(domain, all_of(feats)) %>%
      filter(domain %in% c("biomedical", "general")) %>%
      filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
      mutate(domain = factor(domain, levels = c("biomedical", "general")))
  }
  prep_test <- function(x) {
    x %>%
      select(all_of(unique(c(eval_group_col, "domain", feats)))) %>%
      filter(domain %in% c("biomedical", "general")) %>%
      filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
      mutate(domain = factor(domain, levels = c("biomedical", "general")))
  }

  train <- prep_train(train_df)
  real_test <- prep_test(real_test_df)
  generated_test <- prep_test(generated_test_df)
  train_counts <- table(train$domain)
  real_counts <- table(real_test$domain)
  generated_counts <- table(generated_test$domain)
  if (nrow(train) == 0 || nrow(real_test) == 0 || nrow(generated_test) == 0 ||
      any(train_counts < min_train_per_domain) ||
      any(real_counts < min_real_test_per_domain) ||
      any(generated_counts < min_generated_test_per_domain)) {
    return(tibble())
  }

  out <- vector("list", R)
  for (r in seq_len(R)) {
    cache_key <- if (is.null(cache_key_prefix)) {
      NULL
    } else {
      paste0(cache_key_prefix, "__rep_", sprintf("%03d", r))
    }
    fit <- fit_ranger_cached(
      train = train,
      feats = feats,
      trees = trees,
      probability = probability,
      seed = seed + r,
      num_threads = num_threads,
      cache = cache,
      cache_key = cache_key
    )
    real_pred <- rf_predict_labels(
      fit, real_test %>% select(all_of(feats)), probability = probability
    )
    generated_pred <- rf_predict_labels(
      fit, generated_test %>% select(all_of(feats)), probability = probability
    )
    real_bal_acc <- bal_acc2(real_test$domain, real_pred)
    generated_bal_acc <- bal_acc2(generated_test$domain, generated_pred)

    real_macro_bal_acc <- NA_real_
    generated_macro_bal_acc <- NA_real_
    if (!is.null(eval_group_col)) {
      real_macro_bal_acc <- macro_bal_acc_by_group(
        real_test$domain, real_pred, real_test[[eval_group_col]]
      )
      generated_macro_bal_acc <- macro_bal_acc_by_group(
        generated_test$domain, generated_pred,
        generated_test[[eval_group_col]]
      )
    }

    out[[r]] <- tibble(
      rep = r,
      real_bal_acc = real_bal_acc,
      generated_bal_acc = generated_bal_acc,
      delta_bal_acc = generated_bal_acc - real_bal_acc,
      real_macro_bal_acc = real_macro_bal_acc,
      generated_macro_bal_acc = generated_macro_bal_acc,
      delta_macro_bal_acc = generated_macro_bal_acc - real_macro_bal_acc,
      n_train = nrow(train),
      n_real_test = nrow(real_test),
      n_generated_test = nrow(generated_test),
      n_train_biomedical = unname(train_counts[["biomedical"]]),
      n_train_general = unname(train_counts[["general"]]),
      n_real_test_biomedical = unname(real_counts[["biomedical"]]),
      n_real_test_general = unname(real_counts[["general"]]),
      n_generated_test_biomedical = unname(generated_counts[["biomedical"]]),
      n_generated_test_general = unname(generated_counts[["general"]]),
      n_features = length(feats)
    )
    rm(fit, real_pred, generated_pred)
    gc(verbose = FALSE)
  }
  bind_rows(out)
}

summarize_paired_rf <- function(raw, group_cols = character(), conf = 0.95) {
  if (nrow(raw) == 0) return(tibble())
  alpha <- (1 - conf) / 2
  has_fold <- "fold" %in% names(raw)
  raw %>%
    group_by(across(all_of(group_cols))) %>%
    summarise(
      evaluations = sum(is.finite(delta_bal_acc)),
      mean_real_bal_acc = safe_finite_mean(real_bal_acc),
      mean_generated_bal_acc = safe_finite_mean(generated_bal_acc),
      mean_delta_bal_acc = safe_finite_mean(delta_bal_acc),
      sd_delta_bal_acc = safe_finite_sd(delta_bal_acc),
      delta_lo = safe_finite_quantile(delta_bal_acc, alpha),
      delta_hi = safe_finite_quantile(delta_bal_acc, 1 - alpha),
      mean_real_macro_bal_acc = safe_finite_mean(real_macro_bal_acc),
      mean_generated_macro_bal_acc = safe_finite_mean(generated_macro_bal_acc),
      mean_delta_macro_bal_acc = safe_finite_mean(delta_macro_bal_acc),
      macro_delta_lo = safe_finite_quantile(delta_macro_bal_acc, alpha),
      macro_delta_hi = safe_finite_quantile(delta_macro_bal_acc, 1 - alpha),
      n_folds = if (has_fold) {
        n_distinct(fold)
      } else {
        NA_integer_
      },
      n_train = round(stats::median(n_train)),
      n_real_test = round(stats::median(n_real_test)),
      n_generated_test = round(stats::median(n_generated_test)),
      n_features = first(n_features),
      .groups = "drop"
    )
}

add_paired_metadata <- function(x, setting, scaler_scope) {
  x %>%
    mutate(
      analysis = "primary_paired_real_vs_generated",
      setting = setting,
      train_model_id = REAL_REFERENCE_MODEL_ID,
      generated_model_id = SELECTED_MODEL_ID,
      feature_source = "global_shared_panel",
      scaler_source = scaler_scope,
      .before = 1
    )
}

rf_paired_per_language_cv <- function(
    real_df, generated_df, feats, k = 5, trees = 500,
    seed = 123, R = 5, conf = 0.95, min_per_domain = 5,
    probability = FALSE, num_threads = 1, cache = NULL) {
  langs <- intersect(
    sort(unique(as.character(real_df$language))),
    sort(unique(as.character(generated_df$language)))
  )
  raw_all <- list()
  idx <- 0L

  for (i in seq_along(langs)) {
    lang <- langs[[i]]
    real_lang <- real_df %>%
      filter(as.character(.data$language) == .env$lang) %>%
      select(language, domain, all_of(feats)) %>%
      filter(domain %in% c("biomedical", "general")) %>%
      filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
      mutate(domain = factor(domain, levels = c("biomedical", "general")))
    generated_lang <- generated_df %>%
      filter(as.character(.data$language) == .env$lang) %>%
      select(language, domain, all_of(feats))
    if (nrow(real_lang) < 10 || any(table(real_lang$domain) < min_per_domain)) next

    fold_id <- make_stratified_folds(
      real_lang, k = k, strata_cols = "domain", seed = seed + i
    )
    for (f in seq_len(k)) {
      train_raw <- real_lang[fold_id != f, , drop = FALSE]
      real_test_raw <- real_lang[fold_id == f, , drop = FALSE]
      scaler <- fit_real_reference_scaler(train_raw, feats)
      one <- rf_paired_eval_repeated(
        train_df = apply_real_reference_scaler(train_raw, scaler),
        real_test_df = apply_real_reference_scaler(real_test_raw, scaler),
        generated_test_df = apply_real_reference_scaler(generated_lang, scaler),
        feats = feats,
        trees = trees,
        seed = seed + i * 100000L + f * 1000L,
        R = R,
        min_train_per_domain = min_per_domain,
        min_real_test_per_domain = 1,
        min_generated_test_per_domain = min_per_domain,
        probability = probability,
        num_threads = num_threads,
        cache = cache,
        cache_key_prefix = paste0(
          "per_language__", sanitize_cache_key(lang),
          "__fold_", sprintf("%02d", f)
        )
      )
      if (nrow(one) > 0) {
        idx <- idx + 1L
        raw_all[[idx]] <- one %>%
          mutate(language = lang, fold = f, .before = 1)
      }
    }
  }

  raw <- bind_rows(raw_all) %>%
    add_paired_metadata("per_language_real_kfold", "real_training_fold_by_language")
  summary <- summarize_paired_rf(raw, "language", conf) %>%
    add_paired_metadata("per_language_real_kfold", "real_training_fold_by_language")
  if (nrow(summary) > 0) {
    summary <- summary %>% arrange(desc(mean_delta_bal_acc), language)
  }
  list(raw = raw, summary = summary)
}

# =========================================================
# REPEATED BALANCED-REFIT SAMPLING STABILITY
# =========================================================
# This sensitivity analysis is separate from the primary paired endpoint.
# Every iteration draws a new balanced authentic sample, creates a fresh
# stratified train/held-out split, fits a smaller authentic RF, and balances
# the authentic/generated evaluation samples.
balanced_domain_indices <- function(domain, n_per_domain, seed) {
  domain <- factor(domain, levels = c("biomedical", "general"))
  set.seed(seed)
  unlist(lapply(levels(domain), function(level) {
    available <- which(domain == level)
    if (length(available) < n_per_domain) {
      stop(
        "Requested ", n_per_domain, " observations for domain '", level,
        "' but only ", length(available), " are available"
      )
    }
    sample(available, n_per_domain, replace = FALSE)
  }), use.names = FALSE)
}

stable_string_seed_offset <- function(x, modulus = 100000L) {
  values <- utf8ToInt(enc2utf8(as.character(x)))
  if (length(values) == 0L) return(0L)
  as.integer(sum(values * seq_along(values) * 131) %% modulus)
}

balanced_train_test_indices <- function(
    domain, max_n_per_domain, train_fraction, min_train, seed) {
  domain <- factor(domain, levels = c("biomedical", "general"))
  counts <- table(domain)
  pool_n <- min(as.integer(counts), as.integer(max_n_per_domain))
  train_n <- floor(pool_n * train_fraction)
  test_n <- pool_n - train_n

  if (!is.finite(pool_n) || train_n < min_train || test_n < 2L) {
    return(NULL)
  }

  set.seed(seed)
  split_by_domain <- lapply(levels(domain), function(level) {
    selected <- sample(which(domain == level), pool_n, replace = FALSE)
    list(
      train = selected[seq_len(train_n)],
      test = selected[train_n + seq_len(test_n)]
    )
  })

  list(
    train = unlist(lapply(split_by_domain, `[[`, "train"),
                   use.names = FALSE),
    test = unlist(lapply(split_by_domain, `[[`, "test"),
                  use.names = FALSE),
    pool_n_per_domain = as.integer(pool_n),
    train_n_per_domain = as.integer(train_n),
    heldout_n_per_domain = as.integer(test_n)
  )
}

summarize_balanced_refit_stability <- function(
    raw, full_summary, expected_iterations, confidence,
    min_coverage, min_sign_agreement, delta_tolerance) {
  if (nrow(raw) == 0L) return(tibble())
  alpha <- (1 - confidence) / 2

  full_values <- full_summary %>%
    select(
      language,
      full_mean_real_bal_acc = mean_real_bal_acc,
      full_mean_generated_bal_acc = mean_generated_bal_acc,
      full_mean_delta_bal_acc = mean_delta_bal_acc
    ) %>%
    distinct(language, .keep_all = TRUE)

  raw %>%
    left_join(full_values, by = "language") %>%
    group_by(language) %>%
    summarise(
      n_iterations = n_distinct(iteration),
      n_estimable_iterations = sum(is.finite(delta_bal_acc)),
      iteration_coverage = n_estimable_iterations / expected_iterations,
      balanced_pool_n_per_domain_min =
        safe_finite_min(balanced_pool_n_per_domain),
      balanced_pool_n_per_domain_max =
        safe_finite_max(balanced_pool_n_per_domain),
      train_n_per_domain_min = safe_finite_min(train_n_per_domain),
      train_n_per_domain_max = safe_finite_max(train_n_per_domain),
      evaluation_n_per_domain_min =
        safe_finite_min(evaluation_n_per_domain),
      evaluation_n_per_domain_max =
        safe_finite_max(evaluation_n_per_domain),
      mean_real_bal_acc = safe_finite_mean(real_bal_acc),
      median_real_bal_acc = safe_finite_quantile(real_bal_acc, 0.5),
      mean_generated_bal_acc = safe_finite_mean(generated_bal_acc),
      median_generated_bal_acc =
        safe_finite_quantile(generated_bal_acc, 0.5),
      mean_delta_bal_acc = safe_finite_mean(delta_bal_acc),
      median_delta_bal_acc = safe_finite_quantile(delta_bal_acc, 0.5),
      sd_delta_bal_acc = safe_finite_sd(delta_bal_acc),
      delta_q25 = safe_finite_quantile(delta_bal_acc, 0.25),
      delta_q75 = safe_finite_quantile(delta_bal_acc, 0.75),
      descriptive_delta_lo =
        safe_finite_quantile(delta_bal_acc, alpha),
      descriptive_delta_hi =
        safe_finite_quantile(delta_bal_acc, 1 - alpha),
      full_mean_real_bal_acc = first(full_mean_real_bal_acc),
      full_mean_generated_bal_acc =
        first(full_mean_generated_bal_acc),
      full_mean_delta_bal_acc = first(full_mean_delta_bal_acc),
      delta_sign_agreement = {
        full_sign <- sign(first(full_mean_delta_bal_acc))
        finite_delta <- delta_bal_acc[is.finite(delta_bal_acc)]
        if (!is.finite(full_sign) || full_sign == 0 ||
            length(finite_delta) == 0L) {
          NA_real_
        } else {
          mean(sign(finite_delta) == full_sign)
        }
      },
      delta_within_tolerance_rate = {
        full_delta <- first(full_mean_delta_bal_acc)
        finite_delta <- delta_bal_acc[is.finite(delta_bal_acc)]
        if (!is.finite(full_delta) || length(finite_delta) == 0L) {
          NA_real_
        } else {
          mean(abs(finite_delta - full_delta) <= delta_tolerance)
        }
      },
      failed_iterations = paste(
        unique(iteration_status[iteration_status != "ok"]),
        collapse = ","
      ),
      .groups = "drop"
    ) %>%
    mutate(
      stable_coverage = is.finite(iteration_coverage) &
        iteration_coverage >= min_coverage,
      stable_direction = is.finite(delta_sign_agreement) &
        delta_sign_agreement >= min_sign_agreement,
      stable_all = stable_coverage & stable_direction,
      stability_status = case_when(
        !stable_coverage ~ "low iteration coverage",
        !is.finite(delta_sign_agreement) ~
          "full-data delta is zero or unavailable",
        !stable_direction ~ "unstable direction",
        stable_all ~ "stable",
        TRUE ~ "not classified"
      ),
      analysis = "paired_real_vs_generated_balanced_refit_stability",
      setting = "per_language_repeated_balanced_holdout_refit",
      sampling_scope = paste(
        "New balanced authentic train/held-out split and new RF in",
        "every iteration; balanced authentic/generated evaluation."
      ),
      interval_note = paste(
        "descriptive_delta_lo/hi are subsampling quantiles,",
        "not formal confidence limits."
      ),
      min_iteration_coverage = min_coverage,
      min_delta_sign_agreement = min_sign_agreement,
      delta_tolerance = delta_tolerance,
      train_model_id = REAL_REFERENCE_MODEL_ID,
      generated_model_id = SELECTED_MODEL_ID,
      .before = 1
    ) %>%
    arrange(language)
}

rf_paired_balanced_refit_stability <- function(
    real_df, generated_df, feats, full_summary, trees, iterations,
    train_fraction, max_n_per_domain, confidence, min_per_domain,
    min_coverage, min_sign_agreement, delta_tolerance,
    split_seed, fit_seed, probability, num_threads, cache) {
  languages <- intersect(
    sort(unique(as.character(real_df$language))),
    sort(unique(as.character(generated_df$language)))
  )
  raw_all <- list()
  result_index <- 0L

  prep <- function(x) {
    x %>%
      select(domain, all_of(feats)) %>%
      filter(domain %in% c("biomedical", "general")) %>%
      filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
      mutate(domain = factor(
        domain, levels = c("biomedical", "general")
      ))
  }

  add_iteration_result <- function(
      language, iteration, pool_n = NA_integer_,
      train_n = NA_integer_, heldout_n = NA_integer_,
      evaluation_n = NA_integer_, real_bal_acc = NA_real_,
      generated_bal_acc = NA_real_, status = "ok",
      error_message = NA_character_) {
    tibble(
      language = language,
      iteration = iteration,
      balanced_pool_n_per_domain = pool_n,
      train_n_per_domain = train_n,
      heldout_n_per_domain = heldout_n,
      evaluation_n_per_domain = evaluation_n,
      real_bal_acc = real_bal_acc,
      generated_bal_acc = generated_bal_acc,
      delta_bal_acc = generated_bal_acc - real_bal_acc,
      iteration_status = status,
      error_message = error_message,
      n_features = length(feats),
      trees = trees,
      train_fraction = train_fraction,
      max_n_per_domain = max_n_per_domain
    )
  }

  for (i in seq_along(languages)) {
    language <- languages[[i]]
    language_seed_offset <- stable_string_seed_offset(language)
    item_started <- Sys.time()
    log_item_start(
      "balanced-refit stability", language, i, length(languages)
    )

    real_language <- real_df %>%
      filter(as.character(.data$language) == .env$language) %>%
      select(language, domain, all_of(feats)) %>%
      filter(domain %in% c("biomedical", "general")) %>%
      filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
      mutate(domain = factor(
        domain, levels = c("biomedical", "general")
      ))
    generated_language <- generated_df %>%
      filter(as.character(.data$language) == .env$language) %>%
      select(language, domain, all_of(feats)) %>%
      filter(domain %in% c("biomedical", "general")) %>%
      filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
      mutate(domain = factor(
        domain, levels = c("biomedical", "general")
      ))

    for (iteration in seq_len(iterations)) {
      iteration_split_seed <- split_seed +
        language_seed_offset * 1000L +
        iteration * 100L
      split <- balanced_train_test_indices(
        real_language$domain,
        max_n_per_domain = max_n_per_domain,
        train_fraction = train_fraction,
        min_train = min_per_domain,
        seed = iteration_split_seed
      )

      if (is.null(split)) {
        result_index <- result_index + 1L
        raw_all[[result_index]] <- add_iteration_result(
          language, iteration,
          status = "insufficient_authentic_data"
        )
        next
      }

      one <- tryCatch({
        train_raw <- real_language[split$train, , drop = FALSE]
        real_heldout_raw <- real_language[split$test, , drop = FALSE]
        scaler <- fit_real_reference_scaler(train_raw, feats)
        train <- prep(apply_real_reference_scaler(train_raw, scaler))
        real_heldout <- prep(apply_real_reference_scaler(
          real_heldout_raw, scaler
        ))
        generated_all <- prep(apply_real_reference_scaler(
          generated_language, scaler
        ))

        evaluation_n <- min(
          as.integer(table(real_heldout$domain)),
          as.integer(table(generated_all$domain))
        )
        if (!is.finite(evaluation_n) || evaluation_n < 2L) {
          add_iteration_result(
            language, iteration,
            pool_n = split$pool_n_per_domain,
            train_n = split$train_n_per_domain,
            heldout_n = split$heldout_n_per_domain,
            status = "insufficient_evaluation_data"
          )
        } else {
          evaluation_n <- as.integer(evaluation_n)
          real_index <- balanced_domain_indices(
            real_heldout$domain,
            evaluation_n,
            iteration_split_seed + 11L
          )
          generated_index <- balanced_domain_indices(
            generated_all$domain,
            evaluation_n,
            iteration_split_seed + 29L
          )
          cache_key <- paste0(
            "balanced_refit__",
            sanitize_cache_key(language),
            "__iteration_",
            sprintf("%03d", iteration)
          )
          fit <- fit_ranger_cached(
            train = train,
            feats = feats,
            trees = trees,
            probability = probability,
            seed = fit_seed +
              language_seed_offset * 1000L +
              iteration,
            num_threads = num_threads,
            cache = cache,
            cache_key = cache_key
          )
          real_prediction <- rf_predict_labels(
            fit,
            real_heldout[real_index, , drop = FALSE] %>%
              select(all_of(feats)),
            probability = probability
          )
          generated_prediction <- rf_predict_labels(
            fit,
            generated_all[generated_index, , drop = FALSE] %>%
              select(all_of(feats)),
            probability = probability
          )
          real_bal_acc <- bal_acc2(
            real_heldout$domain[real_index],
            real_prediction
          )
          generated_bal_acc <- bal_acc2(
            generated_all$domain[generated_index],
            generated_prediction
          )
          rm(fit, real_prediction, generated_prediction)

          add_iteration_result(
            language, iteration,
            pool_n = split$pool_n_per_domain,
            train_n = split$train_n_per_domain,
            heldout_n = split$heldout_n_per_domain,
            evaluation_n = evaluation_n,
            real_bal_acc = real_bal_acc,
            generated_bal_acc = generated_bal_acc
          )
        }
      }, error = function(error) {
        add_iteration_result(
          language, iteration,
          pool_n = split$pool_n_per_domain,
          train_n = split$train_n_per_domain,
          heldout_n = split$heldout_n_per_domain,
          status = "iteration_error",
          error_message = conditionMessage(error)
        )
      })

      result_index <- result_index + 1L
      raw_all[[result_index]] <- one
      gc(verbose = FALSE)
    }

    log_item_end(
      "balanced-refit stability",
      language,
      item_started,
      i,
      length(languages)
    )
  }

  raw <- bind_rows(raw_all) %>%
    mutate(
      analysis = "paired_real_vs_generated_balanced_refit_stability",
      setting = "per_language_repeated_balanced_holdout_refit",
      train_model_id = REAL_REFERENCE_MODEL_ID,
      generated_model_id = SELECTED_MODEL_ID,
      feature_source = "global_shared_panel",
      scaler_source = "iteration_authentic_training_sample",
      .before = 1
    )
  if (sum(is.finite(raw$delta_bal_acc)) == 0L) {
    stop(
      "Balanced-refit stability produced no estimable iterations. ",
      "Inspect iteration_status and error_message."
    )
  }

  list(
    raw = raw,
    summary = summarize_balanced_refit_stability(
      raw = raw,
      full_summary = full_summary,
      expected_iterations = iterations,
      confidence = confidence,
      min_coverage = min_coverage,
      min_sign_agreement = min_sign_agreement,
      delta_tolerance = delta_tolerance
    )
  )
}

rf_paired_pooled_cv <- function(
    real_df, generated_df, feats, k = 5, trees = 500,
    seed = 123, R = 5, conf = 0.95, min_per_lang_domain = 5,
    probability = FALSE, num_threads = 1, cache = NULL) {
  real <- real_df %>%
    select(language, domain, all_of(feats)) %>%
    filter(domain %in% c("biomedical", "general")) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
    mutate(domain = factor(domain, levels = c("biomedical", "general")))
  ok_real_langs <- real %>%
    mutate(language = factor(language)) %>%
    count(language, domain, .drop = FALSE) %>%
    group_by(language) %>%
    summarise(min_n = min(n), .groups = "drop") %>%
    filter(min_n >= min_per_lang_domain) %>%
    pull(language) %>%
    as.character()
  real <- real %>% filter(as.character(language) %in% ok_real_langs)
  generated <- generated_df %>%
    select(language, domain, all_of(feats)) %>%
    filter(as.character(language) %in% ok_real_langs)

  fold_id <- make_stratified_folds(
    real, k = k, strata_cols = c("language", "domain"), seed = seed
  )
  raw_all <- list()
  for (f in seq_len(k)) {
    train_raw <- real[fold_id != f, , drop = FALSE]
    real_test_raw <- real[fold_id == f, , drop = FALSE]
    scaler <- fit_real_reference_scaler(train_raw, feats)
    one <- rf_paired_eval_repeated(
      train_df = apply_real_reference_scaler(train_raw, scaler),
      real_test_df = apply_real_reference_scaler(real_test_raw, scaler),
      generated_test_df = apply_real_reference_scaler(generated, scaler),
      feats = feats,
      trees = trees,
      seed = seed + f * 1000L,
      R = R,
      min_train_per_domain = min_per_lang_domain,
      min_real_test_per_domain = 1,
      min_generated_test_per_domain = min_per_lang_domain,
      probability = probability,
      num_threads = num_threads,
      eval_group_col = "language",
      cache = cache,
      cache_key_prefix = paste0("pooled__fold_", sprintf("%02d", f))
    )
    if (nrow(one) > 0) raw_all[[f]] <- one %>% mutate(fold = f, .before = 1)
  }

  raw <- bind_rows(raw_all) %>%
    add_paired_metadata("pooled_real_kfold", "real_training_fold_by_language")
  summary <- summarize_paired_rf(raw, character(), conf) %>%
    add_paired_metadata("pooled_real_kfold", "real_training_fold_by_language")
  list(raw = raw, summary = summary)
}

rf_paired_lolo <- function(
    real_df, generated_df, feats, trees = 500, seed = 123,
    R = 5, conf = 0.95, min_per_domain = 5,
    probability = FALSE, num_threads = 1, cache = NULL) {
  real <- real_df %>%
    select(language, domain, all_of(feats)) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x)))
  generated <- generated_df %>% select(language, domain, all_of(feats))
  scaler <- fit_real_reference_scaler(real, feats)
  real <- apply_real_reference_scaler(real, scaler)
  generated <- apply_real_reference_scaler(generated, scaler)
  langs <- intersect(
    sort(unique(as.character(real$language))),
    sort(unique(as.character(generated$language)))
  )
  raw_all <- list()

  for (i in seq_along(langs)) {
    lang_out <- langs[[i]]
    one <- rf_paired_eval_repeated(
      train_df = real %>% filter(as.character(.data$language) != .env$lang_out),
      real_test_df = real %>% filter(as.character(.data$language) == .env$lang_out),
      generated_test_df = generated %>%
        filter(as.character(.data$language) == .env$lang_out),
      feats = feats,
      trees = trees,
      seed = seed + i * 100000L,
      R = R,
      min_train_per_domain = min_per_domain,
      min_real_test_per_domain = min_per_domain,
      min_generated_test_per_domain = min_per_domain,
      probability = probability,
      num_threads = num_threads,
      cache = cache,
      cache_key_prefix = paste0(
        "lolo__heldout_", sanitize_cache_key(lang_out)
      )
    )
    if (nrow(one) > 0) {
      raw_all[[i]] <- one %>% mutate(heldout_language = lang_out, .before = 1)
    }
  }

  raw <- bind_rows(raw_all) %>%
    add_paired_metadata("real_lolo", "full_real_reference_by_language")
  summary <- summarize_paired_rf(raw, "heldout_language", conf) %>%
    add_paired_metadata("real_lolo", "full_real_reference_by_language")
  if (nrow(summary) > 0) {
    summary <- summary %>% arrange(desc(mean_delta_bal_acc), heldout_language)
  }
  list(raw = raw, summary = summary)
}

rf_paired_transfer <- function(
    real_df, generated_df, feats, trees = 500, seed = 123,
    R = 5, conf = 0.95, min_per_domain = 5,
    probability = FALSE, num_threads = 1, max_n_per_domain = NULL,
    cache = NULL) {
  real <- real_df %>%
    select(language, domain, all_of(feats)) %>%
    filter(domain %in% c("biomedical", "general")) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
    mutate(domain = factor(domain, levels = c("biomedical", "general")))
  generated <- generated_df %>%
    select(language, domain, all_of(feats)) %>%
    filter(domain %in% c("biomedical", "general")) %>%
    filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
    mutate(domain = factor(domain, levels = c("biomedical", "general")))
  scaler <- fit_real_reference_scaler(real, feats)
  real <- apply_real_reference_scaler(real, scaler)
  generated <- apply_real_reference_scaler(generated, scaler)
  langs <- intersect(
    sort(unique(as.character(real$language))),
    sort(unique(as.character(generated$language)))
  )
  raw_all <- list()
  idx <- 0L
  fit_idx <- 0L
  n_fits <- length(langs) * R

  for (i in seq_along(langs)) {
    src <- langs[[i]]
    train <- real %>%
      filter(as.character(.data$language) == .env$src) %>%
      select(domain, all_of(feats))
    if (!is.null(max_n_per_domain)) {
      set.seed(seed + i)
      train <- train %>%
        group_by(domain) %>%
        group_modify(~ {
          if (nrow(.x) <= max_n_per_domain) .x else
            .x[sample.int(nrow(.x), max_n_per_domain), , drop = FALSE]
        }) %>%
        ungroup()
    }

    train_counts <- table(train$domain)
    if (nrow(train) == 0 || any(train_counts < min_per_domain)) next

    for (r in seq_len(R)) {
      fit_idx <- fit_idx + 1L
      fit_label <- sprintf("%s (rep %d)", src, r)
      t_fit <- Sys.time()
      log_item_start("paired transfer source fit", fit_label, fit_idx, n_fits)

      fit <- fit_ranger_cached(
        train = train,
        feats = feats,
        trees = trees,
        probability = probability,
        seed = seed + i * 100000L + r,
        num_threads = num_threads,
        cache = cache,
        cache_key = paste0(
          "transfer__source_", sanitize_cache_key(src),
          "__rep_", sprintf("%03d", r)
        )
      )

      for (tgt in setdiff(langs, src)) {
        real_test <- real %>%
          filter(as.character(.data$language) == .env$tgt) %>%
          select(domain, all_of(feats))
        generated_test <- generated %>%
          filter(as.character(.data$language) == .env$tgt) %>%
          select(domain, all_of(feats))
        real_counts <- table(real_test$domain)
        generated_counts <- table(generated_test$domain)
        if (nrow(real_test) == 0 || nrow(generated_test) == 0 ||
            any(real_counts < min_per_domain) ||
            any(generated_counts < min_per_domain)) {
          next
        }

        real_pred <- rf_predict_labels(
          fit, real_test %>% select(all_of(feats)), probability = probability
        )
        generated_pred <- rf_predict_labels(
          fit, generated_test %>% select(all_of(feats)),
          probability = probability
        )

        idx <- idx + 1L
        real_bal_acc <- bal_acc2(real_test$domain, real_pred)
        generated_bal_acc <- bal_acc2(generated_test$domain, generated_pred)
        raw_all[[idx]] <- tibble(
          src_language = src,
          tgt_language = tgt,
          rep = r,
          real_bal_acc = real_bal_acc,
          generated_bal_acc = generated_bal_acc,
          delta_bal_acc = generated_bal_acc - real_bal_acc,
          real_macro_bal_acc = NA_real_,
          generated_macro_bal_acc = NA_real_,
          delta_macro_bal_acc = NA_real_,
          n_train = nrow(train),
          n_real_test = nrow(real_test),
          n_generated_test = nrow(generated_test),
          n_train_biomedical = unname(train_counts[["biomedical"]]),
          n_train_general = unname(train_counts[["general"]]),
          n_real_test_biomedical = unname(real_counts[["biomedical"]]),
          n_real_test_general = unname(real_counts[["general"]]),
          n_generated_test_biomedical =
            unname(generated_counts[["biomedical"]]),
          n_generated_test_general = unname(generated_counts[["general"]]),
          n_features = length(feats)
        )
        rm(real_pred, generated_pred)
      }

      rm(fit)
      gc(verbose = FALSE)
      log_item_end("paired transfer source fit",
                   fit_label, t_fit, fit_idx, n_fits)
    }
  }

  raw <- bind_rows(raw_all) %>%
    add_paired_metadata("real_source_to_target", "full_real_reference_by_language")
  summary <- summarize_paired_rf(
    raw, c("src_language", "tgt_language"), conf
  ) %>%
    add_paired_metadata("real_source_to_target", "full_real_reference_by_language")
  if (nrow(summary) > 0) {
    summary <- summary %>%
      arrange(src_language, desc(mean_delta_bal_acc), tgt_language)
  }
  list(raw = raw, summary = summary)
}

# =========================================================
# SECONDARY: FEATURE RELEVANCE
# =========================================================
rf_per_language_feature_relevance_parallel <- function(df_z, feats_by_lang,
                                                       trees = 500,
                                                       seed = 123,
                                                       B = 200,
                                                       min_per_domain = 5,
                                                       alpha = 0.05,
                                                       adjust_method = "BH",
                                                       importance_mode = "permutation",
                                                       probability = TRUE,
                                                       workers = 1,
                                                       num_threads = 1) {
  set_future_plan(workers = workers)
  langs <- sort(unique(df_z$language))
  
  all_observed <- list()
  all_null <- list()
  all_tested <- list()
  
  for (i in seq_along(langs)) {
    lang <- langs[[i]]
    t_lang <- Sys.time()
    log_item_start("feature relevance", as.character(lang), i, length(langs))
    
    feats <- feats_by_lang %>%
      filter(.data$language == .env$lang) %>%
      pull(feature) %>%
      unique()
    
    if (length(feats) < 2) {
      log_item_end("feature relevance", paste0(lang, " [skipped: <2 feats]"), t_lang, i, length(langs))
      next
    }
    
    d <- df_z %>%
      filter(.data$language == .env$lang) %>%
      select(domain, all_of(feats)) %>%
      filter(domain %in% c("biomedical", "general")) %>%
      filter(if_all(all_of(feats), ~ is.finite(.x))) %>%
      mutate(domain = factor(domain, levels = c("biomedical", "general")))
    
    if (any(table(d$domain) < min_per_domain)) {
      log_item_end("feature relevance", paste0(lang, " [skipped: min_per_domain]"), t_lang, i, length(langs))
      next
    }
    
    obs_imp <- rf_fit_importance_once(
      d = d,
      feats = feats,
      trees = trees,
      seed = seed + i * 100000L,
      importance_mode = importance_mode,
      probability = probability,
      num_threads = num_threads
    ) %>%
      mutate(language = as.character(lang), perm_id = 0L)
    
    perm_list <- lapply(seq_len(B), function(b) {
      t_perm <- Sys.time()
      perm_label <- paste0("perm_", b)
      log_item_start(sprintf("feature relevance permutations [%s]", lang), perm_label, b, B)
      
      d_perm <- d
      d_perm$domain <- sample(d_perm$domain, size = nrow(d_perm), replace = FALSE)
      
      out <- rf_fit_importance_once(
        d = d_perm,
        feats = feats,
        trees = trees,
        seed = seed + i * 100000L + b,
        importance_mode = importance_mode,
        probability = probability,
        num_threads = num_threads
      ) %>%
        mutate(language = as.character(lang), perm_id = b)
      
      log_item_end(sprintf("feature relevance permutations [%s]", lang), perm_label, t_perm, b, B)
      out
    })
    
    null_imp <- bind_rows(perm_list)
    null_split <- split(null_imp$importance, null_imp$feature)
    
    sig_tbl <- obs_imp %>%
      transmute(
        language = language,
        feature = feature,
        observed_importance = importance,
        p_emp = purrr::map_dbl(
          feature,
          ~ empirical_p_greater(
            obs = obs_imp$importance[obs_imp$feature == .x][1],
            null_vec = null_split[[.x]]
          )
        )
      ) %>%
      mutate(
        q_emp = p.adjust(p_emp, method = adjust_method),
        is_relevant = is.finite(q_emp) & q_emp <= alpha
      ) %>%
      arrange(q_emp, desc(observed_importance), feature)
    
    all_observed[[lang]] <- obs_imp
    all_null[[lang]] <- null_imp
    all_tested[[lang]] <- sig_tbl
    
    log_item_end("feature relevance", as.character(lang), t_lang, i, length(langs))
  }
  
  list(
    observed = bind_rows(all_observed),
    null = bind_rows(all_null),
    tested = bind_rows(all_tested)
  )
}

cluster_rf_features_per_language <- function(df_z, rf_feature_tbl,
                                             q_max = 0.05,
                                             cutoff_corr = 0.8,
                                             method = "spearman",
                                             center_within_domain = TRUE,
                                             cap_per_domain = 3000,
                                             seed = 123) {
  set.seed(seed)
  langs <- sort(unique(rf_feature_tbl$language))
  clusters_all <- list()
  
  for (lang in langs) {
    sc <- rf_feature_tbl %>%
      filter(
        .data$language == .env$lang,
        .data$is_relevant,
        is.finite(.data$q_emp),
        .data$q_emp <= q_max
      )
    
    feats <- unique(sc$feature)
    if (length(feats) == 0) {
      clusters_all[[lang]] <- tibble()
      next
    }
    
    if (length(feats) == 1) {
      clusters_all[[lang]] <- sc %>%
        transmute(
          language = language,
          cluster_id = paste0(language, "__", feature),
          winner_feature = feature,
          winner_importance = observed_importance,
          winner_q = q_emp,
          n_members = 1L,
          cluster_importance_sum = observed_importance,
          cluster_importance_mean = observed_importance,
          cluster_importance_med = observed_importance,
          cluster_importance_max = observed_importance,
          members = feature
        )
      next
    }
    
    d <- df_z %>%
      filter(as.character(.data$language) == as.character(.env$lang)) %>%
      select(domain, all_of(feats)) %>%
      group_by(domain) %>%
      group_modify(~ if (nrow(.x) <= cap_per_domain) .x else .x[sample.int(nrow(.x), cap_per_domain), , drop = FALSE]) %>%
      ungroup()
    
    if (center_within_domain) {
      d <- d %>%
        group_by(domain) %>%
        mutate(across(all_of(feats), ~ . - mean(., na.rm = TRUE))) %>%
        ungroup()
    }
    
    X <- d %>% select(-domain) %>% as.matrix()
    if (ncol(X) < 2) next
    
    C <- suppressWarnings(cor(X, use = "pairwise.complete.obs", method = method))
    idx <- which(abs(C) >= cutoff_corr & upper.tri(C), arr.ind = TRUE)
    
    edges <- if (nrow(idx) == 0) {
      tibble(f1 = character(), f2 = character())
    } else {
      tibble(f1 = colnames(C)[idx[, 1]], f2 = colnames(C)[idx[, 2]])
    }
    
    comp_tbl <- alignment_components(nodes = colnames(C), edges_df = edges) %>%
      mutate(language = lang, cluster_id = paste0(lang, "__", comp)) %>%
      select(language, cluster_id, feature)
    
    comp_sc <- comp_tbl %>%
      left_join(sc %>% select(language, feature, observed_importance, p_emp, q_emp), by = c("language", "feature"))
    
    winners <- comp_sc %>%
      group_by(language, cluster_id) %>%
      arrange(q_emp, desc(observed_importance), feature, .by_group = TRUE) %>%
      slice(1) %>%
      ungroup() %>%
      transmute(language, cluster_id, winner_feature = feature, winner_importance = observed_importance, winner_q = q_emp)
    
    cluster_sum <- comp_sc %>%
      group_by(language, cluster_id) %>%
      summarise(
        n_members = n_distinct(feature),
        cluster_importance_sum  = sum(observed_importance, na.rm = TRUE),
        cluster_importance_mean = mean(observed_importance, na.rm = TRUE),
        cluster_importance_med  = median(observed_importance, na.rm = TRUE),
        cluster_importance_max  = max(observed_importance, na.rm = TRUE),
        members = paste(sort(unique(feature)), collapse = ", "),
        .groups = "drop"
      )
    
    clusters_all[[lang]] <- winners %>% left_join(cluster_sum, by = c("language", "cluster_id"))
  }
  
  bind_rows(clusters_all) %>% arrange(language, winner_q, desc(cluster_importance_sum), winner_feature)
}

build_rf_meta_clusters <- function(rf_lang_clusters, jaccs = c(0.2, 0.3, 0.4, 0.6, 0.8)) {
  cl_sets <- rf_lang_clusters %>%
    transmute(
      cluster_key = paste(language, cluster_id, sep = "||"),
      language = as.character(language),
      cluster_id = as.character(cluster_id),
      winner_feature,
      winner_importance,
      winner_q,
      members
    ) %>%
    distinct()
  
  if (nrow(cl_sets) == 0) return(tibble())
  
  members_list <- strsplit(cl_sets$members, "\\s*,\\s*")
  names(members_list) <- cl_sets$cluster_key
  
  jaccard <- function(a, b) {
    ia <- intersect(a, b)
    ua <- union(a, b)
    if (length(ua) == 0) return(0)
    length(ia) / length(ua)
  }
  
  out_all <- list()
  for (jacc in jaccs) {
    keys <- cl_sets$cluster_key
    n <- length(keys)
    if (n < 2) next
    
    edges <- list()
    k <- 0L
    for (i in 1:(n - 1)) {
      li <- cl_sets$language[i]
      ai <- members_list[[keys[i]]]
      for (j in (i + 1):n) {
        if (cl_sets$language[j] == li) next
        bj <- members_list[[keys[j]]]
        s <- jaccard(ai, bj)
        if (is.finite(s) && s >= jacc) {
          k <- k + 1L
          edges[[k]] <- c(keys[i], keys[j])
        }
      }
    }
    
    edges_df <- if (k == 0L) {
      tibble(f1 = character(), f2 = character())
    } else {
      tibble(f1 = vapply(edges, `[[`, character(1), 1), f2 = vapply(edges, `[[`, character(1), 2))
    }
    
    comp <- alignment_components(nodes = keys, edges_df = edges_df) %>% mutate(meta_cluster_id = paste0("jacc", jacc, "__", comp))
    
    meta_tbl <- cl_sets %>%
      left_join(comp %>% select(feature, comp, meta_cluster_id), by = c("cluster_key" = "feature")) %>%
      select(meta_cluster_id, comp, language, cluster_id, winner_feature, winner_importance, winner_q, members) %>%
      mutate(jacc = jacc)
    
    meta_tbl <- meta_tbl %>%
      group_by(jacc, meta_cluster_id) %>%
      mutate(meta_label = names(sort(table(winner_feature), decreasing = TRUE))[1]) %>%
      ungroup() %>%
      mutate(meta_cluster_id2 = paste0("jacc", jacc, "__", meta_label))
    
    out_all[[as.character(jacc)]] <- meta_tbl
  }
  
  bind_rows(out_all)
}

summarize_rf_meta_clusters <- function(rf_meta_clusters, alpha = 0.05) {
  if (nrow(rf_meta_clusters) == 0) return(tibble())
  
  rf_meta_clusters %>%
    group_by(jacc, meta_cluster_id) %>%
    summarise(
      n_langs = n_distinct(language),
      langs = paste(sort(unique(language)), collapse = ", "),
      n_sig = sum(is.finite(winner_q) & winner_q <= alpha),
      min_q = suppressWarnings(min(winner_q, na.rm = TRUE)),
      med_q = suppressWarnings(median(winner_q, na.rm = TRUE)),
      n_unique_winners = n_distinct(winner_feature),
      winners_by_lang = paste(paste0(language, ":", winner_feature), collapse = "; "),
      .groups = "drop"
    ) %>%
    arrange(jacc, desc(n_langs), min_q)
}

# =========================================================
# MAIN
# =========================================================
main <- function() {
  if (!nzchar(SELECTED_MODEL_ID)) {
    stop("ALIGNMENT_MODEL_ID must be set when running the multivariate analysis.")
  }

  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  
  message("Selected model: ", SELECTED_MODEL_ID)
  message("Loading z-scored inputs from: ", MODEL_INPUT_DIR)
  message("RF output directory: ", OUTPUT_DIR)
  message("SLURM CPUs / ranger threads: ", RANGER_NUM_THREADS)
  df_z <- load_tbl(INPUT_DF_Z, is_csv = DF_Z_IS_CSV)
  feats_ok <- load_tbl(INPUT_FEATS_OK, is_csv = FEATS_OK_IS_CSV)
  
  if (!all(c("language", "feature") %in% colnames(feats_ok))) stop("feats_ok must contain columns: language, feature")
  if (!all(c("language", "domain") %in% colnames(df_z))) stop("df_z must contain at least columns: language, domain")

  if ("model_id" %in% names(df_z)) {
    available <- unique(as.character(df_z$model_id))
    if (!SELECTED_MODEL_ID %in% available) {
      stop("df_z.rds does not contain selected model '", SELECTED_MODEL_ID, "'. Found: ", paste(available, collapse = ", "))
    }
    df_z <- df_z %>% filter(.data$model_id == .env$SELECTED_MODEL_ID)
  }

  if ("model_id" %in% names(feats_ok)) {
    available <- unique(as.character(feats_ok$model_id))
    if (!SELECTED_MODEL_ID %in% available) {
      stop("feats_ok.rds does not contain selected model '", SELECTED_MODEL_ID, "'. Found: ", paste(available, collapse = ", "))
    }
    feats_ok <- feats_ok %>% filter(.data$model_id == .env$SELECTED_MODEL_ID)
  }

  missing_features <- setdiff(unique(feats_ok$feature), names(df_z))
  if (length(missing_features) > 0) {
    stop(
      "feats_ok references features absent from df_z: ",
      paste(head(missing_features, 20), collapse = ", "),
      if (length(missing_features) > 20) " ..." else ""
    )
  }

  if (nrow(df_z) == 0 || nrow(feats_ok) == 0) {
    stop("Selected model has an empty df_z or feats_ok input.")
  }

  is_target_task <- SELECTED_MODEL_ID != REAL_REFERENCE_MODEL_ID
  run_real_to_generated <- RF_REAL_TO_GENERATED_ENABLED && is_target_task
  run_sampling_stability <- RF_SAMPLING_STABILITY_ENABLED &&
    run_real_to_generated
  run_biobio <- RF_BIOBIO_ENABLED && is_target_task
  run_target_comparisons <- run_real_to_generated || run_biobio
  real_df_z <- NULL
  real_feats_ok <- NULL
  real_df_clean <- NULL
  generated_df_clean <- NULL
  shared_panel <- NULL
  shared_features <- character()
  same_language_panel <- NA
  shared_feature_panel_hash <- NA_character_
  authentic_cache <- new_authentic_cache(
    enabled = FALSE,
    root = RF_AUTHENTIC_CACHE_DIR,
    namespace = "disabled",
    wait_seconds = RF_AUTHENTIC_CACHE_WAIT_SECONDS
  )
  stability_cache <- new_authentic_cache(
    enabled = FALSE,
    root = file.path(RF_AUTHENTIC_CACHE_DIR, "balanced_refit_stability"),
    namespace = "disabled",
    wait_seconds = RF_AUTHENTIC_CACHE_WAIT_SECONDS
  )

  if (RF_REAL_TO_GENERATED_ENABLED &&
      SELECTED_MODEL_ID == REAL_REFERENCE_MODEL_ID) {
    message(
      "Skipping paired real-vs-generated validation for the real reference task; ",
      "training and testing on the same real observations would leak data."
    )
  }
  if (RF_BIOBIO_ENABLED &&
      SELECTED_MODEL_ID == REAL_REFERENCE_MODEL_ID) {
    message(
      "Skipping direct bio-bio validation for the real reference task; ",
      "the same authentic biomedical observations cannot form both classes."
    )
  }

  if (run_target_comparisons) {
    message("Loading real reference inputs from: ", REAL_INPUT_DIR)
    real_df_z <- load_tbl(REAL_INPUT_DF_Z, is_csv = FALSE)
    real_feats_ok <- load_tbl(REAL_INPUT_FEATS_OK, is_csv = FALSE)
    real_df_clean <- load_tbl(REAL_INPUT_DF_CLEAN, is_csv = FALSE)
    generated_df_clean <- load_tbl(INPUT_DF_CLEAN, is_csv = FALSE)

    if (!all(c("language", "feature") %in% names(real_feats_ok))) {
      stop("The real feats_ok must contain columns: language, feature")
    }
    if (!all(c("language", "domain") %in% names(real_df_z))) {
      stop("The real df_z must contain at least columns: language, domain")
    }
    if (!all(c("language", "domain") %in% names(real_df_clean)) ||
        !all(c("language", "domain") %in% names(generated_df_clean))) {
      stop("Both real and generated df_clean must contain language and domain")
    }

    if ("model_id" %in% names(real_df_z)) {
      available <- unique(as.character(real_df_z$model_id))
      if (!REAL_REFERENCE_MODEL_ID %in% available) {
        stop(
          "The real df_z does not contain reference model '",
          REAL_REFERENCE_MODEL_ID, "'. Found: ", paste(available, collapse = ", ")
        )
      }
      real_df_z <- real_df_z %>%
        filter(.data$model_id == .env$REAL_REFERENCE_MODEL_ID)
    }
    if ("model_id" %in% names(real_df_clean)) {
      real_df_clean <- real_df_clean %>%
        filter(as.character(.data$model_id) == .env$REAL_REFERENCE_MODEL_ID)
    }
    if ("model_id" %in% names(generated_df_clean)) {
      generated_df_clean <- generated_df_clean %>%
        filter(as.character(.data$model_id) == .env$SELECTED_MODEL_ID)
    }
    if ("model_id" %in% names(real_feats_ok)) {
      available <- unique(as.character(real_feats_ok$model_id))
      if (!REAL_REFERENCE_MODEL_ID %in% available) {
        stop(
          "The real feats_ok does not contain reference model '",
          REAL_REFERENCE_MODEL_ID, "'. Found: ", paste(available, collapse = ", ")
        )
      }
      real_feats_ok <- real_feats_ok %>%
        filter(.data$model_id == .env$REAL_REFERENCE_MODEL_ID)
    }

    missing_real_features <- setdiff(unique(real_feats_ok$feature), names(real_df_z))
    if (length(missing_real_features) > 0) {
      stop(
        "The real feats_ok references features absent from the real df_z: ",
        paste(head(missing_real_features, 20), collapse = ", "),
        if (length(missing_real_features) > 20) " ..." else ""
      )
    }
    if (nrow(real_df_z) == 0 || nrow(real_feats_ok) == 0 ||
        nrow(real_df_clean) == 0 || nrow(generated_df_clean) == 0) {
      stop("The real/generated reference inputs contain an empty required table.")
    }

    if (outputs_exist(FILE_RF_SHARED_PANEL)) {
      log_existing_outputs("shared feature panel", FILE_RF_SHARED_PANEL)
      shared_panel <- readRDS(FILE_RF_SHARED_PANEL)
    } else {
      shared_panel <- build_shared_feature_panel(
        root = Z_OUTPUT_ROOT,
        requested_model_ids = RF_SHARED_FEATURE_MODEL_IDS
      )
      shared_panel$selected_model_id <- SELECTED_MODEL_ID
      shared_panel$real_reference_model_id <- REAL_REFERENCE_MODEL_ID
      shared_panel$scaling <- paste(
        "Primary paired analysis: centers and scales estimated from authentic",
        "reference data and applied unchanged to both evaluations. Direct",
        "bio-bio RFs: common raw feature values without separate source-wise",
        "standardization."
      )
      saveRDS(shared_panel, FILE_RF_SHARED_PANEL)
    }
    # Canonicalize the panel once. Every cached forest is fitted and predicted
    # with this exact predictor vector.
    shared_features <- sort(unique(as.character(shared_panel$features)))
    if (length(shared_features) < 2) {
      stop(
        "The global shared panel contains fewer than two features across its ",
        "model-language cells."
      )
    }
    missing_real_clean <- setdiff(shared_features, names(real_df_clean))
    missing_generated_clean <- setdiff(shared_features, names(generated_df_clean))
    if (length(missing_real_clean) > 0 || length(missing_generated_clean) > 0) {
      stop(
        "Shared-panel features are missing from df_clean. Real missing: ",
        paste(missing_real_clean, collapse = ", "),
        "; generated missing: ", paste(missing_generated_clean, collapse = ", ")
      )
    }
    message("Primary paired shared feature count: ", length(shared_features))

    real_languages <- sort(unique(as.character(real_feats_ok$language)))
    target_languages <- sort(unique(as.character(feats_ok$language)))
    same_language_panel <- identical(real_languages, target_languages)
    shared_feature_panel_hash <- md5_object(shared_features)

    # Reuse is safe only inside a namespace defined by the exact predictor
    # panel (plus the authentic data, language coverage, settings and seeds).
    # A different panel therefore forces a different namespace and a refit.
    cache_eligible <- run_real_to_generated &&
      RF_AUTHENTIC_CACHE_ENABLED &&
      same_language_panel

    cache_namespace <- md5_object(list(
      real_input = file_signature(REAL_INPUT_DF_CLEAN),
      real_model_id = REAL_REFERENCE_MODEL_ID,
      shared_features = shared_features,
      shared_feature_panel_hash = shared_feature_panel_hash,
      languages = real_languages,
      k_folds = K_FOLDS,
      trees = TREES_FINAL,
      repetitions = RF_REAL_TO_GENERATED_R,
      probability = PROBABILITY_FOREST,
      base_seed = BASE_SEED,
      max_n_per_domain = MAX_N_PER_DOMAIN,
      script_revision = AUTHENTIC_CACHE_REVISION,
      ranger_version = as.character(utils::packageVersion("ranger")),
      r_version = paste(R.version$major, R.version$minor, sep = ".")
    ))
    authentic_cache <- new_authentic_cache(
      enabled = cache_eligible,
      root = RF_AUTHENTIC_CACHE_DIR,
      namespace = cache_namespace,
      wait_seconds = RF_AUTHENTIC_CACHE_WAIT_SECONDS
    )
    stability_cache_root <- file.path(
      RF_AUTHENTIC_CACHE_DIR,
      "balanced_refit_stability"
    )
    stability_cache_namespace <- md5_object(list(
      real_input = file_signature(REAL_INPUT_DF_CLEAN),
      real_model_id = REAL_REFERENCE_MODEL_ID,
      shared_features = shared_features,
      shared_feature_panel_hash = shared_feature_panel_hash,
      languages = real_languages,
      trees = RF_SAMPLING_STABILITY_TREES,
      max_n_per_domain = RF_SAMPLING_STABILITY_MAX_N_PER_DOMAIN,
      train_fraction = RF_SAMPLING_STABILITY_TRAIN_FRACTION,
      min_train_per_domain = MIN_N_PER_DOMAIN,
      probability = PROBABILITY_FOREST,
      split_seed = BASE_SEED + 15000000L,
      fit_seed = BASE_SEED + 16000000L,
      algorithm_revision = "balanced-refit-stability-v2",
      ranger_version = as.character(utils::packageVersion("ranger")),
      r_version = paste(R.version$major, R.version$minor, sep = ".")
    ))
    stability_cache <- new_authentic_cache(
      enabled = run_sampling_stability && RF_AUTHENTIC_CACHE_ENABLED,
      root = stability_cache_root,
      namespace = stability_cache_namespace,
      wait_seconds = RF_AUTHENTIC_CACHE_WAIT_SECONDS
    )

    cache_audit <- tibble(
      selected_model_id = SELECTED_MODEL_ID,
      real_reference_model_id = REAL_REFERENCE_MODEL_ID,
      real_languages = paste(real_languages, collapse = ","),
      target_languages = paste(target_languages, collapse = ","),
      same_language_panel = same_language_panel,
      n_shared_features = length(shared_features),
      shared_feature_panel_hash = shared_feature_panel_hash,
      exact_feature_panel_required = TRUE,
      cache_requested = RF_AUTHENTIC_CACHE_ENABLED,
      cache_enabled = cache_eligible,
      cache_namespace = cache_namespace,
      cache_root = RF_AUTHENTIC_CACHE_DIR,
      stability_cache_enabled = stability_cache$enabled,
      stability_cache_namespace = stability_cache$namespace,
      stability_cache_root = stability_cache$root
    )
    if (run_real_to_generated) {
      save_outputs(cache_audit, FILE_RF_AUTHENTIC_CACHE_AUDIT)
      message(
        "Authentic forest cache ",
        if (cache_eligible) "enabled" else "disabled",
        "; exact shared feature panel hash=", shared_feature_panel_hash,
        "; language coverage ",
        if (same_language_panel) "matches" else "differs"
      )
    }
  }

  run_model_specific <- SELECTED_MODEL_ID == REAL_REFERENCE_MODEL_ID ||
    RF_MODEL_SPECIFIC_ENABLED ||
    RF_DOWNSAMPLE_RUN_THIS_MODEL

  run_manifest <- tibble(
    script_revision = SCRIPT_REVISION,
    model_id = SELECTED_MODEL_ID,
    input_df_z = INPUT_DF_Z,
    input_feats_ok = INPUT_FEATS_OK,
    input_df_clean = INPUT_DF_CLEAN,
    output_dir = OUTPUT_DIR,
    n_rows = nrow(df_z),
    n_languages = n_distinct(df_z$language),
    n_candidate_features = n_distinct(feats_ok$feature),
    k_folds = K_FOLDS,
    trees = TREES_FINAL,
    transfer_repetitions = TRANSFER_R,
    model_specific_enabled = run_model_specific,
    model_specific_explicitly_enabled = RF_MODEL_SPECIFIC_ENABLED,
    downsample_requested = RF_DOWNSAMPLE_ENABLED,
    downsample_run_this_model = RF_DOWNSAMPLE_RUN_THIS_MODEL,
    downsample_model_allowlist = paste(RF_DOWNSAMPLE_MODEL_IDS, collapse = ","),
    downsample_repetitions = RF_DOWNSAMPLE_REPS,
    downsample_target_requested = RF_DOWNSAMPLE_TARGET_N,
    decision_tree_enabled = RF_DECISION_TREE_ENABLED,
    decision_tree_cp = DT_CP,
    decision_tree_maxdepth = DT_MAXDEPTH,
    decision_tree_minsplit = DT_MINSPLIT,
    decision_tree_minbucket = DT_MINBUCKET,
    real_to_generated_enabled = RF_REAL_TO_GENERATED_ENABLED,
    real_to_generated_run = run_real_to_generated,
    biobio_enabled = RF_BIOBIO_ENABLED,
    biobio_run = run_biobio,
    biobio_max_n_per_class = RF_BIOBIO_MAX_N_PER_CLASS,
    biobio_transfer_repetitions = RF_BIOBIO_TRANSFER_R,
    real_reference_model_id = REAL_REFERENCE_MODEL_ID,
    real_reference_input_dir = REAL_INPUT_DIR,
    real_reference_df_clean = REAL_INPUT_DF_CLEAN,
    real_to_generated_repetitions = RF_REAL_TO_GENERATED_R,
    real_to_generated_conf = RF_REAL_TO_GENERATED_CONF,
    sampling_stability_enabled = RF_SAMPLING_STABILITY_ENABLED,
    sampling_stability_run = run_sampling_stability,
    sampling_stability_iterations = RF_SAMPLING_STABILITY_ITERATIONS,
    sampling_stability_trees = RF_SAMPLING_STABILITY_TREES,
    sampling_stability_max_n_per_domain =
      RF_SAMPLING_STABILITY_MAX_N_PER_DOMAIN,
    sampling_stability_train_fraction =
      RF_SAMPLING_STABILITY_TRAIN_FRACTION,
    sampling_stability_min_coverage =
      RF_SAMPLING_STABILITY_MIN_COVERAGE,
    sampling_stability_min_sign_agreement =
      RF_SAMPLING_STABILITY_MIN_SIGN_AGREEMENT,
    sampling_stability_delta_tolerance =
      RF_SAMPLING_STABILITY_DELTA_TOLERANCE,
    sampling_stability_scope = paste(
      "Repeated balanced authentic sampling, new train/held-out split",
      "and smaller authentic RF in each iteration."
    ),
    sampling_stability_cache_enabled = stability_cache$enabled,
    sampling_stability_cache_namespace = if (run_sampling_stability) {
      stability_cache$namespace
    } else {
      NA_character_
    },
    sampling_stability_cache_root = stability_cache$root,
    real_reference_n_rows = if (run_target_comparisons) nrow(real_df_z) else NA_integer_,
    real_reference_n_features = if (run_target_comparisons) {
      n_distinct(real_feats_ok$feature)
    } else {
      NA_integer_
    },
    paired_shared_n_models = if (run_target_comparisons) shared_panel$n_models else NA_integer_,
    paired_shared_n_features = if (run_target_comparisons) length(shared_features) else NA_integer_,
    authentic_cache_requested = RF_AUTHENTIC_CACHE_ENABLED,
    authentic_cache_revision = AUTHENTIC_CACHE_REVISION,
    authentic_cache_enabled = if (run_real_to_generated) {
      authentic_cache$enabled
    } else {
      FALSE
    },
    authentic_cache_exact_feature_panel_required = TRUE,
    authentic_cache_shared_feature_panel_hash =
      shared_feature_panel_hash,
    authentic_cache_same_language_panel = same_language_panel,
    authentic_cache_namespace = if (run_real_to_generated) {
      authentic_cache$namespace
    } else {
      NA_character_
    },
    paired_scaling = if (run_real_to_generated) {
      "real-reference scaling applied to raw df_clean"
    } else {
      NA_character_
    },
    ranger_threads = RANGER_NUM_THREADS,
    started_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
  )
  # Write the active diagnostics and cache namespaces on every invocation so
  # the manifest describes the outputs produced by this run configuration.
  save_outputs(run_manifest, FILE_RF_RUN_MANIFEST)
  
  global_feats_intersection <- prep_global_feats_intersection(feats_ok)
  message("Global feature intersection size: ", length(global_feats_intersection))
  if (length(global_feats_intersection) < 2) {
    stop("Fewer than two features are available in every language; pooled/transfer RF models cannot be fit.")
  }

  balance_plan <- NULL
  if (run_model_specific) {
  if (RF_DOWNSAMPLE_RUN_THIS_MODEL) {
    if (outputs_exist(FILE_RF_BALANCE_AUDIT)) {
      log_existing_outputs("downsampling balance audit", FILE_RF_BALANCE_AUDIT)
      balance_audit <- readRDS(FILE_RF_BALANCE_AUDIT)
      balance_plan <- list(
        target_n = unique(balance_audit$target_n_used)[[1]],
        cv_train_target_n = unique(balance_audit$cv_train_target_n)[[1]],
        audit = balance_audit
      )
    } else {
      usable_counts <- compute_rf_usable_cell_counts(
        df_z = df_z,
        feats_by_lang = feats_ok,
        global_feats = global_feats_intersection
      )
      balance_plan <- resolve_rf_downsample_target(
        usable_counts = usable_counts,
        requested_n = RF_DOWNSAMPLE_TARGET_N,
        k = K_FOLDS,
        min_per_domain = MIN_N_PER_DOMAIN
      )

      if (is.na(RF_DOWNSAMPLE_TARGET_N)) {
        warning(
          "RF downsampling target was inferred from the currently selected model only (n=",
          balance_plan$target_n, "). For strict cross-model comparability, set ",
          "RF_DOWNSAMPLE_TARGET_N to one shared global minimum for every model task."
        )
      }

      save_outputs(balance_plan$audit, FILE_RF_BALANCE_AUDIT)
    }
    message(
      "RF downsampling targets: full training cells n=", balance_plan$target_n,
      "; CV training cells n=", balance_plan$cv_train_target_n
    )
  }
  
  t_step <- Sys.time()
  log_step_start("per-language CV")
  if (outputs_exist(FILE_RF_LANG_CV)) {
    log_existing_outputs("per-language CV", FILE_RF_LANG_CV)
    rf_lang_cv <- readRDS(FILE_RF_LANG_CV)
  } else {
    rf_lang_cv <- rf_per_language_cv_parallel(
      df_z = df_z,
      feats_by_lang = feats_ok,
      k = K_FOLDS,
      trees = TREES_FINAL,
      seed = BASE_SEED,
      min_per_domain = MIN_N_PER_DOMAIN,
      probability = PROBABILITY_FOREST,
      workers = WORKERS,
      num_threads = RANGER_NUM_THREADS
    )
    save_outputs(rf_lang_cv, FILE_RF_LANG_CV)
  }

  if (RF_DOWNSAMPLE_RUN_THIS_MODEL) {
    lang_ds_outputs <- c(FILE_RF_LANG_CV_DS_RAW, FILE_RF_LANG_CV_DS)
    if (outputs_exist(lang_ds_outputs)) {
      log_existing_outputs("downsampled per-language CV", lang_ds_outputs)
    } else {
      rf_lang_cv_ds <- rf_per_language_cv_downsampled(
        df_z = df_z,
        feats_by_lang = feats_ok,
        k = K_FOLDS,
        trees = TREES_FINAL,
        seed = BASE_SEED + 2000000L,
        reps = RF_DOWNSAMPLE_REPS,
        train_target_n = balance_plan$cv_train_target_n,
        conf = RF_DOWNSAMPLE_CONF,
        probability = PROBABILITY_FOREST,
        num_threads = RANGER_NUM_THREADS
      )
      rf_lang_cv_ds$summary <- rf_lang_cv_ds$summary %>%
        left_join(
          rf_lang_cv %>%
            select(language, full_mean_bal_acc = mean_bal_acc,
                   full_sd_bal_acc = sd_bal_acc),
          by = "language"
        ) %>%
        mutate(delta_median_vs_full = bal_acc_median - full_mean_bal_acc)
      save_outputs(rf_lang_cv_ds$raw, FILE_RF_LANG_CV_DS_RAW)
      save_outputs(rf_lang_cv_ds$summary, FILE_RF_LANG_CV_DS)
      rm(rf_lang_cv_ds)
    }
  }

  rm(rf_lang_cv)
  gc(verbose = FALSE)
  log_step_end("per-language CV", t_step)

  t_step <- Sys.time()
  log_step_start("pooled language-agnostic CV")
  if (outputs_exist(FILE_RF_AGNOSTIC)) {
    log_existing_outputs("pooled language-agnostic CV", FILE_RF_AGNOSTIC)
    rf_agnostic <- readRDS(FILE_RF_AGNOSTIC)
  } else {
    rf_agnostic <- rf_language_agnostic_cv(
      df_z = df_z,
      feats = global_feats_intersection,
      k = K_FOLDS,
      trees = TREES_FINAL,
      seed = BASE_SEED,
      min_per_lang_domain = MIN_N_PER_DOMAIN,
      probability = PROBABILITY_FOREST,
      num_threads = RANGER_NUM_THREADS
    )
    save_outputs(rf_agnostic, FILE_RF_AGNOSTIC)
  }

  if (RF_DOWNSAMPLE_RUN_THIS_MODEL) {
    agnostic_ds_outputs <- c(FILE_RF_AGNOSTIC_DS_RAW, FILE_RF_AGNOSTIC_DS)
    if (outputs_exist(agnostic_ds_outputs)) {
      log_existing_outputs("downsampled pooled language-agnostic CV", agnostic_ds_outputs)
    } else {
      rf_agnostic_ds <- rf_language_agnostic_cv_downsampled(
        df_z = df_z,
        feats = global_feats_intersection,
        k = K_FOLDS,
        trees = TREES_FINAL,
        seed = BASE_SEED + 3000000L,
        reps = RF_DOWNSAMPLE_REPS,
        train_target_n = balance_plan$cv_train_target_n,
        conf = RF_DOWNSAMPLE_CONF,
        probability = PROBABILITY_FOREST,
        num_threads = RANGER_NUM_THREADS
      )
      full_pooled_bal_acc <- if (nrow(rf_agnostic) > 0) {
        rf_agnostic$mean_bal_acc[[1]]
      } else {
        NA_real_
      }
      rf_agnostic_ds$summary <- rf_agnostic_ds$summary %>%
        mutate(
          full_pooled_bal_acc = .env$full_pooled_bal_acc,
          delta_median_vs_full = bal_acc_median - full_pooled_bal_acc
        )
      save_outputs(rf_agnostic_ds$raw, FILE_RF_AGNOSTIC_DS_RAW)
      save_outputs(rf_agnostic_ds$summary, FILE_RF_AGNOSTIC_DS)
      rm(rf_agnostic_ds)
    }
  }

  rm(rf_agnostic)
  gc(verbose = FALSE)
  log_step_end("pooled language-agnostic CV", t_step)

  t_step <- Sys.time()
  log_step_start("LOLO")
  if (outputs_exist(FILE_RF_LOLO)) {
    log_existing_outputs("LOLO", FILE_RF_LOLO)
    rf_lolo <- readRDS(FILE_RF_LOLO)
  } else {
    rf_lolo <- rf_language_agnostic_lolo(
      df_z = df_z,
      feats = global_feats_intersection,
      trees = TREES_FINAL,
      seed = BASE_SEED,
      min_per_domain = MIN_N_PER_DOMAIN,
      probability = PROBABILITY_FOREST,
      num_threads = RANGER_NUM_THREADS
    )
    save_outputs(rf_lolo, FILE_RF_LOLO)
  }

  if (RF_DOWNSAMPLE_RUN_THIS_MODEL) {
    lolo_ds_outputs <- c(FILE_RF_LOLO_DS_RAW, FILE_RF_LOLO_DS)
    if (outputs_exist(lolo_ds_outputs)) {
      log_existing_outputs("downsampled LOLO", lolo_ds_outputs)
    } else {
      rf_lolo_ds <- rf_language_agnostic_lolo_downsampled(
        df_z = df_z,
        feats = global_feats_intersection,
        trees = TREES_FINAL,
        seed = BASE_SEED + 4000000L,
        reps = RF_DOWNSAMPLE_REPS,
        target_n = balance_plan$target_n,
        conf = RF_DOWNSAMPLE_CONF,
        probability = PROBABILITY_FOREST,
        num_threads = RANGER_NUM_THREADS
      )
      rf_lolo_ds$summary <- rf_lolo_ds$summary %>%
        left_join(
          rf_lolo %>%
            select(heldout_language, full_bal_acc = bal_acc),
          by = "heldout_language"
        ) %>%
        mutate(delta_median_vs_full = bal_acc_median - full_bal_acc)
      save_outputs(rf_lolo_ds$raw, FILE_RF_LOLO_DS_RAW)
      save_outputs(rf_lolo_ds$summary, FILE_RF_LOLO_DS)
      rm(rf_lolo_ds)
    }
  }

  rm(rf_lolo)
  gc(verbose = FALSE)
  log_step_end("LOLO", t_step)

  t_step <- Sys.time()
  log_step_start("transfer with CI")
  if (outputs_exist(FILE_RF_TRANSFER)) {
    log_existing_outputs("transfer with CI", FILE_RF_TRANSFER)
    rf_transfer_ci <- readRDS(FILE_RF_TRANSFER)
  } else {
    rf_transfer_ci <- rf_transfer_with_ci_lowmem(
      df_z = df_z,
      feats = global_feats_intersection,
      trees = TREES_FINAL,
      seed = BASE_SEED,
      R = TRANSFER_R,
      conf = TRANSFER_CONF,
      min_per_domain = MIN_N_PER_DOMAIN,
      probability = PROBABILITY_FOREST,
      num_threads = RANGER_NUM_THREADS,
      max_n_per_domain = MAX_N_PER_DOMAIN
    )
    save_outputs(rf_transfer_ci, FILE_RF_TRANSFER)
  }

  if (RF_DOWNSAMPLE_RUN_THIS_MODEL) {
    transfer_ds_outputs <- c(FILE_RF_TRANSFER_DS_RAW, FILE_RF_TRANSFER_DS)
    if (outputs_exist(transfer_ds_outputs)) {
      log_existing_outputs("downsampled transfer", transfer_ds_outputs)
    } else {
      rf_transfer_ds <- rf_transfer_training_downsampled(
        df_z = df_z,
        feats = global_feats_intersection,
        trees = TREES_FINAL,
        seed = BASE_SEED + 5000000L,
        reps = RF_DOWNSAMPLE_REPS,
        target_n = balance_plan$target_n,
        conf = RF_DOWNSAMPLE_CONF,
        probability = PROBABILITY_FOREST,
        num_threads = RANGER_NUM_THREADS
      )
      rf_transfer_ds$summary <- rf_transfer_ds$summary %>%
        left_join(
          rf_transfer_ci %>%
            select(src_language, tgt_language,
                   full_mean_bal_acc = mean_bal_acc),
          by = c("src_language", "tgt_language")
        ) %>%
        mutate(delta_median_vs_full = bal_acc_median - full_mean_bal_acc)
      save_outputs(rf_transfer_ds$raw, FILE_RF_TRANSFER_DS_RAW)
      save_outputs(rf_transfer_ds$summary, FILE_RF_TRANSFER_DS)
      rm(rf_transfer_ds)
    }
  }

  rm(rf_transfer_ci)
  gc(verbose = FALSE)
  log_step_end("transfer with CI", t_step)
  } else {
    message(
      "Skipping synthetic-trained model-specific within-language, pooled, ",
      "LOLO and transfer analyses. The four paired authentic-to-generated ",
      "analyses remain enabled."
    )
  }

  if (run_real_to_generated) {
    t_step <- Sys.time()
    log_step_start("primary paired real-vs-generated RF analysis")

    paired_lang_outputs <- c(FILE_RF_REAL_GEN_LANG_RAW, FILE_RF_REAL_GEN_LANG)
    if (outputs_exist(paired_lang_outputs)) {
      log_existing_outputs("paired real-vs-generated per-language CV", paired_lang_outputs)
    } else {
      rf_real_gen_lang <- rf_paired_per_language_cv(
        real_df = real_df_clean,
        generated_df = generated_df_clean,
        feats = shared_features,
        k = K_FOLDS,
        trees = TREES_FINAL,
        seed = BASE_SEED + 6000000L,
        R = RF_REAL_TO_GENERATED_R,
        conf = RF_REAL_TO_GENERATED_CONF,
        min_per_domain = MIN_N_PER_DOMAIN,
        probability = PROBABILITY_FOREST,
        num_threads = RANGER_NUM_THREADS,
        cache = authentic_cache
      )
      save_outputs(rf_real_gen_lang$raw, FILE_RF_REAL_GEN_LANG_RAW)
      save_outputs(rf_real_gen_lang$summary, FILE_RF_REAL_GEN_LANG)
      rm(rf_real_gen_lang)
    }

    paired_agnostic_outputs <- c(
      FILE_RF_REAL_GEN_AGNOSTIC_RAW, FILE_RF_REAL_GEN_AGNOSTIC
    )
    if (outputs_exist(paired_agnostic_outputs)) {
      log_existing_outputs("paired real-vs-generated pooled CV", paired_agnostic_outputs)
    } else {
      rf_real_gen_agnostic <- rf_paired_pooled_cv(
        real_df = real_df_clean,
        generated_df = generated_df_clean,
        feats = shared_features,
        k = K_FOLDS,
        trees = TREES_FINAL,
        seed = BASE_SEED + 7000000L,
        R = RF_REAL_TO_GENERATED_R,
        conf = RF_REAL_TO_GENERATED_CONF,
        min_per_lang_domain = MIN_N_PER_DOMAIN,
        probability = PROBABILITY_FOREST,
        num_threads = RANGER_NUM_THREADS,
        cache = authentic_cache
      )
      save_outputs(rf_real_gen_agnostic$raw, FILE_RF_REAL_GEN_AGNOSTIC_RAW)
      save_outputs(rf_real_gen_agnostic$summary, FILE_RF_REAL_GEN_AGNOSTIC)
      rm(rf_real_gen_agnostic)
    }

    paired_lolo_outputs <- c(FILE_RF_REAL_GEN_LOLO_RAW, FILE_RF_REAL_GEN_LOLO)
    if (outputs_exist(paired_lolo_outputs)) {
      log_existing_outputs("paired real-vs-generated LOLO", paired_lolo_outputs)
    } else {
      rf_real_gen_lolo <- rf_paired_lolo(
        real_df = real_df_clean,
        generated_df = generated_df_clean,
        feats = shared_features,
        trees = TREES_FINAL,
        seed = BASE_SEED + 8000000L,
        R = RF_REAL_TO_GENERATED_R,
        conf = RF_REAL_TO_GENERATED_CONF,
        min_per_domain = MIN_N_PER_DOMAIN,
        probability = PROBABILITY_FOREST,
        num_threads = RANGER_NUM_THREADS,
        cache = authentic_cache
      )
      save_outputs(rf_real_gen_lolo$raw, FILE_RF_REAL_GEN_LOLO_RAW)
      save_outputs(rf_real_gen_lolo$summary, FILE_RF_REAL_GEN_LOLO)
      rm(rf_real_gen_lolo)
    }

    paired_transfer_outputs <- c(
      FILE_RF_REAL_GEN_TRANSFER_RAW, FILE_RF_REAL_GEN_TRANSFER
    )
    if (outputs_exist(paired_transfer_outputs)) {
      log_existing_outputs("paired real-vs-generated transfer", paired_transfer_outputs)
    } else {
      rf_real_gen_transfer <- rf_paired_transfer(
        real_df = real_df_clean,
        generated_df = generated_df_clean,
        feats = shared_features,
        trees = TREES_FINAL,
        seed = BASE_SEED + 9000000L,
        R = RF_REAL_TO_GENERATED_R,
        conf = RF_REAL_TO_GENERATED_CONF,
        min_per_domain = MIN_N_PER_DOMAIN,
        probability = PROBABILITY_FOREST,
        num_threads = RANGER_NUM_THREADS,
        max_n_per_domain = MAX_N_PER_DOMAIN,
        cache = authentic_cache
      )
      save_outputs(rf_real_gen_transfer$raw, FILE_RF_REAL_GEN_TRANSFER_RAW)
      save_outputs(rf_real_gen_transfer$summary, FILE_RF_REAL_GEN_TRANSFER)
      rm(rf_real_gen_transfer)
    }

    gc(verbose = FALSE)
    log_step_end("primary paired real-vs-generated RF analysis", t_step)
  }

  if (run_sampling_stability) {
    stability_outputs <- c(
      FILE_RF_SAMPLING_STABILITY_RAW,
      FILE_RF_SAMPLING_STABILITY
    )
    t_step <- Sys.time()
    log_step_start(
      "paired per-language balanced-refit sampling stability"
    )
    if (outputs_exist(stability_outputs)) {
      log_existing_outputs(
        "paired per-language balanced-refit sampling stability",
        stability_outputs
      )
    } else {
      if (!isTRUE(stability_cache$enabled)) {
        stop(
          "Balanced-refit stability requires its shared authentic-forest ",
          "cache. Enable RF_AUTHENTIC_CACHE_ENABLED or disable ",
          "RF_SAMPLING_STABILITY_ENABLED."
        )
      }
      if (!file.exists(FILE_RF_REAL_GEN_LANG)) {
        stop(
          "Balanced-refit stability requires the primary per-language ",
          "paired summary: ", FILE_RF_REAL_GEN_LANG
        )
      }
      full_paired_summary <- readRDS(FILE_RF_REAL_GEN_LANG)
      stability <- rf_paired_balanced_refit_stability(
        real_df = real_df_clean,
        generated_df = generated_df_clean,
        feats = shared_features,
        full_summary = full_paired_summary,
        trees = RF_SAMPLING_STABILITY_TREES,
        iterations = RF_SAMPLING_STABILITY_ITERATIONS,
        train_fraction = RF_SAMPLING_STABILITY_TRAIN_FRACTION,
        max_n_per_domain = RF_SAMPLING_STABILITY_MAX_N_PER_DOMAIN,
        confidence = RF_REAL_TO_GENERATED_CONF,
        min_per_domain = MIN_N_PER_DOMAIN,
        min_coverage = RF_SAMPLING_STABILITY_MIN_COVERAGE,
        min_sign_agreement =
          RF_SAMPLING_STABILITY_MIN_SIGN_AGREEMENT,
        delta_tolerance =
          RF_SAMPLING_STABILITY_DELTA_TOLERANCE,
        split_seed = BASE_SEED + 15000000L,
        fit_seed = BASE_SEED + 16000000L,
        probability = PROBABILITY_FOREST,
        num_threads = RANGER_NUM_THREADS,
        cache = stability_cache
      )
      save_outputs(
        stability$raw,
        FILE_RF_SAMPLING_STABILITY_RAW
      )
      save_outputs(
        stability$summary,
        FILE_RF_SAMPLING_STABILITY
      )
      rm(stability, full_paired_summary)
    }
    gc(verbose = FALSE)
    log_step_end(
      "paired per-language balanced-refit sampling stability",
      t_step
    )
  }

  if (run_biobio) {
    t_step <- Sys.time()
    log_step_start("direct real-biomedical vs target-biomedical RF analysis")

    biobio <- prepare_biobio_balanced(
      real_df = real_df_clean,
      target_df = generated_df_clean,
      feats = shared_features,
      max_n_per_class = RF_BIOBIO_MAX_N_PER_CLASS,
      min_per_class = MIN_N_PER_DOMAIN,
      seed = BASE_SEED + 10000000L
    )
    biobio_data <- biobio$data
    biobio_feats_by_lang <- tidyr::crossing(
      language = sort(unique(as.character(biobio_data$language))),
      feature = shared_features
    )

    if (outputs_exist(FILE_RF_BIOBIO_BALANCE_AUDIT)) {
      log_existing_outputs(
        "bio-bio balance audit", FILE_RF_BIOBIO_BALANCE_AUDIT
      )
    } else {
      save_outputs(biobio$audit, FILE_RF_BIOBIO_BALANCE_AUDIT)
    }
    message(
      "Bio-bio balanced languages: ",
      n_distinct(biobio_data$language),
      "; observations: ", nrow(biobio_data),
      "; maximum per language/class: ", RF_BIOBIO_MAX_N_PER_CLASS
    )

    if (outputs_exist(FILE_RF_BIOBIO_LANG)) {
      log_existing_outputs("bio-bio per-language CV", FILE_RF_BIOBIO_LANG)
    } else {
      rf_biobio_lang <- rf_per_language_cv_parallel(
        df_z = biobio_data,
        feats_by_lang = biobio_feats_by_lang,
        k = K_FOLDS,
        trees = TREES_FINAL,
        seed = BASE_SEED + 11000000L,
        min_per_domain = MIN_N_PER_DOMAIN,
        probability = PROBABILITY_FOREST,
        workers = WORKERS,
        num_threads = RANGER_NUM_THREADS
      ) %>%
        add_biobio_metadata("per_language_stratified_kfold")
      save_outputs(rf_biobio_lang, FILE_RF_BIOBIO_LANG)
      rm(rf_biobio_lang)
    }

    if (outputs_exist(FILE_RF_BIOBIO_AGNOSTIC)) {
      log_existing_outputs(
        "bio-bio pooled language-agnostic CV", FILE_RF_BIOBIO_AGNOSTIC
      )
    } else {
      rf_biobio_agnostic <- rf_language_agnostic_cv(
        df_z = biobio_data,
        feats = shared_features,
        k = K_FOLDS,
        trees = TREES_FINAL,
        seed = BASE_SEED + 12000000L,
        min_per_lang_domain = MIN_N_PER_DOMAIN,
        probability = PROBABILITY_FOREST,
        num_threads = RANGER_NUM_THREADS
      ) %>%
        add_biobio_metadata("pooled_language_agnostic_stratified_kfold")
      save_outputs(rf_biobio_agnostic, FILE_RF_BIOBIO_AGNOSTIC)
      rm(rf_biobio_agnostic)
    }

    if (outputs_exist(FILE_RF_BIOBIO_LOLO)) {
      log_existing_outputs("bio-bio LOLO", FILE_RF_BIOBIO_LOLO)
    } else {
      rf_biobio_lolo <- rf_language_agnostic_lolo(
        df_z = biobio_data,
        feats = shared_features,
        trees = TREES_FINAL,
        seed = BASE_SEED + 13000000L,
        min_per_domain = MIN_N_PER_DOMAIN,
        probability = PROBABILITY_FOREST,
        num_threads = RANGER_NUM_THREADS
      ) %>%
        add_biobio_metadata("leave_one_language_out")
      save_outputs(rf_biobio_lolo, FILE_RF_BIOBIO_LOLO)
      rm(rf_biobio_lolo)
    }

    if (outputs_exist(FILE_RF_BIOBIO_TRANSFER)) {
      log_existing_outputs(
        "bio-bio directed cross-language transfer",
        FILE_RF_BIOBIO_TRANSFER
      )
    } else {
      rf_biobio_transfer <- rf_transfer_with_ci_lowmem(
        df_z = biobio_data,
        feats = shared_features,
        trees = TREES_FINAL,
        seed = BASE_SEED + 14000000L,
        R = RF_BIOBIO_TRANSFER_R,
        conf = TRANSFER_CONF,
        min_per_domain = MIN_N_PER_DOMAIN,
        probability = PROBABILITY_FOREST,
        num_threads = RANGER_NUM_THREADS,
        max_n_per_domain = NULL
      ) %>%
        add_biobio_metadata("directed_cross_language_transfer")
      save_outputs(rf_biobio_transfer, FILE_RF_BIOBIO_TRANSFER)
      rm(rf_biobio_transfer)
    }

    rm(biobio, biobio_data, biobio_feats_by_lang)
    gc(verbose = FALSE)
    log_step_end(
      "direct real-biomedical vs target-biomedical RF analysis", t_step
    )
  }

  if (RF_DECISION_TREE_ENABLED && run_model_specific) {
    t_step <- Sys.time()
    log_step_start("diagnostic decision trees")

    if (outputs_exist(FILE_DT_LANG_CV)) {
      log_existing_outputs("per-language diagnostic trees", FILE_DT_LANG_CV)
    } else {
      dt_lang_cv <- dt_per_language_cv(
        df_z = df_z,
        feats_by_lang = feats_ok,
        setting = "per_language_stratified_kfold",
        k = K_FOLDS,
        seed = BASE_SEED,
        min_per_domain = MIN_N_PER_DOMAIN
      )
      save_dt_bundle(dt_lang_cv, FILE_DT_LANG_CV)
      rm(dt_lang_cv)
    }

    if (outputs_exist(FILE_DT_AGNOSTIC)) {
      log_existing_outputs("pooled diagnostic trees", FILE_DT_AGNOSTIC)
    } else {
      dt_agnostic <- dt_pooled_cv(
        df_z = df_z,
        feats = global_feats_intersection,
        setting = "pooled_language_agnostic_stratified_kfold",
        k = K_FOLDS,
        seed = BASE_SEED,
        min_per_lang_domain = MIN_N_PER_DOMAIN
      )
      save_dt_bundle(dt_agnostic, FILE_DT_AGNOSTIC)
      rm(dt_agnostic)
    }

    if (outputs_exist(FILE_DT_LOLO)) {
      log_existing_outputs("LOLO diagnostic trees", FILE_DT_LOLO)
    } else {
      dt_lolo <- dt_lolo_models(
        df_z = df_z,
        feats = global_feats_intersection,
        setting = "leave_one_language_out",
        seed = BASE_SEED,
        min_per_domain = MIN_N_PER_DOMAIN
      )
      save_dt_bundle(dt_lolo, FILE_DT_LOLO)
      rm(dt_lolo)
    }

    if (outputs_exist(FILE_DT_TRANSFER)) {
      log_existing_outputs("transfer diagnostic trees", FILE_DT_TRANSFER)
    } else {
      dt_transfer <- dt_transfer_models(
        df_z = df_z,
        feats = global_feats_intersection,
        setting = "cross_language_transfer",
        seed = BASE_SEED,
        min_per_domain = MIN_N_PER_DOMAIN,
        max_n_per_domain = MAX_N_PER_DOMAIN,
        rf_repetitions_represented = TRANSFER_R
      )
      save_dt_bundle(dt_transfer, FILE_DT_TRANSFER)
      rm(dt_transfer)
    }

    if (RF_DOWNSAMPLE_RUN_THIS_MODEL) {
      if (outputs_exist(FILE_DT_LANG_CV_DS)) {
        log_existing_outputs("downsampled per-language diagnostic trees", FILE_DT_LANG_CV_DS)
      } else {
        dt_lang_cv_ds <- dt_per_language_cv(
          df_z = df_z,
          feats_by_lang = feats_ok,
          setting = "per_language_training_downsampled_kfold",
          k = K_FOLDS,
          seed = BASE_SEED + 2000000L,
          min_per_domain = MIN_N_PER_DOMAIN,
          reps = RF_DOWNSAMPLE_REPS,
          train_target_n = balance_plan$cv_train_target_n
        )
        save_dt_bundle(dt_lang_cv_ds, FILE_DT_LANG_CV_DS)
        rm(dt_lang_cv_ds)
      }

      if (outputs_exist(FILE_DT_AGNOSTIC_DS)) {
        log_existing_outputs("downsampled pooled diagnostic trees", FILE_DT_AGNOSTIC_DS)
      } else {
        dt_agnostic_ds <- dt_pooled_cv(
          df_z = df_z,
          feats = global_feats_intersection,
          setting = "pooled_language_agnostic_training_downsampled_kfold",
          k = K_FOLDS,
          seed = BASE_SEED + 3000000L,
          min_per_lang_domain = MIN_N_PER_DOMAIN,
          reps = RF_DOWNSAMPLE_REPS,
          train_target_n = balance_plan$cv_train_target_n
        )
        save_dt_bundle(dt_agnostic_ds, FILE_DT_AGNOSTIC_DS)
        rm(dt_agnostic_ds)
      }

      if (outputs_exist(FILE_DT_LOLO_DS)) {
        log_existing_outputs("downsampled LOLO diagnostic trees", FILE_DT_LOLO_DS)
      } else {
        dt_lolo_ds <- dt_lolo_models(
          df_z = df_z,
          feats = global_feats_intersection,
          setting = "leave_one_language_out_training_downsampled",
          seed = BASE_SEED + 4000000L,
          min_per_domain = MIN_N_PER_DOMAIN,
          reps = RF_DOWNSAMPLE_REPS,
          target_n = balance_plan$target_n
        )
        save_dt_bundle(dt_lolo_ds, FILE_DT_LOLO_DS)
        rm(dt_lolo_ds)
      }

      if (outputs_exist(FILE_DT_TRANSFER_DS)) {
        log_existing_outputs("downsampled transfer diagnostic trees", FILE_DT_TRANSFER_DS)
      } else {
        dt_transfer_ds <- dt_transfer_models(
          df_z = df_z,
          feats = global_feats_intersection,
          setting = "cross_language_transfer_training_downsampled",
          seed = BASE_SEED + 5000000L,
          min_per_domain = MIN_N_PER_DOMAIN,
          reps = RF_DOWNSAMPLE_REPS,
          target_n = balance_plan$target_n
        )
        save_dt_bundle(dt_transfer_ds, FILE_DT_TRANSFER_DS)
        rm(dt_transfer_ds)
      }
    }

    gc(verbose = FALSE)
    log_step_end("diagnostic decision trees", t_step)
  }
  
  # t_step <- Sys.time()
  # log_step_start("per-language feature relevance")
  # rf_lang_feat <- rf_per_language_feature_relevance_parallel(
  #   df_z = df_z,
  #   feats_by_lang = feats_ok,
  #   trees = TREES_FINAL,
  #   seed = BASE_SEED,
  #   B = RF_RELEVANCE_B,
  #   min_per_domain = MIN_N_PER_DOMAIN,
  #   alpha = RF_RELEVANCE_ALPHA,
  #   adjust_method = RF_RELEVANCE_ADJUST,
  #   importance_mode = RF_IMPORTANCE_MODE,
  #   probability = PROBABILITY_FOREST,
  #   workers = WORKERS,
  #   num_threads = RANGER_NUM_THREADS
  # )
  # save_outputs(rf_lang_feat, FILE_RF_LANG_FEAT)
  # log_step_end("per-language feature relevance", t_step)
  # 
  # t_step <- Sys.time()
  # log_step_start("feature clustering")
  # rf_lang_clusters <- cluster_rf_features_per_language(
  #   df_z = df_z,
  #   rf_feature_tbl = rf_lang_feat$tested,
  #   q_max = CLUSTER_Q_MAX,
  #   cutoff_corr = CLUSTER_CUTOFF_CORR,
  #   method = COR_METHOD,
  #   center_within_domain = USE_DOMAIN_CENTERING,
  #   cap_per_domain = MAX_N_PER_DOMAIN,
  #   seed = BASE_SEED
  # )
  # save_outputs(rf_lang_clusters, FILE_RF_LANG_CLUSTERS)
  # log_step_end("feature clustering", t_step)
  # 
  # t_step <- Sys.time()
  # log_step_start("meta-clusters")
  # rf_meta_clusters <- build_rf_meta_clusters(rf_lang_clusters = rf_lang_clusters, jaccs = META_JACCS)
  # save_outputs(rf_meta_clusters, FILE_RF_META)
  # log_step_end("meta-clusters", t_step)
  # 
  # t_step <- Sys.time()
  # log_step_start("meta-cluster summary")
  # rf_meta_summary <- summarize_rf_meta_clusters(rf_meta_clusters = rf_meta_clusters, alpha = RF_RELEVANCE_ALPHA)
  # save_outputs(rf_meta_summary, FILE_RF_META_SUMMARY)
  # log_step_end("meta-cluster summary", t_step)
  
  message("Done.")
}

print_usage <- function() {
  cat(
    "Usage: ALIGNMENT_MODEL_ID=<model> Rscript src/alignment/multivariate.R\n",
    "       Rscript src/alignment/multivariate.R --list-models\n",
    "\nThe script consumes model-specific checkpoints from common.R.\n",
    "See src/alignment/README.md for configuration.\n",
    sep = ""
  )
}

if (sys.nframe() == 0L) {
  cli_args <- commandArgs(trailingOnly = TRUE)
  if (any(cli_args %in% c("--help", "-h"))) {
    print_usage()
  } else if ("--list-models" %in% cli_args) {
    models <- discover_completed_models(Z_OUTPUT_ROOT)
    cat(paste(models, collapse = "\n"), if (length(models) > 0) "\n" else "", sep = "")
  } else {
    main()
  }
}
