#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
  library(tibble)
  library(readr)
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

PROJECT_ROOT <- ALIGNMENT_PROJECT_ROOT
FEATURE_ROOT <- alignment_env_value(
  "ALIGNMENT_FEATURE_ROOT",
  file.path(PROJECT_ROOT, "data", "features")
)

GENERAL_INPUT_PATTERN <- "^(?:completions|features)_.*_general(?:_[A-Za-z]{2,3})?\\.csv$"
BIOMEDICAL_INPUT_PATTERN <- "^(?:completions|features)_.+_biomedical(?:_[A-Za-z]{2,3})?\\.csv$"

# Real reference feature files. Domain-specific filename patterns determine
# which files are read if both domains are stored in the same directory.
REAL_GENERAL_DIR <- alignment_env_value(
  "ALIGNMENT_REAL_GENERAL_DIR",
  file.path(FEATURE_ROOT, "real")
)
REAL_BIOMEDICAL_DIR <- alignment_env_value(
  "ALIGNMENT_REAL_BIOMEDICAL_DIR",
  file.path(FEATURE_ROOT, "real")
)

# Root that contains either the feature-family directories directly or an
# outputs/ child containing them. All matching biomedical CSVs are discovered
# recursively. vLLM files can contain any model name.
BIOMEDICAL_OUTPUTS_DIR <- alignment_env_value(
  "ALIGNMENT_BIOMEDICAL_FEATURE_ROOT",
  FEATURE_ROOT
)
BIOMEDICAL_FAMILIES <- alignment_env_csv(
  "ALIGNMENT_BIOMEDICAL_FAMILIES",
  c(
    "biogpt_features",
    "completions20260216_bio",
    "unigram_features",
    "upper_bound_features",
    "vllm_features"
  )
)

# Restrict vLLM discovery to one run date. Set to NULL to include every run.
VLLM_RUN_DATE_RAW <- alignment_env_value("ALIGNMENT_VLLM_RUN_DATE", "20260708")
VLLM_RUN_DATE <- if (str_to_lower(VLLM_RUN_DATE_RAW) %in% c("all", "any", "none")) {
  NULL
} else {
  VLLM_RUN_DATE_RAW
}

# Additional directories containing files named
# features_<model>_biomedical[_<language>].csv.
ADDITIONAL_BIOMEDICAL_DIRS <- alignment_env_csv(
  "ALIGNMENT_ADDITIONAL_BIOMEDICAL_DIRS",
  character()
)

# A selected model is appended to the output path, allowing one model per
# scheduler task while keeping the combined-run behavior as the default.
OUTPUT_ROOT <- alignment_env_value(
  "ALIGNMENT_UNIVARIATE_OUTPUT_DIR",
  file.path(PROJECT_ROOT, "outputs", "alignment", "univariate")
)
SELECTED_MODEL_ID <- alignment_env_value("ALIGNMENT_MODEL_ID", "", "DOMAIN_SHIFT_MODEL_ID")
alignment_validate_model_id(SELECTED_MODEL_ID)
OUTPUT_DIR <- if (nzchar(SELECTED_MODEL_ID)) {
  file.path(OUTPUT_ROOT, SELECTED_MODEL_ID)
} else {
  OUTPUT_ROOT
}

# Reuse the z-scoring and permutation checkpoints when all required files exist.
RESUME_EXISTING <- alignment_env_flag("ALIGNMENT_RESUME", TRUE, "DOMAIN_SHIFT_RESUME")
PREPARE_ONLY <- alignment_env_flag("ALIGNMENT_PREPARE_ONLY", FALSE)

# Run the full analysis a second time with the dedicated real-biomedical
# directory as the reference and every discovered biomedical input as a target.
RUN_REAL_BIOMEDICAL_CONTRAST <- alignment_env_flag(
  "ALIGNMENT_REAL_BIOMEDICAL_CONTRAST",
  TRUE,
  "DOMAIN_SHIFT_REAL_BIOMEDICAL_CONTRAST"
)
REAL_BIOMEDICAL_SUBDIR <- "real_biomedical_vs_biomedical"

ISO3_TO_ISO2 <- c(
  deu = "de", eng = "en", fra = "fr", ita = "it", nld = "nl",
  pol = "pl", ron = "ro", spa = "es", swe = "sv"
)

ID_COLS <- c(
  "model", "model_id", "model_label", "country", "source_file", "doc_index", "doc_id",
  "domain", "language", "area", "subspecialty", "modality", "contributor_code"
)

MIN_N_PER_DOMAIN     <- 5
EPS_SD               <- 1e-12

# correlation pruning
CUTOFF_CORR          <- 0.85
COR_METHOD           <- "spearman"
USE_DOMAIN_CENTERING <- TRUE
MAX_N_PER_DOMAIN     <- 3000

# JSD config
JSD_GRID_N           <- 512
JSD_BW               <- "nrd0"
JSD_EPS              <- 1e-12

# per-language scoring
SCORE_METRICS        <- c("cohens_d", "w1_z", "jsd")
SCORE_B              <- 500

# post-hoc clustering
CLUSTER_Q_MAX        <- 0.05
CLUSTER_TOP_N        <- NULL
CLUSTER_CUTOFF_CORR  <- 0.8

# fingerprints
FINGERPRINT_K        <- 20

# meta-clusters
META_JACCS           <- c(0.2, 0.3, 0.4, 0.6, 0.8)
META_ALPHA           <- 0.05
PAIRCOUNT_JACC       <- 0.8

BASE_SEED            <- 123
set.seed(BASE_SEED)

# repeated equal-n sensitivity analysis
#
# The primary analysis above continues to use every valid row. This additional
# analysis repeatedly downsamples every model/language/domain cell to the same
# number of rows and measures how stable the observed metrics and rankings are.
BALANCED_STABILITY_ENABLED <- alignment_env_flag(
  "ALIGNMENT_BALANCED_STABILITY",
  TRUE,
  "DOMAIN_SHIFT_BALANCED_STABILITY"
)
BALANCED_STABILITY_REPS <- suppressWarnings(as.integer(
  alignment_env_value("ALIGNMENT_BALANCED_REPS", "100", "DOMAIN_SHIFT_BALANCED_REPS")
))
BALANCED_STABILITY_TARGET_RAW <- trimws(
  alignment_env_value("ALIGNMENT_BALANCED_TARGET_N", "", "DOMAIN_SHIFT_BALANCED_TARGET_N")
)
BALANCED_STABILITY_TARGET_N <- if (nzchar(BALANCED_STABILITY_TARGET_RAW)) {
  suppressWarnings(as.integer(BALANCED_STABILITY_TARGET_RAW))
} else {
  NA_integer_
}
BALANCED_STABILITY_CONF_LEVEL <- 0.95
BALANCED_STABILITY_TOP_K <- FINGERPRINT_K

if (BALANCED_STABILITY_ENABLED &&
    (!is.finite(BALANCED_STABILITY_REPS) || BALANCED_STABILITY_REPS < 2)) {
  stop("ALIGNMENT_BALANCED_REPS must be an integer >= 2")
}
if (nzchar(BALANCED_STABILITY_TARGET_RAW) && is.na(BALANCED_STABILITY_TARGET_N)) {
  stop("ALIGNMENT_BALANCED_TARGET_N must be empty or a valid integer")
}
if (!is.na(BALANCED_STABILITY_TARGET_N) && BALANCED_STABILITY_TARGET_N < MIN_N_PER_DOMAIN) {
  stop("ALIGNMENT_BALANCED_TARGET_N must be >= MIN_N_PER_DOMAIN")
}

# =========================================================
# IO / LOGGING
# =========================================================

save_rds_obj <- function(x, filename, output_dir = OUTPUT_DIR) {
  saveRDS(x, file.path(output_dir, filename))
}

log_msg <- function(...) {
  message(sprintf("[%s] %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), paste0(..., collapse = "")))
  flush.console()
}

validate_prepared_alignment <- function(df_z, feats_ok,
                                        selected_model_id = SELECTED_MODEL_ID) {
  alignment_require_columns(
    df_z,
    c("model_id", "model_label", "language", "domain"),
    "df_z.rds"
  )
  alignment_require_columns(
    feats_ok,
    c("model_id", "language", "feature"),
    "feats_ok.rds"
  )

  if (nzchar(selected_model_id)) {
    checkpoint_models <- unique(c(
      as.character(df_z$model_id),
      as.character(feats_ok$model_id)
    ))
    checkpoint_models <- sort(checkpoint_models[!is.na(checkpoint_models)])
    if (!identical(checkpoint_models, selected_model_id)) {
      stop(
        "Shared checkpoints do not belong exclusively to selected model '",
        selected_model_id,
        "'. Found: ",
        paste(checkpoint_models, collapse = ", ")
      )
    }
  }
  invisible(NULL)
}

# =========================================================
# HELPERS
# =========================================================

normalize_language_code <- function(x, iso3_to_iso2 = ISO3_TO_ISO2) {
  x <- str_to_lower(as.character(x))
  case_when(
    str_detect(x, "^[a-z]{2}$") ~ x,
    str_detect(x, "^[a-z]{3}$") & x %in% names(iso3_to_iso2) ~ unname(iso3_to_iso2[x]),
    TRUE ~ NA_character_
  )
}

extract_biomedical_language <- function(path,
                                         iso3_to_iso2 = ISO3_TO_ISO2,
                                         required = TRUE) {
  path <- str_replace_all(path, "\\\\", "/")

  # BioGPT, unigram and vLLM paths end their generation-directory name with
  # .<language>.biomedical[.json]. Upper-bound paths use biomedical_<language>.json.
  raw_lang <- str_match(
    path,
    "(?:^|[./_])([A-Za-z]{2,3})\\.biomedical(?:\\.json)?(?:/|$)"
  )[1, 2]

  if (is.na(raw_lang)) {
    raw_lang <- str_match(
      path,
      "(?:^|/)biomedical_([A-Za-z]{2,3})(?:\\.json)?(?:/|$)"
    )[1, 2]
  }

  if (is.na(raw_lang)) {
    raw_lang <- str_match(
      basename(path),
      "_biomedical_([A-Za-z]{2,3})\\.csv$"
    )[1, 2]
  }

  lang <- normalize_language_code(raw_lang, iso3_to_iso2)
  if (is.na(lang) && required) {
    stop("Could not derive a supported language from path: ", path)
  }
  lang
}

resolve_family_dir <- function(root, family) {
  candidates <- c(file.path(root, family), file.path(root, "outputs", family))
  existing <- candidates[dir.exists(candidates)]
  if (length(existing) == 0) return(NA_character_)
  existing[[1]]
}

discover_biomedical_manifest <- function(root,
                                         families = BIOMEDICAL_FAMILIES,
                                         additional_dirs = ADDITIONAL_BIOMEDICAL_DIRS,
                                         pattern = BIOMEDICAL_INPUT_PATTERN,
                                         vllm_run_date = VLLM_RUN_DATE,
                                         iso3_to_iso2 = ISO3_TO_ISO2) {
  family_dirs <- tibble(
    feature_family = families,
    family_dir = map_chr(families, ~ resolve_family_dir(root, .x))
  )

  if (length(additional_dirs) > 0) {
    family_dirs <- bind_rows(
      family_dirs,
      tibble(
        feature_family = "additional_biomedical_features",
        family_dir = additional_dirs
      )
    )
  }

  missing_families <- family_dirs %>% filter(is.na(family_dir)) %>% pull(feature_family)
  if (length(missing_families) > 0) {
    warning("Feature families not found and skipped: ", paste(missing_families, collapse = ", "))
  }

  # The unified extractor may write date/model directories directly below the
  # feature root. Use that root when no named family directory is present.
  if (!any(!is.na(family_dirs$family_dir)) && dir.exists(root)) {
    family_dirs <- tibble(
      feature_family = "biomedical_features",
      family_dir = root
    )
  }

  manifest <- family_dirs %>%
    filter(!is.na(family_dir)) %>%
    mutate(
      files = map(family_dir, ~ list.files(
        .x, pattern = pattern, recursive = TRUE, full.names = TRUE
      ))
    ) %>%
    select(feature_family, files) %>%
    unnest_longer(files, values_to = "file") %>%
    mutate(file = as.character(file))

  if (!is.null(vllm_run_date)) {
    date_path <- paste0("/", vllm_run_date, "/")
    manifest <- manifest %>%
      filter(feature_family != "vllm_features" | str_detect(file, fixed(date_path)))
  }

  if (nrow(manifest) == 0) {
    stop("No biomedical feature files were discovered below: ", root)
  }

  manifest <- manifest %>%
    mutate(
      file_name = basename(file),
      file_stub = tools::file_path_sans_ext(file_name),
      model_id_raw = coalesce(
        str_match(
          file_name,
          "^(?:completions|features)_(.+)_biomedical\\.csv$"
        )[, 2],
        str_match(
          file_name,
          "^(?:completions|features)_(.+)_biomedical_[A-Za-z]{2,3}\\.csv$"
        )[, 2]
      ),
      generation_input = case_when(
        feature_family == "completions20260216_bio" ~ "completion",
        feature_family == "vllm_features" ~ "prompt",
        TRUE ~ NA_character_
      ),
      model_id = case_when(
        generation_input == "completion" & !str_ends(model_id_raw, "_completion") ~
          paste0(model_id_raw, "_completion"),
        generation_input == "prompt" & !str_ends(model_id_raw, "_prompt") ~
          paste0(model_id_raw, "_prompt"),
        TRUE ~ model_id_raw
      ),
      model_label = case_when(
        model_id_raw == "biounigram" ~ "unigram",
        TRUE ~ str_replace_all(model_id, "[-_]+", " ")
      ),
      language_from_file = map_chr(
        file,
        ~ extract_biomedical_language(.x, iso3_to_iso2, required = FALSE)
      ),
      domain = "biomedical",
      source_origin = case_when(
        feature_family == "upper_bound_features" ~ "real",
        feature_family == "unigram_features" ~ "unigram",
        TRUE ~ "model"
      ),
      contrast_group = paste0("biomedical__", model_id)
    ) %>%
    arrange(model_id, language_from_file, file)

  unparsed <- manifest %>% filter(is.na(model_id_raw) | !nzchar(model_id_raw))
  if (nrow(unparsed) > 0) {
    stop(
      "Could not extract a model id from these biomedical filenames: ",
      paste(unparsed$file_name, collapse = ", ")
    )
  }

  duplicates <- manifest %>%
    filter(!is.na(language_from_file)) %>%
    count(model_id, language_from_file) %>%
    filter(n > 1)
  if (nrow(duplicates) > 0) {
    stop(
      "More than one input file was found for these model/language pairs: ",
      paste0(duplicates$model_id, "/", duplicates$language_from_file, collapse = ", "),
      ". Restrict the discovery root, feature families, or vLLM run date."
    )
  }

  manifest
}

read_manifest_files <- function(manifest) {
  id_like_cols <- unique(c(
    ID_COLS, "lang", "language_file", "source_batch", "forced_prefix_fallback",
    "source_origin", "contrast_group", "feature_family"
  ))

  map_dfr(seq_len(nrow(manifest)), function(i) {
    meta <- manifest[i, ]
    log_msg(
      "Reading biomedical file ", i, "/", nrow(manifest), " [",
      meta$model_id[[1]], "/",
      ifelse(is.na(meta$language_from_file[[1]]), "record metadata", meta$language_from_file[[1]]),
      "]: ", meta$file[[1]]
    )

    x <- read_csv(meta$file[[1]], show_col_types = FALSE)
    present_id_cols <- intersect(id_like_cols, names(x))
    if (length(present_id_cols) > 0) {
      x <- x %>% mutate(across(all_of(present_id_cols), as.character))
    }

    x %>% mutate(
      model_id = meta$model_id[[1]],
      model_label = meta$model_label[[1]],
      domain = "biomedical",
      language_file = meta$language_from_file[[1]],
      source_batch = meta$file_stub[[1]],
      source_origin = meta$source_origin[[1]],
      contrast_group = meta$contrast_group[[1]],
      feature_family = meta$feature_family[[1]]
    )
  })
}

read_real_general_files <- function(input_dir,
                                    pattern = GENERAL_INPUT_PATTERN) {
  if (!dir.exists(input_dir)) stop("Real-general directory does not exist: ", input_dir)
  files <- list.files(input_dir, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) stop("No real-general feature files found in: ", input_dir)

  map_dfr(files, function(file) {
    log_msg("Reading real-general file: ", file)
    x <- read_csv(file, show_col_types = FALSE)
    id_like <- intersect(unique(c(ID_COLS, "lang", "forced_prefix_fallback")), names(x))
    if (length(id_like) > 0) x <- x %>% mutate(across(all_of(id_like), as.character))
    x %>% mutate(
      model_id = NA_character_,
      model_label = NA_character_,
      domain = "general",
      language_file = NA_character_,
      source_batch = tools::file_path_sans_ext(basename(file)),
      source_origin = "real",
      contrast_group = "real_general",
      feature_family = "real_general"
    )
  })
}

read_real_biomedical_files <- function(input_dir,
                                       pattern = BIOMEDICAL_INPUT_PATTERN) {
  if (!dir.exists(input_dir)) {
    stop("Real-biomedical directory does not exist: ", input_dir)
  }
  files <- list.files(input_dir, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) {
    stop("No real-biomedical feature files found in: ", input_dir)
  }

  out <- map_dfr(files, function(file) {
    log_msg("Reading real-biomedical file: ", file)
    x <- read_csv(file, show_col_types = FALSE)
    id_like <- intersect(unique(c(ID_COLS, "lang", "forced_prefix_fallback")), names(x))
    if (length(id_like) > 0) {
      x <- x %>% mutate(across(all_of(id_like), as.character))
    }

    # If a mixed-domain file is ever placed here, retain only its explicitly
    # biomedical rows. Files without a domain column are treated as biomedical
    # because the directory itself is the declared reference source.
    if ("domain" %in% names(x)) {
      x <- x %>%
        filter(str_to_lower(trimws(domain)) == "biomedical")
    }

    language_from_path <- tryCatch(
      extract_biomedical_language(file),
      error = function(e) NA_character_
    )

    x %>% mutate(
      model_id = NA_character_,
      model_label = NA_character_,
      domain = "biomedical",
      language_file = language_from_path,
      source_batch = tools::file_path_sans_ext(basename(file)),
      source_origin = "real",
      contrast_group = "real_biomedical_reference",
      feature_family = "real_biomedical"
    )
  })

  if (nrow(out) == 0) {
    stop("No rows with domain == 'biomedical' were found in: ", input_dir)
  }
  out
}

discover_real_biomedical_manifest <- function(
    input_dir,
    pattern = BIOMEDICAL_INPUT_PATTERN,
    iso3_to_iso2 = ISO3_TO_ISO2) {
  if (!dir.exists(input_dir)) {
    stop("Real-biomedical directory does not exist: ", input_dir)
  }
  files <- list.files(input_dir, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) {
    stop("No real-biomedical feature files found in: ", input_dir)
  }

  tibble(
    file = files,
    file_name = basename(files),
    file_stub = tools::file_path_sans_ext(basename(files)),
    model_id_raw = "real",
    generation_input = NA_character_,
    model_id = "real",
    model_label = "real",
    language_from_file = map_chr(
      files,
      ~ extract_biomedical_language(.x, iso3_to_iso2, required = FALSE)
    ),
    domain = "biomedical",
    source_origin = "real",
    contrast_group = "biomedical__real",
    feature_family = "real_biomedical"
  ) %>%
    arrange(language_from_file, file)
}

discover_available_model_ids <- function() {
  model_ids <- tryCatch(
    discover_biomedical_manifest(
      root = BIOMEDICAL_OUTPUTS_DIR,
      families = BIOMEDICAL_FAMILIES,
      pattern = BIOMEDICAL_INPUT_PATTERN,
      vllm_run_date = VLLM_RUN_DATE,
      iso3_to_iso2 = ISO3_TO_ISO2
    )$model_id,
    error = function(error) character()
  )

  has_real_biomedical <- dir.exists(REAL_BIOMEDICAL_DIR) &&
    length(list.files(
      REAL_BIOMEDICAL_DIR,
      pattern = BIOMEDICAL_INPUT_PATTERN
    )) > 0
  if (has_real_biomedical) model_ids <- c(model_ids, "real")

  model_ids <- sort(unique(model_ids))
  if (length(model_ids) == 0) {
    stop("No biomedical feature inputs were discovered")
  }
  model_ids
}

add_language <- function(df, country_col = "country", iso3_to_iso2 = ISO3_TO_ISO2) {
  has_country <- country_col %in% names(df)
  
  df2 <- if (has_country) {
    df %>%
      mutate(
        lang_raw = str_to_lower(str_extract(.data[[country_col]], "[^_]+$")),
        language_from_country = case_when(
          str_detect(lang_raw, "^[a-z]{2}$") ~ lang_raw,
          str_detect(lang_raw, "^[a-z]{3}$") & lang_raw %in% names(iso3_to_iso2) ~ unname(iso3_to_iso2[lang_raw]),
          TRUE ~ NA_character_
        )
      )
  } else {
    df %>%
      mutate(
        lang_raw = NA_character_,
        language_from_country = NA_character_
      )
  }
  
  df2 <- df2 %>%
    mutate(language = coalesce(language_from_country, language_file))
  
  if (any(is.na(df2$language))) {
    stop("Could not derive language for some rows.")
  }
  
  if (has_country) {
    mismatch <- df2 %>%
      filter(
        !is.na(language_from_country),
        !is.na(language_file),
        language_from_country != language_file
      )
    
    if (nrow(mismatch) > 0) {
      stop(
        "Language mismatch between filename and country column in files: ",
        paste(unique(mismatch$source_batch), collapse = ", ")
      )
    }
  }
  
  df2 %>%
    select(-lang_raw, -language_from_country)
}

normalize_false_flag <- function(x) {
  x_chr <- str_to_lower(trimws(as.character(x)))
  
  case_when(
    x_chr %in% c("false", "f", "0", "no", "n") ~ FALSE,
    x_chr %in% c("true",  "t", "1", "yes", "y") ~ TRUE,
    is.na(x) | x_chr %in% c("na", "null", "") ~ NA,
    TRUE ~ NA
  )
}

zscore_by_model_language <- function(df, feats, eps = 1e-12) {
  df %>%
    group_by(model_id, language) %>%
    mutate(across(all_of(feats), ~ {
      m <- mean(., na.rm = TRUE)
      s <- sd(., na.rm = TRUE)
      if (!is.finite(s) || s <= eps) return(NA_real_)
      (. - m) / s
    })) %>%
    ungroup()
}

wasserstein1_exact_1d <- function(x, y) {
  x <- x[is.finite(x)]
  y <- y[is.finite(y)]
  if (length(x) < 2 || length(y) < 2) return(NA_real_)
  xs <- sort(x)
  ys <- sort(y)
  nx <- length(xs)
  ny <- length(ys)
  z <- sort(unique(c(xs, ys)))
  Fx <- findInterval(z, xs) / nx
  Fy <- findInterval(z, ys) / ny
  dz <- diff(z)
  if (length(dz) == 0) return(0)
  sum(abs(Fx[-length(Fx)] - Fy[-length(Fy)]) * dz)
}

jsd_kde <- function(x, y, grid_n = 512, bw = "nrd0", eps = 1e-12) {
  x <- x[is.finite(x)]
  y <- y[is.finite(y)]
  if (length(x) < 2 || length(y) < 2) return(NA_real_)
  rng <- range(c(x, y))
  if (!all(is.finite(rng)) || diff(rng) <= 0) return(NA_real_)
  
  dx <- density(x, from = rng[1], to = rng[2], n = grid_n, bw = bw)
  dy <- density(y, from = rng[1], to = rng[2], n = grid_n, bw = bw)
  
  px <- pmax(dx$y, 0) + eps
  py <- pmax(dy$y, 0) + eps
  px <- px / sum(px)
  py <- py / sum(py)
  m  <- 0.5 * (px + py)
  
  kl <- function(p, q) sum(p * log2(p / q))
  0.5 * kl(px, m) + 0.5 * kl(py, m)
}

# =========================================================
# FEATURE PRUNING
# =========================================================

find_degenerate_lang_features <- function(df, feats, min_n = 5, eps = 1e-12) {
  stats <- df %>%
    select(model_id, language, domain, all_of(feats)) %>%
    pivot_longer(all_of(feats), names_to = "feature", values_to = "value") %>%
    mutate(value = as.numeric(value)) %>%
    group_by(model_id, language, domain, feature) %>%
    summarise(
      n = sum(is.finite(value)),
      mean = mean(value, na.rm = TRUE),
      sd = sd(value, na.rm = TRUE),
      .groups = "drop"
    )
  
  bad <- stats %>%
    group_by(model_id, language, feature) %>%
    summarise(
      n_domains_present = sum(n >= min_n),
      any_sd_tiny = any(!is.finite(sd) | sd <= eps),
      any_all_zero = any(isTRUE(all.equal(mean, 0)) & isTRUE(all.equal(sd, 0))),
      .groups = "drop"
    ) %>%
    filter(n_domains_present < 2 | any_sd_tiny | any_all_zero) %>%
    select(model_id, language, feature)
  
  list(stats = stats, bad = bad)
}

prune_collinear_per_language <- function(df, feats_by_lang,
                                         cutoff = 0.95,
                                         method = "spearman",
                                         center_within_domain = TRUE,
                                         cap_per_domain = 3000,
                                         seed = 123) {
  set.seed(seed)
  
  keys <- df %>% distinct(model_id, language) %>% arrange(model_id, language)
  drops_all <- list()
  reps_all  <- list()
  
  for (i in seq_len(nrow(keys))) {
    model_key <- keys$model_id[[i]]
    lang <- keys$language[[i]]
    feats <- feats_by_lang %>%
      filter(.data$model_id == .env$model_key, .data$language == .env$lang) %>%
      pull(feature) %>% unique()
    if (length(feats) < 2) next
    
    d <- df %>%
      filter(.data$model_id == .env$model_key, .data$language == .env$lang) %>%
      select(domain, all_of(feats))
    
    d <- d %>%
      group_by(domain) %>%
      group_modify(~ {
        if (nrow(.x) <= cap_per_domain) .x else .x[sample.int(nrow(.x), cap_per_domain), , drop = FALSE]
      }) %>%
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
    idx <- which(abs(C) >= cutoff & upper.tri(C), arr.ind = TRUE)
    if (nrow(idx) == 0) next
    
    edges <- tibble(
      f1 = colnames(C)[idx[, 1]],
      f2 = colnames(C)[idx[, 2]]
    )
    
    comp_tbl <- alignment_components(nodes = colnames(C), edges_df = edges) %>%
      mutate(model_id = .env$model_key, language = .env$lang)
    
    var_tbl <- df %>%
      filter(.data$model_id == .env$model_key, .data$language == .env$lang) %>%
      summarise(across(all_of(feats), ~ var(., na.rm = TRUE))) %>%
      pivot_longer(everything(), names_to = "feature", values_to = "var")
    
    comp_tbl <- comp_tbl %>% left_join(var_tbl, by = "feature")
    
    reps <- comp_tbl %>%
      group_by(comp) %>%
      arrange(desc(var), feature, .by_group = TRUE) %>%
      slice(1) %>%
      ungroup() %>%
      transmute(model_id = .env$model_key, language = .env$lang, comp, representative = feature)
    
    rep_set <- reps$representative
    
    drops <- comp_tbl %>%
      filter(!feature %in% rep_set) %>%
      transmute(model_id = .env$model_key, language = .env$lang, feature)
    
    key <- paste(model_key, lang, sep = "||")
    drops_all[[key]] <- drops
    reps_all[[key]]  <- reps
  }
  
  list(
    drops = bind_rows(drops_all),
    reps  = bind_rows(reps_all)
  )
}

# =========================================================
# SHIFT METRICS / SCORING
# =========================================================

compute_shift_metrics_one_lang <- function(d_lang, feats, min_n = 5,
                                           jsd_grid_n = 512, jsd_bw = "nrd0",
                                           jsd_eps = 1e-12, eps_sd = 1e-12) {
  out <- vector("list", length(feats))
  names(out) <- feats
  
  dom <- as.character(d_lang$domain)
  ok_dom <- dom %in% c("general", "biomedical")
  d_lang <- d_lang[ok_dom, , drop = FALSE]
  dom <- as.character(d_lang$domain)
  
  for (j in seq_along(feats)) {
    f <- feats[j]
    v <- d_lang[[f]]
    xg <- v[dom == "general"]
    xb <- v[dom == "biomedical"]
    xg <- xg[is.finite(xg)]
    xb <- xb[is.finite(xb)]
    
    ng <- length(xg)
    nb <- length(xb)
    if (ng < min_n || nb < min_n) {
      out[[j]] <- NULL
      next
    }
    
    mg <- mean(xg)
    mb <- mean(xb)
    sg <- sd(xg)
    sb <- sd(xb)
    if (!is.finite(sg) || !is.finite(sb) || sg <= eps_sd || sb <= eps_sd) {
      out[[j]] <- NULL
      next
    }
    
    delta <- mg - mb
    spooled <- sqrt(((ng - 1) * sg^2 + (nb - 1) * sb^2) / (ng + nb - 2))
    if (!is.finite(spooled) || spooled <= eps_sd) {
      out[[j]] <- NULL
      next
    }
    
    dval <- delta / spooled
    w1   <- wasserstein1_exact_1d(xg, xb)
    jsv  <- jsd_kde(xg, xb, grid_n = jsd_grid_n, bw = jsd_bw, eps = jsd_eps)
    
    out[[j]] <- tibble(
      feature  = f,
      delta_z  = delta,
      cohens_d = dval,
      w1_z     = w1,
      jsd      = jsv
    )
  }
  
  bind_rows(out)
}

score_features_per_language <- function(df_z, feats_by_lang,
                                        metrics = c("cohens_d", "w1_z", "jsd"),
                                        B = 500, seed = 123,
                                        min_n = 5,
                                        jsd_grid_n = 512, jsd_bw = "nrd0", jsd_eps = 1e-12,
                                        eps_sd = 1e-12,
                                        reference_label = "real general",
                                        target_label = "biomedical") {
  set.seed(seed)
  keys <- df_z %>% distinct(model_id, model_label, language) %>% arrange(model_id, language)
  res_all <- list()
  
  for (i in seq_len(nrow(keys))) {
    model_key <- keys$model_id[[i]]
    model_label_key <- keys$model_label[[i]]
    lang <- keys$language[[i]]
    log_msg("Scoring features for model/language: ", model_key, "/", as.character(lang))
    
    feats <- feats_by_lang %>%
      filter(.data$model_id == .env$model_key, .data$language == .env$lang) %>%
      pull(feature) %>% unique()
    if (length(feats) == 0) next
    
    d_lang <- df_z %>%
      filter(.data$model_id == .env$model_key, .data$language == .env$lang) %>%
      select(domain, all_of(feats))
    dom <- as.character(d_lang$domain)
    log_msg(
      "  candidates=", length(feats),
      ", n_general=", sum(dom == "general"),
      ", n_biomedical=", sum(dom == "biomedical")
    )
    if (sum(dom == "general") < min_n || sum(dom == "biomedical") < min_n) next
    
    obs <- compute_shift_metrics_one_lang(
      d_lang = d_lang,
      feats = feats,
      min_n = min_n,
      jsd_grid_n = jsd_grid_n,
      jsd_bw = jsd_bw,
      jsd_eps = jsd_eps,
      eps_sd = eps_sd
    )
    if (nrow(obs) == 0) {
      log_msg("  no features produced finite two-domain shift metrics")
      next
    }
    
    obs_long <- obs %>%
      select(feature, delta_z, all_of(metrics)) %>%
      pivot_longer(all_of(metrics), names_to = "metric", values_to = "value") %>%
      mutate(abs_value = abs(value)) %>%
      group_by(metric) %>%
      mutate(pct = percent_rank(abs_value)) %>%
      ungroup()
    
    obs_score <- obs_long %>%
      group_by(feature) %>%
      summarise(
        model_id = .env$model_key,
        model_label = .env$model_label_key,
        language = as.character(lang),
        delta_z = first(delta_z),
        borda_score = mean(pct, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      left_join(
        obs %>% mutate(model_id = .env$model_key, language = as.character(.env$lang)),
        by = c("model_id", "language", "feature", "delta_z")
      ) %>%
      select(model_id, model_label, language, feature, all_of(metrics), borda_score, delta_z)
    
    null_scores <- matrix(NA_real_, nrow = nrow(obs_score), ncol = B)
    rownames(null_scores) <- obs_score$feature
    dom0 <- as.character(d_lang$domain)
    
    for (b in seq_len(B)) {
      if (b %% 50 == 0) log_msg("  ", model_key, "/", as.character(lang), ": permutation ", b, "/", B)
      
      d_perm <- d_lang
      d_perm$domain <- sample(dom0, length(dom0), replace = FALSE)
      
      pm <- compute_shift_metrics_one_lang(
        d_lang = d_perm,
        feats = feats,
        min_n = min_n,
        jsd_grid_n = jsd_grid_n,
        jsd_bw = jsd_bw,
        jsd_eps = jsd_eps,
        eps_sd = eps_sd
      )
      if (nrow(pm) == 0) next
      
      pm_long <- pm %>%
        select(feature, all_of(metrics)) %>%
        pivot_longer(all_of(metrics), names_to = "metric", values_to = "value") %>%
        mutate(abs_value = abs(value)) %>%
        group_by(metric) %>%
        mutate(pct = percent_rank(abs_value)) %>%
        ungroup()
      
      pm_score <- pm_long %>%
        group_by(feature) %>%
        summarise(borda_score = mean(pct, na.rm = TRUE), .groups = "drop") %>%
        right_join(tibble(feature = obs_score$feature), by = "feature") %>%
        mutate(borda_score = replace_na(borda_score, 0))
      
      null_scores[pm_score$feature, b] <- pm_score$borda_score
    }
    
    obs_vec <- obs_score$borda_score
    features <- obs_score$feature
    
    p_perm <- vapply(seq_along(features), function(i) {
      f <- features[i]
      ns <- null_scores[f, ]
      ns <- ns[is.finite(ns)]
      if (length(ns) == 0) return(NA_real_)
      (1 + sum(ns >= obs_vec[i])) / (1 + length(ns))
    }, numeric(1))
    
    q_perm <- p.adjust(p_perm, method = "BH")
    
    out <- obs_score %>%
      mutate(
        p_perm = p_perm,
        q_perm = q_perm,
        direction = case_when(
          is.na(delta_z) ~ NA_character_,
          delta_z > 0 ~ paste0(.env$reference_label, " > ", .data$model_label, " ", .env$target_label),
          delta_z < 0 ~ paste0(.env$reference_label, " < ", .data$model_label, " ", .env$target_label),
          TRUE ~ "no change"
        )
      ) %>%
      arrange(q_perm, desc(borda_score), feature)
    
    res_all[[paste(model_key, lang, sep = "||")]] <- out
  }
  
  bind_rows(res_all)
}

# Compute only the observed (non-permuted) feature metrics and within-cell
# Borda rankings. This keeps the repeated sensitivity analysis tractable: the
# full-data permutation p/q values remain the inferential results, while the
# balanced repetitions quantify sampling stability around their effect sizes
# and rankings.
score_observed_features <- function(df_z, feats_by_lang,
                                    metrics = c("cohens_d", "w1_z", "jsd"),
                                    min_n = 5,
                                    jsd_grid_n = 512, jsd_bw = "nrd0",
                                    jsd_eps = 1e-12, eps_sd = 1e-12) {
  keys <- df_z %>%
    distinct(model_id, model_label, language) %>%
    arrange(model_id, language)
  res_all <- list()

  for (i in seq_len(nrow(keys))) {
    model_key <- keys$model_id[[i]]
    model_label_key <- keys$model_label[[i]]
    lang <- keys$language[[i]]

    feats <- feats_by_lang %>%
      filter(.data$model_id == .env$model_key, .data$language == .env$lang) %>%
      pull(feature) %>%
      unique()
    if (length(feats) == 0) next

    d_lang <- df_z %>%
      filter(.data$model_id == .env$model_key, .data$language == .env$lang) %>%
      select(domain, all_of(feats))

    obs <- compute_shift_metrics_one_lang(
      d_lang = d_lang,
      feats = feats,
      min_n = min_n,
      jsd_grid_n = jsd_grid_n,
      jsd_bw = jsd_bw,
      jsd_eps = jsd_eps,
      eps_sd = eps_sd
    )
    if (nrow(obs) == 0) next

    scores <- obs %>%
      select(feature, delta_z, all_of(metrics)) %>%
      pivot_longer(all_of(metrics), names_to = "metric", values_to = "value") %>%
      mutate(abs_value = abs(value)) %>%
      group_by(metric) %>%
      mutate(pct = percent_rank(abs_value)) %>%
      ungroup() %>%
      group_by(feature) %>%
      summarise(
        delta_z = first(delta_z),
        borda_score = mean(pct, na.rm = TRUE),
        .groups = "drop"
      )

    res_all[[paste(model_key, lang, sep = "||")]] <- obs %>%
      left_join(scores, by = c("feature", "delta_z")) %>%
      mutate(
        model_id = model_key,
        model_label = model_label_key,
        language = as.character(lang)
      ) %>%
      select(model_id, model_label, language, feature,
             all_of(metrics), borda_score, delta_z)
  }

  bind_rows(res_all)
}

run_balanced_stability <- function(df_z, feats_by_lang, full_scores,
                                   reps = 100, target_n = NA_integer_,
                                   conf_level = 0.95, top_k = 20,
                                   seed = 123, min_n = 5,
                                   metrics = c("cohens_d", "w1_z", "jsd"),
                                   jsd_grid_n = 512, jsd_bw = "nrd0",
                                   jsd_eps = 1e-12, eps_sd = 1e-12) {
  d_work <- df_z %>%
    filter(as.character(domain) %in% c("general", "biomedical")) %>%
    mutate(domain = as.character(domain)) %>%
    group_by(model_id, language, domain) %>%
    mutate(.balance_position = row_number()) %>%
    ungroup()

  cell_counts <- d_work %>%
    count(model_id, model_label, language, domain, name = "n_available") %>%
    arrange(model_id, language, domain)

  if (nrow(cell_counts) == 0) stop("No model/language/domain cells are available for balancing")
  minimum_available <- min(cell_counts$n_available)
  target_n_used <- if (is.na(target_n)) minimum_available else target_n

  if (target_n_used > minimum_available) {
    limiting <- cell_counts %>% filter(n_available == minimum_available)
    stop(
      "Balanced target n=", target_n_used,
      " exceeds the smallest available cell (n=", minimum_available, "): ",
      paste0(limiting$model_id, "/", limiting$language, "/", limiting$domain,
             collapse = ", ")
    )
  }
  if (target_n_used < min_n) {
    stop("Balanced target n=", target_n_used, " is below MIN_N_PER_DOMAIN=", min_n)
  }

  alpha <- (1 - conf_level) / 2
  general_limits <- cell_counts %>%
    filter(domain == "general") %>%
    group_by(language) %>%
    summarise(n_common_positions = min(n_available), .groups = "drop")

  full_reference <- full_scores %>%
    select(model_id, model_label, language, feature,
           full_borda_score = borda_score,
           full_delta_z = delta_z,
           full_cohens_d = cohens_d,
           full_w1_z = w1_z,
           full_jsd = jsd,
           full_p_perm = p_perm,
           full_q_perm = q_perm) %>%
    distinct()

  draws_all <- vector("list", reps)
  for (b in seq_len(reps)) {
    if (b == 1 || b %% 10 == 0 || b == reps) {
      log_msg("Balanced stability repetition ", b, "/", reps,
              " (target n per domain=", target_n_used, ")")
    }
    set.seed(seed + b)

    # General rows were expanded from the same real-language pool for every
    # model. Sampling row positions once per language keeps that reference
    # sample shared across models within a repetition.
    general_keep <- d_work %>%
      filter(domain == "general") %>%
      distinct(language) %>%
      left_join(general_limits, by = "language") %>%
      mutate(
        .balance_position = map(
          n_common_positions,
          ~ sample.int(.x, target_n_used, replace = FALSE)
        )
      ) %>%
      select(-n_common_positions) %>%
      unnest_longer(.balance_position)

    d_general <- d_work %>%
      filter(domain == "general") %>%
      inner_join(general_keep, by = c("language", ".balance_position"))

    # Model-generated biomedical rows are sampled independently within each
    # model/language cell, which is the uncertainty this analysis targets.
    d_biomedical <- d_work %>%
      filter(domain == "biomedical") %>%
      group_by(model_id, language) %>%
      slice_sample(n = target_n_used, replace = FALSE) %>%
      ungroup()

    observed <- score_observed_features(
      df_z = bind_rows(d_general, d_biomedical),
      feats_by_lang = feats_by_lang,
      metrics = metrics,
      min_n = min_n,
      jsd_grid_n = jsd_grid_n,
      jsd_bw = jsd_bw,
      jsd_eps = jsd_eps,
      eps_sd = eps_sd
    )

    draws_all[[b]] <- observed %>%
      mutate(iteration = b, target_n = target_n_used, .before = 1)
  }

  draws <- bind_rows(draws_all) %>%
    group_by(iteration, model_id, model_label, language) %>%
    mutate(
      borda_rank = min_rank(desc(borda_score)),
      n_features_scored = n(),
      is_top_k = borda_rank <= pmin(top_k, n_features_scored)
    ) %>%
    ungroup() %>%
    left_join(full_reference, by = c("model_id", "model_label", "language", "feature"))

  metric_summary <- draws %>%
    select(iteration, model_id, model_label, language, feature,
           all_of(metrics), borda_score, delta_z) %>%
    pivot_longer(
      cols = c(all_of(metrics), borda_score, delta_z),
      names_to = "metric",
      values_to = "value"
    ) %>%
    group_by(model_id, model_label, language, feature, metric) %>%
    summarise(
      n_iterations = n_distinct(iteration),
      mean = mean(value, na.rm = TRUE),
      sd = sd(value, na.rm = TRUE),
      median = median(value, na.rm = TRUE),
      conf_low = quantile(value, probs = alpha, na.rm = TRUE, names = FALSE),
      conf_high = quantile(value, probs = 1 - alpha, na.rm = TRUE, names = FALSE),
      .groups = "drop"
    )

  feature_summary <- draws %>%
    group_by(model_id, model_label, language, feature) %>%
    summarise(
      n_iterations = n_distinct(iteration),
      iteration_coverage = n_iterations / reps,
      median_borda_rank = median(borda_rank, na.rm = TRUE),
      median_borda_score = median(borda_score, na.rm = TRUE),
      top_k_rate = mean(is_top_k, na.rm = TRUE),
      delta_sign_agreement = if (is.finite(first(full_delta_z)) && first(full_delta_z) != 0) {
        mean(sign(delta_z) == sign(first(full_delta_z)), na.rm = TRUE)
      } else {
        NA_real_
      },
      full_borda_score = first(full_borda_score),
      full_delta_z = first(full_delta_z),
      full_cohens_d = first(full_cohens_d),
      full_w1_z = first(full_w1_z),
      full_jsd = first(full_jsd),
      full_p_perm = first(full_p_perm),
      full_q_perm = first(full_q_perm),
      .groups = "drop"
    ) %>%
    arrange(model_id, language, median_borda_rank, feature)

  rank_correlations <- draws %>%
    group_by(iteration, model_id, model_label, language) %>%
    summarise(
      n_common_features = sum(is.finite(borda_score) & is.finite(full_borda_score)),
      spearman_rho = {
        ok <- is.finite(borda_score) & is.finite(full_borda_score)
        if (sum(ok) >= 3) {
          suppressWarnings(cor(borda_score[ok], full_borda_score[ok], method = "spearman"))
        } else {
          NA_real_
        }
      },
      .groups = "drop"
    )

  rank_summary <- rank_correlations %>%
    group_by(model_id, model_label, language) %>%
    summarise(
      n_iterations = n_distinct(iteration),
      median_spearman_rho = median(spearman_rho, na.rm = TRUE),
      conf_low = quantile(spearman_rho, probs = alpha, na.rm = TRUE, names = FALSE),
      conf_high = quantile(spearman_rho, probs = 1 - alpha, na.rm = TRUE, names = FALSE),
      .groups = "drop"
    ) %>%
    arrange(model_id, language)

  config <- tibble(
    repetitions = reps,
    target_n_requested = target_n,
    target_n_used = target_n_used,
    minimum_available_in_current_run = minimum_available,
    confidence_level = conf_level,
    top_k = top_k,
    seed = seed
  )

  list(
    config = config,
    cell_counts = cell_counts,
    draws = draws,
    metric_summary = metric_summary,
    feature_summary = feature_summary,
    rank_correlations = rank_correlations,
    rank_summary = rank_summary
  )
}

cluster_features_per_language <- function(df_z, per_lang_scores,
                                          q_max = 0.05, top_n = NULL,
                                          cutoff_corr = 0.95, method = "spearman",
                                          center_within_domain = TRUE,
                                          cap_per_domain = 3000,
                                          seed = 123,
                                          reference_label = "real general",
                                          target_label = "biomedical") {
  set.seed(seed)

  empty_clusters <- tibble(
    model_id = character(),
    model_label = character(),
    language = character(),
    cluster_id = character(),
    winner_feature = character(),
    winner_score = double(),
    winner_q = double(),
    winner_direction = character(),
    n_members = integer(),
    n_non_na = integer(),
    n_pos = integer(),
    n_neg = integer(),
    n_zero = integer(),
    cluster_direction = character(),
    frac_majority = double(),
    n_deviant = integer(),
    has_deviants = logical(),
    is_unanimous = logical(),
    cluster_delta_z = double(),
    members = character()
  )

  if (nrow(per_lang_scores) == 0) return(empty_clusters)

  keys <- per_lang_scores %>%
    distinct(model_id, model_label, language) %>%
    arrange(model_id, language)
  clusters_all <- list()
  
  for (i in seq_len(nrow(keys))) {
    model_key <- keys$model_id[[i]]
    model_label_key <- keys$model_label[[i]]
    lang <- keys$language[[i]]
    log_msg("Clustering scored features for model/language: ", model_key, "/", as.character(lang))
    
    sc <- per_lang_scores %>%
      filter(.data$model_id == .env$model_key, .data$language == .env$lang)
    
    if (!is.null(top_n)) {
      feats <- sc %>%
        arrange(q_perm, desc(borda_score)) %>%
        slice_head(n = top_n) %>%
        pull(feature)
    } else {
      feats <- sc %>%
        filter(is.finite(q_perm), q_perm <= q_max) %>%
        pull(feature)
    }
    
    feats <- unique(feats)
    if (length(feats) == 0) {
      log_msg("  no features passed the clustering threshold; skipping")
      next
    }

    # A single selected feature is a valid singleton cluster. Correlations are
    # needed only when two or more features survived selection.
    edges <- tibble(f1 = character(), f2 = character())

    if (length(feats) >= 2) {
      d <- df_z %>%
        filter(
          .data$model_id == .env$model_key,
          as.character(.data$language) == as.character(.env$lang)
        ) %>%
        select(domain, all_of(feats)) %>%
        group_by(domain) %>%
        group_modify(~ {
          if (nrow(.x) <= cap_per_domain) .x else .x[sample.int(nrow(.x), cap_per_domain), , drop = FALSE]
        }) %>%
        ungroup()

      if (center_within_domain) {
        d <- d %>%
          group_by(domain) %>%
          mutate(across(all_of(feats), ~ . - mean(., na.rm = TRUE))) %>%
          ungroup()
      }

      X <- d %>% select(-domain) %>% as.matrix()
      C <- suppressWarnings(cor(X, use = "pairwise.complete.obs", method = method))
      idx <- which(abs(C) >= cutoff_corr & upper.tri(C), arr.ind = TRUE)

      if (nrow(idx) > 0) {
        edges <- tibble(
          f1 = colnames(C)[idx[, 1]],
          f2 = colnames(C)[idx[, 2]]
        )
      }
    }
    
    comp_tbl <- alignment_components(nodes = feats, edges_df = edges) %>%
      mutate(
        model_id = .env$model_key,
        model_label = .env$model_label_key,
        language = as.character(.env$lang),
        cluster_id = paste0(as.character(.env$lang), "__", comp)
      ) %>%
      select(model_id, model_label, language, cluster_id, feature)
    
    comp_sc <- comp_tbl %>%
      left_join(sc %>% select(feature, borda_score, q_perm, delta_z, direction), by = "feature")
    
    winners <- comp_sc %>%
      group_by(model_id, model_label, language, cluster_id) %>%
      arrange(q_perm, desc(borda_score), feature, .by_group = TRUE) %>%
      slice(1) %>%
      ungroup() %>%
      transmute(
        model_id,
        model_label,
        language,
        cluster_id,
        winner_feature = feature,
        winner_score = borda_score,
        winner_q = q_perm,
        winner_direction = direction
      )
    
    cluster_dir <- comp_sc %>%
      group_by(model_id, model_label, language, cluster_id) %>%
      summarise(
        n_members = n_distinct(feature),
        n_non_na  = sum(is.finite(delta_z)),
        n_pos     = sum(is.finite(delta_z) & delta_z > 0),
        n_neg     = sum(is.finite(delta_z) & delta_z < 0),
        n_zero    = sum(is.finite(delta_z) & delta_z == 0),
        cluster_direction = case_when(
          n_non_na == 0 ~ NA_character_,
          n_pos >= n_neg & n_pos >= n_zero ~ paste0(.env$reference_label, " > ", .env$model_label_key, " ", .env$target_label),
          n_neg >= n_pos & n_neg >= n_zero ~ paste0(.env$reference_label, " < ", .env$model_label_key, " ", .env$target_label),
          TRUE ~ "no change"
        ),
        majority_n = pmax(n_pos, n_neg, n_zero),
        frac_majority = ifelse(n_non_na > 0, majority_n / n_non_na, NA_real_),
        n_deviant = ifelse(n_non_na > 0, n_non_na - majority_n, NA_integer_),
        has_deviants = n_deviant > 0,
        is_unanimous = ifelse(n_non_na > 0, majority_n == n_non_na, NA),
        cluster_delta_z = if (n_non_na > 0) median(delta_z[is.finite(delta_z)]) else NA_real_,
        members = paste(sort(unique(feature)), collapse = ", "),
        .groups = "drop"
      ) %>%
      select(-majority_n)
    
    clusters_all[[paste(model_key, lang, sep = "||")]] <- winners %>%
      left_join(cluster_dir, by = c("model_id", "model_label", "language", "cluster_id"))
  }

  result <- bind_rows(clusters_all)
  if (nrow(result) == 0) return(empty_clusters)

  result %>% arrange(winner_q, desc(winner_score))
}

fingerprint_per_language_feature <- function(per_lang_scores, K = 20) {
  per_lang_scores %>%
    group_by(model_id, language) %>%
    arrange(q_perm, desc(borda_score), .by_group = TRUE) %>%
    filter(q_perm <= 0.05) %>%
    ungroup() %>%
    mutate(fingerprint_type = "feature")
}

fingerprint_per_language_cluster <- function(per_lang_clusters, K = 20) {
  per_lang_clusters %>%
    group_by(model_id, language) %>%
    arrange(winner_q, desc(winner_score), .by_group = TRUE) %>%
    slice_head(n = K) %>%
    ungroup() %>%
    mutate(fingerprint_type = "cluster")
}

build_meta_clusters <- function(per_lang_clusters, jaccs = c(0.2, 0.3, 0.4)) {
  cl_sets <- per_lang_clusters %>%
    transmute(
      model_id,
      model_label,
      cluster_key = paste(model_id, language, cluster_id, sep = "||"),
      language = as.character(language),
      cluster_id = as.character(cluster_id),
      winner_feature,
      winner_score,
      winner_q,
      direction = cluster_direction,
      members
    ) %>%
    distinct()
  
  if (nrow(cl_sets) == 0) return(tibble())
  
  jaccard <- function(a, b) {
    ia <- intersect(a, b)
    ua <- union(a, b)
    if (length(ua) == 0) return(0)
    length(ia) / length(ua)
  }
  
  out_all <- list()
  models <- sort(unique(cl_sets$model_id))

  for (model_key in models) {
    cl_model <- cl_sets %>% filter(.data$model_id == .env$model_key)
    members_list <- strsplit(cl_model$members, "\\s*,\\s*")
    names(members_list) <- cl_model$cluster_key

    for (jacc in jaccs) {
      log_msg("Building meta-clusters for ", model_key, " at Jaccard threshold: ", jacc)
    
    keys <- cl_model$cluster_key
    n <- length(keys)
    if (n < 2) next
    
    edges <- list()
    k <- 0L
    
    for (i in 1:(n - 1)) {
      li <- cl_model$language[i]
      ai <- members_list[[keys[i]]]
      for (j in (i + 1):n) {
        if (cl_model$language[j] == li) next
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
      tibble(
        f1 = vapply(edges, `[[`, character(1), 1),
        f2 = vapply(edges, `[[`, character(1), 2)
      )
    }
    
    comp <- alignment_components(nodes = keys, edges_df = edges_df) %>%
      mutate(meta_cluster_id = paste0("jacc", jacc, "__", comp))
    
    meta_tbl <- cl_model %>%
      left_join(comp %>% select(feature, comp, meta_cluster_id),
                by = c("cluster_key" = "feature")) %>%
      select(model_id, model_label, meta_cluster_id, comp, language, cluster_id,
             winner_feature, winner_score, winner_q, direction, members) %>%
      mutate(jacc = jacc) %>%
      group_by(model_id, jacc, meta_cluster_id) %>%
      mutate(
        meta_label = {
          ww <- winner_feature
          tab <- sort(table(ww), decreasing = TRUE)
          names(tab)[1]
        }
      ) %>%
      ungroup() %>%
      mutate(meta_cluster_id2 = paste0("jacc", jacc, "__", meta_label))
    out_all[[paste(model_key, jacc, sep = "||")]] <- meta_tbl
    }
  }
  
  bind_rows(out_all)
}

summarize_meta_clusters <- function(meta_clusters, alpha = 0.05) {
  if (nrow(meta_clusters) == 0) return(tibble())
  
  meta_clusters %>%
    group_by(model_id, model_label, jacc, meta_cluster_id) %>%
    summarise(
      n_langs = n_distinct(language),
      langs = paste(sort(unique(language)), collapse = ", "),
      n_sig = sum(is.finite(winner_q) & winner_q <= alpha),
      min_q = suppressWarnings(min(winner_q, na.rm = TRUE)),
      med_q = suppressWarnings(median(winner_q, na.rm = TRUE)),
      dir_mode = {
        d <- na.omit(direction)
        if (length(d) == 0) NA_character_ else names(sort(table(d), decreasing = TRUE))[1]
      },
      dir_agree = {
        d <- na.omit(direction)
        if (length(d) == 0) NA_real_ else max(table(d)) / length(d)
      },
      n_unique_winners = n_distinct(winner_feature),
      winners_by_lang = paste(paste0(language, ":", winner_feature), collapse = "; "),
      .groups = "drop"
    ) %>%
    arrange(model_id, jacc, desc(n_langs), min_q, desc(dir_agree))
}

compute_pair_counts <- function(stable_universal, jacc_threshold = 0.8) {
  if (nrow(stable_universal) == 0) {
    return(tibble(
      model_id = character(), model_label = character(),
      l1 = character(), l2 = character(), n_shared_meta_clusters = integer()
    ))
  }

  stable_universal %>%
    filter(jacc >= jacc_threshold) %>%
    distinct(model_id, model_label, meta_cluster_id, langs) %>%
    mutate(langs_vec = str_split(langs, "\\s*,\\s*")) %>%
    mutate(pairs = map(langs_vec, ~{
      x <- sort(unique(.x))
      if (length(x) < 2) return(tibble(l1 = character(), l2 = character()))
      cmb <- combn(x, 2)
      tibble(l1 = cmb[1, ], l2 = cmb[2, ])
    })) %>%
    select(model_id, model_label, meta_cluster_id, pairs) %>%
    unnest(pairs) %>%
    count(model_id, model_label, l1, l2, name = "n_shared_meta_clusters") %>%
    arrange(model_id, desc(n_shared_meta_clusters))
}

compute_pair_counts_full <- function(stable_universal, pair_counts, jacc_threshold = 0.8) {
  if (nrow(stable_universal) == 0) {
    return(tibble(
      model_id = character(), model_label = character(),
      l1 = character(), l2 = character(), n_shared_meta_clusters = integer()
    ))
  }

  model_langs <- stable_universal %>%
    filter(jacc >= jacc_threshold) %>%
    transmute(model_id, model_label, langs_vec = str_split(langs, "\\s*,\\s*")) %>%
    unnest(langs_vec) %>%
    distinct(model_id, model_label, lang = langs_vec)

  all_pairs <- model_langs %>%
    group_by(model_id, model_label) %>%
    summarise(langs = list(sort(unique(lang))), .groups = "drop") %>%
    mutate(pairs = map(langs, ~ {
      if (length(.x) < 2) return(tibble(l1 = character(), l2 = character()))
      cmb <- combn(.x, 2)
      tibble(l1 = cmb[1, ], l2 = cmb[2, ])
    })) %>%
    select(-langs) %>%
    unnest(pairs)
  
  all_pairs %>%
    left_join(pair_counts, by = c("model_id", "model_label", "l1", "l2")) %>%
    mutate(n_shared_meta_clusters = replace_na(n_shared_meta_clusters, 0L))
}

compute_zero_pairs <- function(pair_counts_full) {
  if (nrow(pair_counts_full) == 0) return(pair_counts_full)

  pair_counts_full %>%
    filter(n_shared_meta_clusters == 0L) %>%
    arrange(model_id, l1, l2)
}

compute_lang_unique_counts <- function(stable_universal, jacc_threshold = 0.8) {
  if (nrow(stable_universal) == 0) {
    return(tibble(
      model_id = character(), model_label = character(),
      language = character(), n_unique_meta_clusters = integer()
    ))
  }

  stable_universal %>%
    filter(jacc >= jacc_threshold, n_langs > 1) %>%
    select(model_id, model_label, meta_cluster_id, langs) %>%
    distinct() %>%
    separate_rows(langs, sep = ",\\s*") %>%
    distinct(model_id, model_label, meta_cluster_id, langs) %>%
    count(model_id, model_label, langs, name = "n_unique_meta_clusters") %>%
    rename(language = langs)
}

# =========================================================
# SECOND CONTRAST: REAL BIOMEDICAL VS OTHER BIOMEDICAL
# =========================================================

run_real_biomedical_contrast <- function() {
  contrast_output_dir <- file.path(OUTPUT_DIR, REAL_BIOMEDICAL_SUBDIR)
  dir.create(contrast_output_dir, recursive = TRUE, showWarnings = FALSE)

  checkpoint_paths <- c(
    df_z = file.path(contrast_output_dir, "df_z.rds"),
    feats_ok = file.path(contrast_output_dir, "feats_ok.rds"),
    per_lang_scores = file.path(contrast_output_dir, "per_lang_scores.rds")
  )
  resume_ready <- RESUME_EXISTING && all(file.exists(checkpoint_paths))

  if (resume_ready) {
    log_msg("Real-biomedical contrast resume mode: loading existing checkpoints")
    df_z <- readRDS(checkpoint_paths[["df_z"]])
    feats_ok <- readRDS(checkpoint_paths[["feats_ok"]])
    per_lang_scores <- readRDS(checkpoint_paths[["per_lang_scores"]])

    required_df_z <- c("model_id", "model_label", "language", "domain")
    required_feats <- c("model_id", "language", "feature")
    required_scores <- c(
      "model_id", "model_label", "language", "feature",
      "cohens_d", "w1_z", "jsd", "borda_score", "p_perm", "q_perm",
      "delta_z", "direction"
    )
    if (!all(required_df_z %in% names(df_z))) {
      stop("Real-biomedical df_z checkpoint is missing required columns")
    }
    if (!all(required_feats %in% names(feats_ok))) {
      stop("Real-biomedical feats_ok checkpoint is missing required columns")
    }
    if (!all(required_scores %in% names(per_lang_scores)) || nrow(per_lang_scores) == 0) {
      stop("Real-biomedical per_lang_scores checkpoint is invalid or empty")
    }

    if (nzchar(SELECTED_MODEL_ID)) {
      checkpoint_models <- unique(as.character(df_z$model_id))
      checkpoint_models <- sort(checkpoint_models[!is.na(checkpoint_models)])
      if (!identical(checkpoint_models, SELECTED_MODEL_ID)) {
        stop(
          "Real-biomedical checkpoints do not belong exclusively to selected model '",
          SELECTED_MODEL_ID, "'. Found: ", paste(checkpoint_models, collapse = ", ")
        )
      }
    }
  } else {
    log_msg("Discovering inputs for real-biomedical vs biomedical contrast")
    manifest_all <- discover_biomedical_manifest(
      root = BIOMEDICAL_OUTPUTS_DIR,
      families = BIOMEDICAL_FAMILIES,
      pattern = BIOMEDICAL_INPUT_PATTERN,
      vllm_run_date = VLLM_RUN_DATE,
      iso3_to_iso2 = ISO3_TO_ISO2
    )

    # Every discovered biomedical file is a target, including upper-bound
    # inputs. The reference is read only from REAL_BIOMEDICAL_DIR.
    target_manifest <- manifest_all

    if (nzchar(SELECTED_MODEL_ID)) {
      target_manifest <- target_manifest %>%
        filter(model_id == SELECTED_MODEL_ID)
      if (nrow(target_manifest) == 0) {
        stop(
          "Selected model '", SELECTED_MODEL_ID,
          "' was not discovered as a biomedical target for the second contrast."
        )
      }
    }

    df_reference <- read_real_biomedical_files(REAL_BIOMEDICAL_DIR) %>%
      add_language()
    reference_languages <- sort(unique(as.character(df_reference$language)))

    target_manifest <- target_manifest %>%
      filter(is.na(language_from_file) | language_from_file %in% reference_languages)
    if (nrow(target_manifest) == 0) {
      stop("No target model/language cells overlap the real-biomedical references")
    }

    reference_files <- list.files(
      REAL_BIOMEDICAL_DIR,
      pattern = BIOMEDICAL_INPUT_PATTERN,
      full.names = TRUE
    )
    reference_manifest <- tibble(
      file = reference_files,
      file_name = basename(reference_files),
      feature_family = "real_biomedical",
      source_origin = "real",
      comparison_role = "reference"
    )
    input_manifest <- bind_rows(
      reference_manifest,
      target_manifest %>% mutate(comparison_role = "target")
    ) %>%
      arrange(comparison_role, model_id, language_from_file)

    target_registry <- target_manifest %>%
      group_by(model_id, model_label, feature_family, generation_input) %>%
      summarise(
        n_languages_in_filenames = n_distinct(language_from_file, na.rm = TRUE),
        languages_in_filenames = paste(
          sort(unique(na.omit(language_from_file))),
          collapse = ", "
        ),
        language_stored_in_records = any(is.na(language_from_file)),
        .groups = "drop"
      ) %>%
      arrange(model_id)

    df_target <- read_manifest_files(target_manifest) %>% add_language()

    if (!"forced_prefix_fallback" %in% names(df_target)) {
      df_target$forced_prefix_fallback <- NA_character_
    }
    if (!"forced_prefix_fallback" %in% names(df_reference)) {
      df_reference$forced_prefix_fallback <- NA_character_
    }
    df_target <- df_target %>%
      mutate(forced_prefix_fallback = normalize_false_flag(forced_prefix_fallback))
    df_reference <- df_reference %>%
      mutate(forced_prefix_fallback = normalize_false_flag(forced_prefix_fallback))

    df_target_fls <- df_target %>%
      filter(is.na(forced_prefix_fallback) | forced_prefix_fallback == FALSE)
    df_reference_fls <- df_reference %>%
      filter(is.na(forced_prefix_fallback) | forced_prefix_fallback == FALSE)

    valid_model_languages <- df_target_fls %>%
      distinct(model_id, model_label, language)

    # Metric functions use the existing two internal levels. Here "general"
    # means the real-biomedical reference; the saved contrast definition makes
    # that mapping explicit and all human-readable directions use the true name.
    df_reference_expanded <- df_reference_fls %>%
      select(-model_id, -model_label) %>%
      inner_join(
        valid_model_languages,
        by = "language",
        relationship = "many-to-many"
      ) %>%
      mutate(
        domain = "general",
        contrast_group = "real_biomedical_reference",
        comparison_role = "reference"
      )
    df_target_fls <- df_target_fls %>%
      mutate(domain = "biomedical", comparison_role = "target")

    df_all_fls <- bind_rows(df_reference_expanded, df_target_fls)
    if (nrow(df_all_fls) == 0) {
      stop("No real-biomedical contrasts remain after fallback filtering")
    }

    df_all_fls$domain <- factor(
      df_all_fls$domain,
      levels = c("biomedical", "general")
    )
    df_all_fls$language <- factor(df_all_fls$language)

    feature_cols <- df_all_fls %>%
      select(
        -any_of(ID_COLS),
        -any_of(c(
          "language_file", "source_batch", "forced_prefix_fallback",
          "source_origin", "contrast_group", "feature_family", "comparison_role"
        ))
      ) %>%
      select(where(is.numeric)) %>%
      names()
    if (length(feature_cols) == 0) {
      stop("No numeric feature columns detected for real-biomedical contrast")
    }

    save_rds_obj(input_manifest, "input_manifest.rds", contrast_output_dir)
    save_rds_obj(target_registry, "model_registry.rds", contrast_output_dir)
    save_rds_obj(df_reference, "df_real_biomedical_reference.rds", contrast_output_dir)
    save_rds_obj(df_target, "df_biomedical_targets.rds", contrast_output_dir)
    save_rds_obj(df_all_fls, "df_all_fls.rds", contrast_output_dir)
    save_rds_obj(feature_cols, "feature_cols.rds", contrast_output_dir)
    save_rds_obj(
      tibble(
        contrast_id = "real_biomedical_vs_biomedical",
        internal_reference_level = "general",
        reference_label = "real biomedical",
        internal_target_level = "biomedical",
        target_label = "biomedical",
        reference_directory = REAL_BIOMEDICAL_DIR,
        upper_bound_included_as_target = TRUE
      ),
      "contrast_definition.rds",
      contrast_output_dir
    )

    fallback_filter_summary <- df_target %>%
      group_by(model_id, model_label) %>%
      summarise(
        n_all = n(),
        n_false = sum(forced_prefix_fallback == FALSE, na.rm = TRUE),
        n_true = sum(forced_prefix_fallback == TRUE, na.rm = TRUE),
        n_na = sum(is.na(forced_prefix_fallback)),
        n_kept = sum(is.na(forced_prefix_fallback) | forced_prefix_fallback == FALSE),
        prop_kept = n_kept / n_all,
        .groups = "drop"
      )
    save_rds_obj(fallback_filter_summary, "fallback_filter_summary.rds", contrast_output_dir)

    contrast_counts <- df_all_fls %>%
      count(model_id, model_label, language, domain, comparison_role,
            contrast_group, name = "n") %>%
      arrange(model_id, language, domain)
    save_rds_obj(contrast_counts, "contrast_counts.rds", contrast_output_dir)

    log_msg("Real-biomedical contrast: finding degenerate features")
    deg <- find_degenerate_lang_features(
      df = df_all_fls,
      feats = feature_cols,
      min_n = MIN_N_PER_DOMAIN,
      eps = EPS_SD
    )
    save_rds_obj(deg, "degenerate_feature_analysis.rds", contrast_output_dir)
    save_rds_obj(deg$stats, "degenerate_feature_stats.rds", contrast_output_dir)
    save_rds_obj(deg$bad, "bad_lang_feat.rds", contrast_output_dir)

    feats_ok <- df_all_fls %>%
      distinct(model_id, language) %>%
      crossing(feature = feature_cols) %>%
      anti_join(deg$bad, by = c("model_id", "language", "feature"))
    save_rds_obj(feats_ok, "feats_ok.rds", contrast_output_dir)

    df_clean <- df_all_fls %>%
      select(
        any_of(c(
          ID_COLS, "language_file", "source_batch", "source_origin",
          "contrast_group", "feature_family", "comparison_role",
          "domain", "language", "forced_prefix_fallback"
        )),
        all_of(feature_cols)
      )
    save_rds_obj(df_clean, "df_clean.rds", contrast_output_dir)

    log_msg("Real-biomedical contrast: z-scoring by model and language")
    df_z <- zscore_by_model_language(df_clean, feature_cols, eps = EPS_SD)
    save_rds_obj(df_z, "df_z.rds", contrast_output_dir)

    scoring_input_summary <- df_z %>%
      count(model_id, model_label, language, domain, name = "n_rows") %>%
      mutate(domain = as.character(domain)) %>%
      pivot_wider(
        names_from = domain,
        values_from = n_rows,
        values_fill = 0,
        names_prefix = "n_internal_"
      ) %>%
      left_join(
        feats_ok %>% count(model_id, language, name = "n_candidate_features"),
        by = c("model_id", "language")
      ) %>%
      mutate(n_candidate_features = replace_na(n_candidate_features, 0L)) %>%
      arrange(model_id, language)
    save_rds_obj(scoring_input_summary, "scoring_input_summary.rds", contrast_output_dir)
    print(scoring_input_summary)

    log_msg("Real-biomedical contrast: per-language feature scoring")
    per_lang_scores <- score_features_per_language(
      df_z = df_z,
      feats_by_lang = feats_ok,
      metrics = SCORE_METRICS,
      B = SCORE_B,
      seed = BASE_SEED + 1000000L,
      min_n = MIN_N_PER_DOMAIN,
      jsd_grid_n = JSD_GRID_N,
      jsd_bw = JSD_BW,
      jsd_eps = JSD_EPS,
      eps_sd = EPS_SD,
      reference_label = "real biomedical",
      target_label = "biomedical"
    )
    save_rds_obj(per_lang_scores, "per_lang_scores.rds", contrast_output_dir)
    if (nrow(per_lang_scores) == 0) {
      stop("No features were scored for the real-biomedical contrast")
    }
  }

  if (BALANCED_STABILITY_ENABLED) {
    log_msg("Real-biomedical contrast: repeated equal-n stability analysis")
    balanced_stability <- run_balanced_stability(
      df_z = df_z,
      feats_by_lang = feats_ok,
      full_scores = per_lang_scores,
      reps = BALANCED_STABILITY_REPS,
      target_n = BALANCED_STABILITY_TARGET_N,
      conf_level = BALANCED_STABILITY_CONF_LEVEL,
      top_k = BALANCED_STABILITY_TOP_K,
      seed = BASE_SEED + 1100000L,
      min_n = MIN_N_PER_DOMAIN,
      metrics = SCORE_METRICS,
      jsd_grid_n = JSD_GRID_N,
      jsd_bw = JSD_BW,
      jsd_eps = JSD_EPS,
      eps_sd = EPS_SD
    )
    save_rds_obj(balanced_stability$config, "balanced_stability_config.rds", contrast_output_dir)
    save_rds_obj(balanced_stability$cell_counts, "balanced_stability_cell_counts.rds", contrast_output_dir)
    save_rds_obj(balanced_stability$draws, "balanced_stability_draws.rds", contrast_output_dir)
    save_rds_obj(balanced_stability$metric_summary, "balanced_stability_metric_summary.rds", contrast_output_dir)
    save_rds_obj(balanced_stability$feature_summary, "balanced_stability_feature_summary.rds", contrast_output_dir)
    save_rds_obj(balanced_stability$rank_correlations, "balanced_stability_rank_correlations.rds", contrast_output_dir)
    save_rds_obj(balanced_stability$rank_summary, "balanced_stability_rank_summary.rds", contrast_output_dir)
    write_csv(
      balanced_stability$feature_summary,
      file.path(contrast_output_dir, "balanced_stability_feature_summary.csv")
    )
    write_csv(
      balanced_stability$rank_summary,
      file.path(contrast_output_dir, "balanced_stability_rank_summary.csv")
    )
  }

  log_msg("Real-biomedical contrast: post-hoc feature clustering")
  per_lang_clusters <- cluster_features_per_language(
    df_z = df_z,
    per_lang_scores = per_lang_scores,
    q_max = CLUSTER_Q_MAX,
    top_n = CLUSTER_TOP_N,
    cutoff_corr = CLUSTER_CUTOFF_CORR,
    method = COR_METHOD,
    center_within_domain = USE_DOMAIN_CENTERING,
    cap_per_domain = MAX_N_PER_DOMAIN,
    seed = BASE_SEED + 1200000L,
    reference_label = "real biomedical",
    target_label = "biomedical"
  )
  save_rds_obj(per_lang_clusters, "per_lang_clusters.rds", contrast_output_dir)

  feature_fps <- fingerprint_per_language_feature(per_lang_scores, K = FINGERPRINT_K)
  cluster_fps <- fingerprint_per_language_cluster(per_lang_clusters, K = FINGERPRINT_K)
  n_feature_fps <- feature_fps %>%
    group_by(model_id, model_label, language) %>%
    summarise(n_feature_fps = n(), .groups = "drop")
  n_cluster_fps <- cluster_fps %>%
    group_by(model_id, model_label, language) %>%
    summarise(n_cluster_fps = n(), .groups = "drop")
  save_rds_obj(feature_fps, "feature_fps.rds", contrast_output_dir)
  save_rds_obj(cluster_fps, "cluster_fps.rds", contrast_output_dir)
  save_rds_obj(n_feature_fps, "n_feature_fps.rds", contrast_output_dir)
  save_rds_obj(n_cluster_fps, "n_cluster_fps.rds", contrast_output_dir)

  meta_clusters <- build_meta_clusters(
    per_lang_clusters = per_lang_clusters,
    jaccs = META_JACCS
  )
  stable_universal <- summarize_meta_clusters(
    meta_clusters = meta_clusters,
    alpha = META_ALPHA
  )
  pair_counts <- compute_pair_counts(
    stable_universal = stable_universal,
    jacc_threshold = PAIRCOUNT_JACC
  )
  pair_counts_full <- compute_pair_counts_full(
    stable_universal = stable_universal,
    pair_counts = pair_counts,
    jacc_threshold = PAIRCOUNT_JACC
  )
  zero_pairs <- compute_zero_pairs(pair_counts_full)
  lang_unique_counts <- compute_lang_unique_counts(
    stable_universal = stable_universal,
    jacc_threshold = PAIRCOUNT_JACC
  )
  save_rds_obj(meta_clusters, "meta_clusters.rds", contrast_output_dir)
  save_rds_obj(stable_universal, "stable_universal.rds", contrast_output_dir)
  save_rds_obj(pair_counts, "pair_counts.rds", contrast_output_dir)
  save_rds_obj(pair_counts_full, "pair_counts_full.rds", contrast_output_dir)
  save_rds_obj(zero_pairs, "zero_pairs.rds", contrast_output_dir)
  save_rds_obj(lang_unique_counts, "lang_unique_counts.rds", contrast_output_dir)

  save_rds_obj(
    tibble(
      model_id = if (nzchar(SELECTED_MODEL_ID)) SELECTED_MODEL_ID else NA_character_,
      contrast_id = "real_biomedical_vs_biomedical",
      completed_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
      resumed_from_scores = resume_ready,
      skipped = FALSE
    ),
    "postprocessing_complete.rds",
    contrast_output_dir
  )
  log_msg("Real-biomedical vs biomedical contrast done: ", contrast_output_dir)
  invisible(NULL)
}

# =========================================================
# MAIN
# =========================================================

main <- function() {
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

  analysis_config <- tibble(
    base_seed = BASE_SEED,
    permutations = SCORE_B,
    score_metrics = list(SCORE_METRICS),
    min_n_per_domain = MIN_N_PER_DOMAIN,
    jsd_grid_n = JSD_GRID_N,
    jsd_bandwidth = JSD_BW,
    balanced_stability_enabled = BALANCED_STABILITY_ENABLED,
    balanced_stability_repetitions = BALANCED_STABILITY_REPS,
    balanced_stability_target_n = BALANCED_STABILITY_TARGET_N,
    balanced_stability_confidence_level = BALANCED_STABILITY_CONF_LEVEL,
    balanced_stability_top_k = BALANCED_STABILITY_TOP_K,
    cluster_q_max = CLUSTER_Q_MAX,
    cluster_correlation_cutoff = CLUSTER_CUTOFF_CORR,
    meta_jaccards = list(META_JACCS),
    real_biomedical_contrast_enabled = RUN_REAL_BIOMEDICAL_CONTRAST
  )
  save_rds_obj(analysis_config, "analysis_config.rds")
  save_rds_obj(sessionInfo(), "session_info.rds")

  checkpoint_paths <- c(
    df_z = file.path(OUTPUT_DIR, "df_z.rds"),
    feats_ok = file.path(OUTPUT_DIR, "feats_ok.rds"),
    per_lang_scores = file.path(OUTPUT_DIR, "per_lang_scores.rds")
  )
  prepared_paths <- c(
    df_z = checkpoint_paths[["df_z"]],
    feats_ok = checkpoint_paths[["feats_ok"]],
    df_clean = file.path(OUTPUT_DIR, "df_clean.rds")
  )
  prepared_ready <- RESUME_EXISTING && all(file.exists(prepared_paths))
  resume_ready <- RESUME_EXISTING && all(file.exists(checkpoint_paths))

  if (PREPARE_ONLY && prepared_ready) {
    validate_prepared_alignment(
      readRDS(prepared_paths[["df_z"]]),
      readRDS(prepared_paths[["feats_ok"]])
    )
    log_msg("Shared alignment checkpoints already exist: ", OUTPUT_DIR)
    return(invisible(NULL))
  }

  if (resume_ready) {
    log_msg("Resume mode: loading existing z-scores, feature set, and permutation scores")
    log_msg("Skipping raw CSV loading, z-scoring, and all 500-permutation scoring")
    log_msg("This univariate workflow does not consume random-forest outputs")

    df_z <- readRDS(checkpoint_paths[["df_z"]])
    feats_ok <- readRDS(checkpoint_paths[["feats_ok"]])
    per_lang_scores <- readRDS(checkpoint_paths[["per_lang_scores"]])

    required_df_z <- c("model_id", "model_label", "language", "domain")
    required_feats <- c("model_id", "language", "feature")
    required_scores <- c(
      "model_id", "model_label", "language", "feature",
      "cohens_d", "w1_z", "jsd", "borda_score", "p_perm", "q_perm",
      "delta_z", "direction"
    )

    if (!all(required_df_z %in% names(df_z))) {
      stop("Resume checkpoint df_z.rds is missing: ", paste(setdiff(required_df_z, names(df_z)), collapse = ", "))
    }
    if (!all(required_feats %in% names(feats_ok))) {
      stop("Resume checkpoint feats_ok.rds is missing: ", paste(setdiff(required_feats, names(feats_ok)), collapse = ", "))
    }
    if (!all(required_scores %in% names(per_lang_scores)) || nrow(per_lang_scores) == 0) {
      stop("Resume checkpoint per_lang_scores.rds is empty or missing required columns")
    }

    if (nzchar(SELECTED_MODEL_ID)) {
      checkpoint_models <- unique(c(
        as.character(df_z$model_id),
        as.character(feats_ok$model_id),
        as.character(per_lang_scores$model_id)
      ))
      checkpoint_models <- checkpoint_models[!is.na(checkpoint_models)]
      if (!identical(sort(checkpoint_models), SELECTED_MODEL_ID)) {
        stop(
          "Resume checkpoints do not belong exclusively to selected model '",
          SELECTED_MODEL_ID, "'. Found: ", paste(sort(checkpoint_models), collapse = ", ")
        )
      }
    }
  } else {
    if (prepared_ready) {
      log_msg("Loading shared data-preparation checkpoints from ", OUTPUT_DIR)
      df_z <- readRDS(prepared_paths[["df_z"]])
      feats_ok <- readRDS(prepared_paths[["feats_ok"]])
      validate_prepared_alignment(df_z, feats_ok)
    } else {
      if (RESUME_EXISTING) {
        missing_prepared <- names(prepared_paths)[!file.exists(prepared_paths)]
        log_msg(
          "Shared checkpoints incomplete; preparing raw inputs. Missing: ",
          paste(missing_prepared, collapse = ", ")
        )
      } else {
        log_msg("Resume disabled; preparing raw inputs")
      }

  is_real_reference_run <- identical(SELECTED_MODEL_ID, "real")
  log_msg("Discovering biomedical model files")
  manifest <- if (is_real_reference_run) {
    discover_real_biomedical_manifest(
      input_dir = REAL_BIOMEDICAL_DIR,
      pattern = BIOMEDICAL_INPUT_PATTERN,
      iso3_to_iso2 = ISO3_TO_ISO2
    )
  } else {
    discover_biomedical_manifest(
      root = BIOMEDICAL_OUTPUTS_DIR,
      families = BIOMEDICAL_FAMILIES,
      pattern = BIOMEDICAL_INPUT_PATTERN,
      vllm_run_date = VLLM_RUN_DATE,
      iso3_to_iso2 = ISO3_TO_ISO2
    )
  }

  if (nzchar(SELECTED_MODEL_ID)) {
    available_models <- sort(unique(manifest$model_id))
    manifest <- manifest %>% filter(model_id == SELECTED_MODEL_ID)
    if (nrow(manifest) == 0) {
      stop(
        "Requested model '", SELECTED_MODEL_ID, "' was not discovered. Available: ",
        paste(available_models, collapse = ", ")
      )
    }
    log_msg("Array task selected model: ", SELECTED_MODEL_ID)
    log_msg("Model-specific output directory: ", OUTPUT_DIR)
  }

  model_registry <- manifest %>%
    group_by(model_id, model_label, feature_family) %>%
    summarise(
      n_languages_in_filenames = n_distinct(language_from_file, na.rm = TRUE),
      languages_in_filenames = paste(
        sort(unique(na.omit(language_from_file))),
        collapse = ", "
      ),
      language_stored_in_records = any(is.na(language_from_file)),
      .groups = "drop"
    ) %>%
    arrange(model_id)

  df_biomedical <- if (is_real_reference_run) {
    read_real_biomedical_files(REAL_BIOMEDICAL_DIR) %>%
      add_language() %>%
      mutate(
        model_id = "real",
        model_label = "real",
        contrast_group = "biomedical__real",
        feature_family = "real_biomedical"
      )
  } else {
    read_manifest_files(manifest) %>% add_language()
  }
  df_real_general <- read_real_general_files(REAL_GENERAL_DIR) %>% add_language()

  if (!"forced_prefix_fallback" %in% names(df_biomedical)) {
    df_biomedical$forced_prefix_fallback <- NA_character_
  }
  if (!"forced_prefix_fallback" %in% names(df_real_general)) {
    df_real_general$forced_prefix_fallback <- NA_character_
  }

  df_biomedical <- df_biomedical %>%
    mutate(forced_prefix_fallback = normalize_false_flag(forced_prefix_fallback))
  df_real_general <- df_real_general %>%
    mutate(forced_prefix_fallback = normalize_false_flag(forced_prefix_fallback))

  # A missing fallback flag is legitimate for baselines such as upper_bound and
  # unigram. Only explicit TRUE rows are removed.
  df_biomedical_fls <- df_biomedical %>%
    filter(is.na(forced_prefix_fallback) | forced_prefix_fallback == FALSE)

  valid_model_languages <- df_biomedical_fls %>%
    distinct(model_id, model_label, language)

  df_real_expanded <- df_real_general %>%
    select(-model_id, -model_label) %>%
    inner_join(valid_model_languages, by = "language") %>%
    mutate(contrast_group = "real_general")

  df_all <- bind_rows(df_real_expanded, df_biomedical)
  df_all_fls <- bind_rows(df_real_expanded, df_biomedical_fls)

  save_rds_obj(manifest, "input_manifest.rds")
  save_rds_obj(model_registry, "model_registry.rds")
  save_rds_obj(df_real_general, "df_real_general.rds")
  save_rds_obj(df_biomedical, "df_biomedical_models.rds")
  save_rds_obj(df_biomedical_fls, "df_biomedical_models_fls.rds")
  
  if (nrow(df_all_fls) == 0) {
    stop("No model/language contrasts remain after input filtering.")
  }
  
  df_all$domain <- factor(df_all$domain, levels = c("biomedical", "general"))
  df_all$language <- factor(df_all$language)
  
  df_all_fls$domain <- factor(df_all_fls$domain, levels = c("biomedical", "general"))
  df_all_fls$language <- factor(df_all_fls$language)
  
  feature_cols <- df_all_fls %>%
    select(
      -any_of(ID_COLS),
      -any_of(c(
        "language_file", "source_batch", "forced_prefix_fallback", "source_origin",
        "contrast_group", "feature_family"
      ))
    ) %>%
    select(where(is.numeric)) %>%
    names()
  
  if (length(feature_cols) == 0) stop("No numeric feature columns detected.")
  
  save_rds_obj(feature_cols, "feature_cols.rds")
  save_rds_obj(df_all, "df_all.rds")
  save_rds_obj(df_all_fls, "df_all_fls.rds")
  
  fallback_filter_summary <- df_biomedical %>%
    group_by(model_id, model_label) %>%
    summarise(
      n_all = n(),
      n_false = sum(forced_prefix_fallback == FALSE, na.rm = TRUE),
      n_true = sum(forced_prefix_fallback == TRUE, na.rm = TRUE),
      n_na = sum(is.na(forced_prefix_fallback)),
      n_kept = sum(is.na(forced_prefix_fallback) | forced_prefix_fallback == FALSE),
      prop_kept = n_kept / n_all,
      .groups = "drop"
    )
  
  save_rds_obj(fallback_filter_summary, "fallback_filter_summary.rds")
  
  contrast_counts <- df_all_fls %>%
    count(model_id, model_label, language, domain, contrast_group, name = "n") %>%
    arrange(model_id, language, domain)
  save_rds_obj(contrast_counts, "contrast_counts.rds")
  
  log_msg("Finding degenerate features")
  deg <- find_degenerate_lang_features(
    df = df_all_fls,
    feats = feature_cols,
    min_n = MIN_N_PER_DOMAIN,
    eps = EPS_SD
  )
  save_rds_obj(deg, "degenerate_feature_analysis.rds")
  save_rds_obj(deg$stats, "degenerate_feature_stats.rds")
  save_rds_obj(deg$bad, "bad_lang_feat.rds")
  
  log_msg("Building feats_ok")
  feats_ok <- df_all_fls %>%
    distinct(model_id, language) %>%
    crossing(feature = feature_cols) %>%
    anti_join(deg$bad, by = c("model_id", "language", "feature"))
  save_rds_obj(feats_ok, "feats_ok.rds")
  
  # Optional collinearity pruning. The univariate specification leaves this
  # step disabled; uncomment the block only for a separately defined analysis.
  # log_msg("Pruning collinear features")
  # colp <- prune_collinear_per_language(
  #   df = df_all_fls,
  #   feats_by_lang = feats_ok,
  #   cutoff = CUTOFF_CORR,
  #   method = COR_METHOD,
  #   center_within_domain = USE_DOMAIN_CENTERING,
  #   cap_per_domain = MAX_N_PER_DOMAIN,
  #   seed = BASE_SEED
  # )
  # save_rds_obj(colp, "collinearity_pruning.rds")
  # save_rds_obj(colp$drops, "collinearity_drops.rds")
  # save_rds_obj(colp$reps, "collinearity_representatives.rds")
  #
  # feats_pruned <- feats_ok %>%
  #   anti_join(colp$drops, by = c("model_id", "language", "feature"))
  # save_rds_obj(feats_pruned, "feats_pruned.rds")
  
  df_clean <- df_all_fls %>%
    select(any_of(c(
      ID_COLS,
      "language_file", "source_batch", "source_origin", "contrast_group",
      "feature_family", "domain", "language", "forced_prefix_fallback"
    )), all_of(feature_cols))
  save_rds_obj(df_clean, "df_clean.rds")
  
  log_msg("Z-scoring by model and language")
  df_z <- zscore_by_model_language(df_clean, feature_cols, eps = EPS_SD)
  save_rds_obj(df_z, "df_z.rds")
    }

  if (PREPARE_ONLY) {
    save_rds_obj(
      tibble(
        model_id = if (nzchar(SELECTED_MODEL_ID)) SELECTED_MODEL_ID else NA_character_,
        completed_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
      ),
      "common_preparation_complete.rds"
    )
    log_msg("Shared alignment data preparation complete: ", OUTPUT_DIR)
    return(invisible(NULL))
  }
  
  log_msg("Per-language feature scoring")
  scoring_input_summary <- df_z %>%
    count(model_id, model_label, language, domain, name = "n_rows") %>%
    mutate(domain = as.character(domain)) %>%
    pivot_wider(
      names_from = domain,
      values_from = n_rows,
      values_fill = 0,
      names_prefix = "n_"
    ) %>%
    left_join(
      feats_ok %>% count(model_id, language, name = "n_candidate_features"),
      by = c("model_id", "language")
    ) %>%
    mutate(n_candidate_features = replace_na(n_candidate_features, 0L)) %>%
    arrange(model_id, language)
  save_rds_obj(scoring_input_summary, "scoring_input_summary.rds")
  print(scoring_input_summary)

  per_lang_scores <- score_features_per_language(
    df_z = df_z,
    feats_by_lang = feats_ok,
    metrics = SCORE_METRICS,
    B = SCORE_B,
    seed = BASE_SEED,
    min_n = MIN_N_PER_DOMAIN,
    jsd_grid_n = JSD_GRID_N,
    jsd_bw = JSD_BW,
    jsd_eps = JSD_EPS,
    eps_sd = EPS_SD
  )
  save_rds_obj(per_lang_scores, "per_lang_scores.rds")

  if (nrow(per_lang_scores) == 0) {
    stop(
      "No features were scored for any model/language. See ",
      file.path(OUTPUT_DIR, "scoring_input_summary.rds"),
      " and the candidates/domain counts printed above."
    )
  }
  }

  if (BALANCED_STABILITY_ENABLED) {
    if (nzchar(SELECTED_MODEL_ID) && is.na(BALANCED_STABILITY_TARGET_N)) {
      warning(
        "Balanced target n is being inferred inside a model-specific array task. ",
        "This balances domains/languages for the selected model but not across models. ",
        "For a shared cross-model target, set ALIGNMENT_BALANCED_TARGET_N to the ",
        "global minimum and use the same value for every array task."
      )
    }

    log_msg("Running repeated equal-n stability analysis")
    balanced_stability <- run_balanced_stability(
      df_z = df_z,
      feats_by_lang = feats_ok,
      full_scores = per_lang_scores,
      reps = BALANCED_STABILITY_REPS,
      target_n = BALANCED_STABILITY_TARGET_N,
      conf_level = BALANCED_STABILITY_CONF_LEVEL,
      top_k = BALANCED_STABILITY_TOP_K,
      seed = BASE_SEED + 100000L,
      min_n = MIN_N_PER_DOMAIN,
      metrics = SCORE_METRICS,
      jsd_grid_n = JSD_GRID_N,
      jsd_bw = JSD_BW,
      jsd_eps = JSD_EPS,
      eps_sd = EPS_SD
    )

    save_rds_obj(balanced_stability$config, "balanced_stability_config.rds")
    save_rds_obj(balanced_stability$cell_counts, "balanced_stability_cell_counts.rds")
    save_rds_obj(balanced_stability$draws, "balanced_stability_draws.rds")
    save_rds_obj(balanced_stability$metric_summary, "balanced_stability_metric_summary.rds")
    save_rds_obj(balanced_stability$feature_summary, "balanced_stability_feature_summary.rds")
    save_rds_obj(balanced_stability$rank_correlations, "balanced_stability_rank_correlations.rds")
    save_rds_obj(balanced_stability$rank_summary, "balanced_stability_rank_summary.rds")

    write_csv(
      balanced_stability$feature_summary,
      file.path(OUTPUT_DIR, "balanced_stability_feature_summary.csv")
    )
    write_csv(
      balanced_stability$rank_summary,
      file.path(OUTPUT_DIR, "balanced_stability_rank_summary.csv")
    )
    print(balanced_stability$config)
    print(balanced_stability$rank_summary)
  } else {
    log_msg("Balanced stability analysis disabled")
  }
  
  log_msg("Post-hoc feature clustering")
  per_lang_clusters <- cluster_features_per_language(
    df_z = df_z,
    per_lang_scores = per_lang_scores,
    q_max = CLUSTER_Q_MAX,
    top_n = CLUSTER_TOP_N,
    cutoff_corr = CLUSTER_CUTOFF_CORR,
    method = COR_METHOD,
    center_within_domain = USE_DOMAIN_CENTERING,
    cap_per_domain = MAX_N_PER_DOMAIN,
    seed = BASE_SEED
  )
  save_rds_obj(per_lang_clusters, "per_lang_clusters.rds")
  
  log_msg("Fingerprints")
  feature_fps <- fingerprint_per_language_feature(per_lang_scores, K = FINGERPRINT_K)
  cluster_fps <- fingerprint_per_language_cluster(per_lang_clusters, K = FINGERPRINT_K)
  
  n_feature_fps <- feature_fps %>%
    group_by(model_id, model_label, language) %>%
    summarise(n_feature_fps = n(), .groups = "drop")
  
  n_cluster_fps <- cluster_fps %>%
    group_by(model_id, model_label, language) %>%
    summarise(n_cluster_fps = n(), .groups = "drop")
  
  save_rds_obj(feature_fps, "feature_fps.rds")
  save_rds_obj(cluster_fps, "cluster_fps.rds")
  save_rds_obj(n_feature_fps, "n_feature_fps.rds")
  save_rds_obj(n_cluster_fps, "n_cluster_fps.rds")
  
  log_msg("Building meta-clusters")
  meta_clusters <- build_meta_clusters(
    per_lang_clusters = per_lang_clusters,
    jaccs = META_JACCS
  )
  save_rds_obj(meta_clusters, "meta_clusters.rds")
  
  stable_universal <- summarize_meta_clusters(
    meta_clusters = meta_clusters,
    alpha = META_ALPHA
  )
  save_rds_obj(stable_universal, "stable_universal.rds")
  
  pair_counts <- compute_pair_counts(
    stable_universal = stable_universal,
    jacc_threshold = PAIRCOUNT_JACC
  )
  save_rds_obj(pair_counts, "pair_counts.rds")
  
  pair_counts_full <- compute_pair_counts_full(
    stable_universal = stable_universal,
    pair_counts = pair_counts,
    jacc_threshold = PAIRCOUNT_JACC
  )
  save_rds_obj(pair_counts_full, "pair_counts_full.rds")
  
  zero_pairs <- compute_zero_pairs(pair_counts_full)
  save_rds_obj(zero_pairs, "zero_pairs.rds")
  
  lang_unique_counts <- compute_lang_unique_counts(
    stable_universal = stable_universal,
    jacc_threshold = PAIRCOUNT_JACC
  )
  save_rds_obj(lang_unique_counts, "lang_unique_counts.rds")

  save_rds_obj(
    tibble(
      model_id = if (nzchar(SELECTED_MODEL_ID)) SELECTED_MODEL_ID else NA_character_,
      completed_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
      resumed_from_scores = resume_ready
    ),
    "postprocessing_complete.rds"
  )

  if (RUN_REAL_BIOMEDICAL_CONTRAST) {
    run_real_biomedical_contrast()
  } else {
    log_msg(
      "Real-biomedical contrast disabled via ",
      "ALIGNMENT_REAL_BIOMEDICAL_CONTRAST"
    )
  }
  
  log_msg("Done")
}

print_usage <- function() {
  cat(
    "Usage: ALIGNMENT_MODEL_ID=<model> Rscript src/alignment/univariate.R\n",
    "       Rscript src/alignment/univariate.R --list-models\n",
    "\nConfiguration is supplied through ALIGNMENT_* environment variables.\n",
    "Shared checkpoints are prepared by src/alignment/common.R.\n",
    "See src/alignment/README.md for paths and method settings.\n",
    sep = ""
  )
}

if (sys.nframe() == 0L) {
  cli_args <- commandArgs(trailingOnly = TRUE)
  if (any(cli_args %in% c("--help", "-h"))) {
    print_usage()
  } else if ("--list-models" %in% cli_args) {
    cat(paste(discover_available_model_ids(), collapse = "\n"), "\n", sep = "")
  } else {
    main()
  }
}
