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

# =============================================================================
# POST-HOC CLUSTERING OF STABLE AND SIGNIFICANT FEATURES
#
# Required files in INPUT_DIR:
#   df_z.rds
#   per_lang_scores.rds
#   balanced_stability_feature_summary.rds
#
# The script does not read raw feature CSVs, z-score features, run permutations,
# or repeat the balanced stability analysis.
# =============================================================================

# ---- Paths ------------------------------------------------------------------

PROJECT_ROOT <- ALIGNMENT_PROJECT_ROOT
INPUT_ROOT <- alignment_env_value(
  "ALIGNMENT_UNIVARIATE_OUTPUT_DIR",
  file.path(PROJECT_ROOT, "outputs", "alignment", "univariate"),
  "DOMAIN_SHIFT_POSTHOC_INPUT_ROOT"
)
MODEL_ID <- alignment_env_value("ALIGNMENT_MODEL_ID", "", "DOMAIN_SHIFT_MODEL_ID")
alignment_validate_model_id(MODEL_ID)

# Model-specific checkpoints are stored in INPUT_ROOT/<MODEL_ID>. Leave the
# model id empty when the first stage was run as a combined analysis.
INPUT_DIR <- if (nzchar(MODEL_ID)) file.path(INPUT_ROOT, MODEL_ID) else INPUT_ROOT

# Stability-filtered clustering has its own output directory so both feature
# selection specifications remain available.
OUTPUT_DIR <- file.path(INPUT_DIR, "stable_significant_clusters")

# ---- Feature selection -------------------------------------------------------

CONTRAST                 <- "real_general_vs_biomedical"
SIGNIFICANCE_Q_MAX       <- 0.05
MIN_ITERATION_COVERAGE   <- 0.95
MIN_DELTA_SIGN_AGREEMENT <- 0.95
MIN_TOP_K_RATE           <- 0.80

# A feature is retained when:
#   1. its full-data permutation q-value is significant;
#   2. it was estimable in at least MIN_ITERATION_COVERAGE of balanced runs;
#   3. its effect direction agreed with the full-data direction in at least
#      MIN_DELTA_SIGN_AGREEMENT of balanced runs; and
#   4. it appeared in the balanced top K in at least MIN_TOP_K_RATE of runs.
#
# Set MIN_TOP_K_RATE <- 0 if "stable" should concern only estimability and
# direction, without requiring stable high rank.

# ---- Clustering --------------------------------------------------------------

CLUSTER_CUTOFF_CORR  <- 0.80
COR_METHOD           <- "spearman"
USE_DOMAIN_CENTERING <- TRUE
MAX_N_PER_DOMAIN     <- 3000

# ---- Meta-clustering ---------------------------------------------------------

META_JACCS     <- c(0.2, 0.3, 0.4, 0.6, 0.8)
META_ALPHA     <- SIGNIFICANCE_Q_MAX
PAIRCOUNT_JACC <- 0.80

BASE_SEED <- 123

log_msg <- function(...) {
  message(sprintf(
    "[%s] %s",
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    paste0(..., collapse = "")
  ))
}

save_output <- function(x, filename) {
  saveRDS(x, file.path(OUTPUT_DIR, filename))
}

classify_feature_stability <- function(per_lang_scores,
                                       stability_summary) {
  # A combined stability table identifies its contrast explicitly. A table
  # stored inside a contrast-specific directory needs no additional filter.
  if ("contrast" %in% names(stability_summary)) {
    stability_summary <- stability_summary %>%
      filter(.data$contrast == .env$CONTRAST)
  }

  stability_summary %>%
    mutate(
      stable_coverage = is.finite(iteration_coverage) &
        iteration_coverage >= MIN_ITERATION_COVERAGE,
      stable_direction = is.finite(delta_sign_agreement) &
        delta_sign_agreement >= MIN_DELTA_SIGN_AGREEMENT,
      stable_top_k = is.finite(top_k_rate) &
        top_k_rate >= MIN_TOP_K_RATE,
      stable_all = stable_coverage & stable_direction & stable_top_k,
      full_significant = is.finite(full_q_perm) &
        full_q_perm <= SIGNIFICANCE_Q_MAX,
      stability_status = case_when(
        !stable_coverage ~ "low iteration coverage",
        !stable_direction ~ "unstable direction",
        !stable_top_k ~ "unstable ranking",
        stable_all ~ "stable",
        TRUE ~ "not classified"
      )
    ) %>%
    select(
      any_of("contrast"),
      model_id, model_label, language, feature,
      n_iterations, iteration_coverage, median_borda_rank,
      median_borda_score, top_k_rate, delta_sign_agreement,
      full_q_perm, stable_coverage, stable_direction, stable_top_k,
      stable_all, full_significant, stability_status
    ) %>%
    distinct() %>%
    inner_join(
      per_lang_scores,
      by = c("model_id", "model_label", "language", "feature")
    ) %>%
    arrange(model_id, language, q_perm, desc(borda_score), feature)
}

cluster_selected_features <- function(df_z, selected_scores,
                                      cutoff_corr = 0.80,
                                      method = "spearman",
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

  if (nrow(selected_scores) == 0) return(empty_clusters)

  keys <- selected_scores %>%
    distinct(model_id, model_label, language) %>%
    arrange(model_id, language)
  clusters_all <- list()

  for (i in seq_len(nrow(keys))) {
    model_key <- keys$model_id[[i]]
    model_label_key <- keys$model_label[[i]]
    lang <- keys$language[[i]]
    log_msg("Clustering stable/significant features for ",
            model_key, "/", as.character(lang))

    sc <- selected_scores %>%
      filter(.data$model_id == .env$model_key,
             as.character(.data$language) == as.character(.env$lang))
    feats <- unique(sc$feature)
    if (length(feats) == 0) next

    missing_from_z <- setdiff(feats, names(df_z))
    if (length(missing_from_z) > 0) {
      stop("Selected features missing from df_z for ", model_key, "/", lang,
           ": ", paste(missing_from_z, collapse = ", "))
    }

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
          if (nrow(.x) <= cap_per_domain) {
            .x
          } else {
            .x[sample.int(nrow(.x), cap_per_domain), , drop = FALSE]
          }
        }) %>%
        ungroup()

      if (center_within_domain) {
        d <- d %>%
          group_by(domain) %>%
          mutate(across(
            all_of(feats),
            ~ . - mean(., na.rm = TRUE)
          )) %>%
          ungroup()
      }

      correlation_matrix <- suppressWarnings(cor(
        d %>% select(-domain) %>% as.matrix(),
        use = "pairwise.complete.obs",
        method = method
      ))
      idx <- which(
        abs(correlation_matrix) >= cutoff_corr &
          upper.tri(correlation_matrix),
        arr.ind = TRUE
      )

      if (nrow(idx) > 0) {
        edges <- tibble(
          f1 = colnames(correlation_matrix)[idx[, 1]],
          f2 = colnames(correlation_matrix)[idx[, 2]]
        )
      }
    }

    comp_sc <- alignment_components(nodes = feats, edges_df = edges) %>%
      mutate(
        model_id = model_key,
        model_label = model_label_key,
        language = as.character(lang),
        cluster_id = paste0(as.character(lang), "__", comp)
      ) %>%
      select(model_id, model_label, language, cluster_id, feature) %>%
      left_join(
        sc %>% select(feature, borda_score, q_perm, delta_z, direction),
        by = "feature"
      )

    winners <- comp_sc %>%
      group_by(model_id, model_label, language, cluster_id) %>%
      arrange(q_perm, desc(borda_score), feature, .by_group = TRUE) %>%
      slice(1) %>%
      ungroup() %>%
      transmute(
        model_id, model_label, language, cluster_id,
        winner_feature = feature,
        winner_score = borda_score,
        winner_q = q_perm,
        winner_direction = direction
      )

    cluster_direction <- comp_sc %>%
      group_by(model_id, model_label, language, cluster_id) %>%
      summarise(
        n_members = n_distinct(feature),
        n_non_na = sum(is.finite(delta_z)),
        n_pos = sum(is.finite(delta_z) & delta_z > 0),
        n_neg = sum(is.finite(delta_z) & delta_z < 0),
        n_zero = sum(is.finite(delta_z) & delta_z == 0),
        cluster_direction = case_when(
          n_non_na == 0 ~ NA_character_,
          n_pos >= n_neg & n_pos >= n_zero ~
            paste0(reference_label, " > ", model_label_key, " ", target_label),
          n_neg >= n_pos & n_neg >= n_zero ~
            paste0(reference_label, " < ", model_label_key, " ", target_label),
          TRUE ~ "no change"
        ),
        majority_n = pmax(n_pos, n_neg, n_zero),
        frac_majority = ifelse(
          n_non_na > 0, majority_n / n_non_na, NA_real_
        ),
        n_deviant = ifelse(
          n_non_na > 0, n_non_na - majority_n, NA_integer_
        ),
        has_deviants = n_deviant > 0,
        is_unanimous = ifelse(
          n_non_na > 0, majority_n == n_non_na, NA
        ),
        cluster_delta_z = if (n_non_na > 0) {
          median(delta_z[is.finite(delta_z)])
        } else {
          NA_real_
        },
        members = paste(sort(unique(feature)), collapse = ", "),
        .groups = "drop"
      ) %>%
      select(-majority_n)

    clusters_all[[paste(model_key, lang, sep = "||")]] <- winners %>%
      left_join(
        cluster_direction,
        by = c("model_id", "model_label", "language", "cluster_id")
      )
  }

  result <- bind_rows(clusters_all)
  if (nrow(result) == 0) return(empty_clusters)
  result %>% arrange(model_id, language, winner_q, desc(winner_score))
}

build_meta_clusters <- function(per_lang_clusters, jaccs) {
  cluster_sets <- per_lang_clusters %>%
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

  if (nrow(cluster_sets) == 0) return(tibble())

  jaccard <- function(a, b) {
    union_ab <- union(a, b)
    if (length(union_ab) == 0) return(0)
    length(intersect(a, b)) / length(union_ab)
  }

  output <- list()
  for (model_key in sort(unique(cluster_sets$model_id))) {
    model_clusters <- cluster_sets %>%
      filter(.data$model_id == .env$model_key)
    member_sets <- strsplit(model_clusters$members, "\\s*,\\s*")
    names(member_sets) <- model_clusters$cluster_key

    for (jacc in jaccs) {
      log_msg("Building meta-clusters for ", model_key,
              " at Jaccard threshold ", jacc)

      keys <- model_clusters$cluster_key
      n_keys <- length(keys)
      edges <- list()
      edge_number <- 0L

      if (n_keys >= 2) {
        for (i in seq_len(n_keys - 1L)) {
          for (j in seq.int(i + 1L, n_keys)) {
            if (model_clusters$language[j] ==
                model_clusters$language[i]) next

            similarity <- jaccard(
              member_sets[[keys[i]]],
              member_sets[[keys[j]]]
            )
            if (is.finite(similarity) && similarity >= jacc) {
              edge_number <- edge_number + 1L
              edges[[edge_number]] <- c(keys[i], keys[j])
            }
          }
        }
      }

      edges_df <- if (edge_number == 0L) {
        tibble(f1 = character(), f2 = character())
      } else {
        tibble(
          f1 = vapply(edges, `[[`, character(1), 1),
          f2 = vapply(edges, `[[`, character(1), 2)
        )
      }

      components <- alignment_components(keys, edges_df) %>%
        mutate(meta_cluster_id = paste0("jacc", jacc, "__", comp))

      output[[paste(model_key, jacc, sep = "||")]] <- model_clusters %>%
        left_join(
          components %>% select(feature, comp, meta_cluster_id),
          by = c("cluster_key" = "feature")
        ) %>%
        select(
          model_id, model_label, meta_cluster_id, comp, language, cluster_id,
          winner_feature, winner_score, winner_q, direction, members
        ) %>%
        mutate(jacc = jacc) %>%
        group_by(model_id, jacc, meta_cluster_id) %>%
        mutate(
          meta_label = {
            counts <- sort(table(winner_feature), decreasing = TRUE)
            names(counts)[1]
          },
          meta_cluster_id2 = paste0("jacc", jacc, "__", meta_label)
        ) %>%
        ungroup()
    }
  }

  bind_rows(output)
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
        directions <- na.omit(direction)
        if (length(directions) == 0) {
          NA_character_
        } else {
          names(sort(table(directions), decreasing = TRUE))[1]
        }
      },
      dir_agree = {
        directions <- na.omit(direction)
        if (length(directions) == 0) {
          NA_real_
        } else {
          max(table(directions)) / length(directions)
        }
      },
      n_unique_winners = n_distinct(winner_feature),
      winners_by_lang = paste(
        paste0(language, ":", winner_feature),
        collapse = "; "
      ),
      .groups = "drop"
    ) %>%
    arrange(model_id, jacc, desc(n_langs), min_q, desc(dir_agree))
}

compute_pair_counts <- function(meta_summary, jacc_threshold = 0.8) {
  if (nrow(meta_summary) == 0) {
    return(tibble(
      model_id = character(), model_label = character(),
      l1 = character(), l2 = character(),
      n_shared_meta_clusters = integer()
    ))
  }

  meta_summary %>%
    filter(jacc >= jacc_threshold) %>%
    distinct(model_id, model_label, meta_cluster_id, langs) %>%
    mutate(
      langs_vec = str_split(langs, "\\s*,\\s*"),
      pairs = map(langs_vec, ~ {
        languages <- sort(unique(.x))
        if (length(languages) < 2) {
          return(tibble(l1 = character(), l2 = character()))
        }
        combinations <- combn(languages, 2)
        tibble(l1 = combinations[1, ], l2 = combinations[2, ])
      })
    ) %>%
    select(model_id, model_label, meta_cluster_id, pairs) %>%
    unnest(pairs) %>%
    count(model_id, model_label, l1, l2,
          name = "n_shared_meta_clusters") %>%
    arrange(model_id, desc(n_shared_meta_clusters))
}

compute_pair_counts_full <- function(meta_summary, pair_counts,
                                     jacc_threshold = 0.8) {
  if (nrow(meta_summary) == 0) return(pair_counts)

  all_pairs <- meta_summary %>%
    filter(jacc >= jacc_threshold) %>%
    transmute(
      model_id,
      model_label,
      language = str_split(langs, "\\s*,\\s*")
    ) %>%
    unnest(language) %>%
    distinct() %>%
    group_by(model_id, model_label) %>%
    summarise(languages = list(sort(unique(language))), .groups = "drop") %>%
    mutate(pairs = map(languages, ~ {
      if (length(.x) < 2) {
        return(tibble(l1 = character(), l2 = character()))
      }
      combinations <- combn(.x, 2)
      tibble(l1 = combinations[1, ], l2 = combinations[2, ])
    })) %>%
    select(-languages) %>%
    unnest(pairs)

  all_pairs %>%
    left_join(
      pair_counts,
      by = c("model_id", "model_label", "l1", "l2")
    ) %>%
    mutate(n_shared_meta_clusters = replace_na(
      n_shared_meta_clusters, 0L
    ))
}

compute_language_counts <- function(meta_summary, jacc_threshold = 0.8) {
  if (nrow(meta_summary) == 0) {
    return(tibble(
      model_id = character(), model_label = character(),
      language = character(), n_unique_meta_clusters = integer()
    ))
  }

  meta_summary %>%
    filter(jacc >= jacc_threshold, n_langs > 1) %>%
    select(model_id, model_label, meta_cluster_id, langs) %>%
    distinct() %>%
    separate_rows(langs, sep = ",\\s*") %>%
    distinct(model_id, model_label, meta_cluster_id, langs) %>%
    count(model_id, model_label, langs,
          name = "n_unique_meta_clusters") %>%
    rename(language = langs)
}

main <- function() {
  input_paths <- c(
    df_z = file.path(INPUT_DIR, "df_z.rds"),
    scores = file.path(INPUT_DIR, "per_lang_scores.rds"),
    stability = file.path(
      INPUT_DIR, "balanced_stability_feature_summary.rds"
    )
  )

  missing_files <- input_paths[!file.exists(input_paths)]
  if (length(missing_files) > 0) {
    stop("Missing required input files: ",
         paste(unname(missing_files), collapse = ", "))
  }

  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  log_msg("Loading existing post-hoc inputs from ", INPUT_DIR)
  df_z <- readRDS(input_paths[["df_z"]])
  per_lang_scores <- readRDS(input_paths[["scores"]])
  stability_summary <- readRDS(input_paths[["stability"]])

  alignment_require_columns(
    df_z,
    c("model_id", "model_label", "language", "domain"),
    "df_z.rds"
  )
  alignment_require_columns(
    per_lang_scores,
    c(
      "model_id", "model_label", "language", "feature",
      "borda_score", "q_perm", "delta_z", "direction"
    ),
    "per_lang_scores.rds"
  )
  alignment_require_columns(
    stability_summary,
    c(
      "model_id", "model_label", "language", "feature",
      "n_iterations", "iteration_coverage", "median_borda_rank",
      "median_borda_score", "top_k_rate", "delta_sign_agreement",
      "full_q_perm"
    ),
    "balanced_stability_feature_summary.rds"
  )

  log_msg("Classifying feature stability and significance")
  feature_stability <- classify_feature_stability(
    per_lang_scores,
    stability_summary
  )
  selected_scores <- feature_stability %>%
    filter(stable_all, full_significant)

  if (nrow(selected_scores) == 0) {
    stop(
      "No features passed the stable-and-significant criteria. ",
      "Review the thresholds near the top of this script."
    )
  }

  selection_counts <- selected_scores %>%
    count(model_id, model_label, language,
          name = "n_stable_significant_features") %>%
    arrange(model_id, language)

  save_output(feature_stability, "feature_stability.rds")
  write_csv(
    feature_stability,
    file.path(OUTPUT_DIR, "feature_stability.csv")
  )
  save_output(selected_scores, "stable_significant_scores.rds")
  save_output(selection_counts, "stable_significant_feature_counts.rds")
  write_csv(
    selected_scores,
    file.path(OUTPUT_DIR, "stable_significant_scores.csv")
  )
  write_csv(
    selection_counts,
    file.path(OUTPUT_DIR, "stable_significant_feature_counts.csv")
  )
  print(selection_counts)

  clusters <- cluster_selected_features(
    df_z = df_z,
    selected_scores = selected_scores,
    cutoff_corr = CLUSTER_CUTOFF_CORR,
    method = COR_METHOD,
    center_within_domain = USE_DOMAIN_CENTERING,
    cap_per_domain = MAX_N_PER_DOMAIN,
    seed = BASE_SEED
  )
  save_output(clusters, "per_lang_clusters.rds")
  write_csv(clusters, file.path(OUTPUT_DIR, "per_lang_clusters.csv"))

  meta_clusters <- build_meta_clusters(clusters, META_JACCS)
  save_output(meta_clusters, "meta_clusters.rds")
  write_csv(meta_clusters, file.path(OUTPUT_DIR, "meta_clusters.csv"))

  meta_summary <- summarize_meta_clusters(meta_clusters, META_ALPHA)
  save_output(meta_summary, "stable_universal.rds")
  write_csv(meta_summary, file.path(OUTPUT_DIR, "stable_universal.csv"))

  pair_counts <- compute_pair_counts(meta_summary, PAIRCOUNT_JACC)
  pair_counts_full <- compute_pair_counts_full(
    meta_summary, pair_counts, PAIRCOUNT_JACC
  )
  zero_pairs <- pair_counts_full %>%
    filter(n_shared_meta_clusters == 0L) %>%
    arrange(model_id, l1, l2)
  language_counts <- compute_language_counts(
    meta_summary, PAIRCOUNT_JACC
  )

  save_output(pair_counts, "pair_counts.rds")
  save_output(pair_counts_full, "pair_counts_full.rds")
  save_output(zero_pairs, "zero_pairs.rds")
  save_output(language_counts, "lang_unique_counts.rds")

  save_output(
    tibble(
      contrast = CONTRAST,
      significance_q_max = SIGNIFICANCE_Q_MAX,
      min_iteration_coverage = MIN_ITERATION_COVERAGE,
      min_delta_sign_agreement = MIN_DELTA_SIGN_AGREEMENT,
      min_top_k_rate = MIN_TOP_K_RATE,
      cluster_cutoff_corr = CLUSTER_CUTOFF_CORR,
      correlation_method = COR_METHOD,
      use_domain_centering = USE_DOMAIN_CENTERING,
      meta_jaccards = list(META_JACCS),
      paircount_jaccard = PAIRCOUNT_JACC,
      completed_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
    ),
    "posthoc_config.rds"
  )

  log_msg("Done. Results written to ", OUTPUT_DIR)
}

list_checkpoint_models <- function(input_root = INPUT_ROOT) {
  model_dirs <- list.dirs(
    input_root,
    full.names = TRUE,
    recursive = FALSE
  )
  required_files <- c(
    "df_z.rds",
    "per_lang_scores.rds",
    "balanced_stability_feature_summary.rds"
  )

  valid <- vapply(
    model_dirs,
    function(model_dir) {
      all(file.exists(file.path(model_dir, required_files)))
    },
    logical(1)
  )
  model_ids <- sort(unique(basename(model_dirs[valid])))

  if (length(model_ids) == 0) {
    stop(
      "No model subdirectories containing all required post-hoc checkpoints ",
      "were found below: ", input_root
    )
  }
  cat(model_ids, sep = "\n")
}

print_usage <- function() {
  cat(
    "Usage: ALIGNMENT_MODEL_ID=<model> Rscript ",
    "src/alignment/univariate_stability.R\n",
    "       Rscript src/alignment/univariate_stability.R --list-models\n",
    "\nThe script reads checkpoints written by univariate.R.\n",
    "See src/alignment/README.md for configuration.\n",
    sep = ""
  )
}

if (sys.nframe() == 0L) {
  cli_args <- commandArgs(trailingOnly = TRUE)
  if (any(cli_args %in% c("--help", "-h"))) {
    print_usage()
  } else if ("--list-models" %in% cli_args) {
    list_checkpoint_models()
  } else {
    main()
  }
}
