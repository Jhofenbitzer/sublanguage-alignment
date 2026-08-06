analysis_env_path <- function(variable, default) {
  path <- Sys.getenv(variable, unset = default)
  normalizePath(path, mustWork = FALSE)
}

analysis_paths <- function(create_outputs = TRUE) {
  output_root <- analysis_env_path(
    "ANALYSIS_OUTPUT_ROOT",
    file.path("outputs", "analysis")
  )

  paths <- list(
    univariate_results = analysis_env_path(
      "ANALYSIS_RESULT_ROOT",
      file.path("data", "processed", "alignment", "univariate")
    ),
    multivariate_results = analysis_env_path(
      "RF_RESULT_ROOT",
      file.path("data", "processed", "alignment", "multivariate")
    ),
    output_root = output_root,
    tables = file.path(output_root, "tables"),
    figures = file.path(output_root, "figures"),
    derived = file.path(output_root, "derived")
  )

  if (create_outputs) {
    purrr::walk(
      paths[c("tables", "figures", "derived")],
      dir.create,
      recursive = TRUE,
      showWarnings = FALSE
    )
  }

  paths
}

analysis_output_path <- function(directory, file_name) {
  file.path(directory, file_name)
}

require_columns <- function(data, required, object_name) {
  missing_columns <- setdiff(required, names(data))
  if (length(missing_columns) > 0L) {
    stop(
      object_name,
      " is missing required columns: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }
}

model_label_lookup <- function() {
  tibble::tribble(
    ~base_model_id, ~model_readable_base,
    "apertus-8b-instruct-2509", "Apertus 8 B",
    "apertus-70b-instruct-2509", "Apertus 70 B",
    "biomistral-7b", "BioMistral 7 B",
    "croissantllmchat-v0.1", "Croissant LLM 1 B",
    "fietje-2-instruct", "Fietje 2 3 B",
    "gemma-3-1b-it", "Gemma 3 1 B",
    "gemma-3-12b-it", "Gemma 3 12 B",
    "gemma-3-27b-it", "Gemma 3 27 B",
    "gpt-oss-20b", "GPT OSS 20 B",
    "gpt-oss-120b", "GPT OSS 120 B",
    "gpt-sw3-1.3b-instruct", "GPT SW 3 1.3 B",
    "gpt-sw3-6.7b-v2-instruct", "GPT SW 3 6.7 B",
    "llama3-med42-70b", "LLaMa 3 Med42 70 B",
    "llama-3.3-70b-instruct", "LLaMa 3.3 70 B",
    "llammlein_1b", "LLaMmlein 1 B",
    "llammlein_7b", "LLaMmlein 7 B",
    "medgemma-4b-it", "Medgemma 4 B",
    "medgemma-27b-text-it", "Medgemma 27 B",
    "minerva-7b-instruct-v1.0", "Minerva 7 B",
    "minilingua-1b-instruct", "Minilingua 1 B",
    "ministral-3-8b-instruct-2512", "Ministral 3 8 B",
    "ministral-3-14b-instruct-2512", "Ministral 3 14 B",
    "llama-pllum-8b-instruct", "PLLuM (LLaMa 3.1) 8 B",
    "llama-pllum-70b-instruct", "PLLuM (LLaMa 3.1) 70 B",
    "pllum-12b-instruct", "PLLuM (Mistral Nemo) 12 B",
    "pllum-12b-nc-instruct", "PLLuM-NC (Mistral Nemo) 12 B",
    "pllum-8x7b-instruct", "PLLuM (Mixtral) 47 B",
    "rogemma-7b-instruct", "RoGemma 7 B",
    "rogemma2-9b-instruct", "RoGemma 2 9 B",
    "rollama2-7b-instruct", "RoLLaMa 2 7 B",
    "rollama3-8b-instruct", "RoLLaMa 3 8 B",
    "rollama3.1-8b-instruct", "RoLLaMa 3.1 8 B",
    "romistral-7b-instruct", "RoMistral 7 B",
    "qwen3-next-80b-a3b-instruct", "Qwen 3 Next 80 B",
    "salamandra-2b-instruct", "Salamandra 2 B",
    "salamandra-7b-instruct", "Salamandra 7 B",
    "upper_bound", "Upper bound",
    "biounigram", "Unigram",
    "biogpt-large", "BioGPT"
  )
}

add_model_readable <- function(data) {
  if (!is.data.frame(data) || !"model_id" %in% names(data)) {
    return(data)
  }

  data <- data[, setdiff(names(data), "model_readable"), drop = FALSE]

  data |>
    dplyr::mutate(
      .model_id_normalized = stringr::str_to_lower(as.character(model_id)),
      .model_variant_from_id = dplyr::case_when(
        stringr::str_ends(.model_id_normalized, "_completion") ~ "completion",
        stringr::str_ends(.model_id_normalized, "_prompt") ~ "prompt",
        TRUE ~ NA_character_
      ),
      .model_variant = if ("generation_input" %in% names(data)) {
        dplyr::coalesce(
          dplyr::if_else(
            generation_input %in% c("completion", "prompt"),
            as.character(generation_input),
            NA_character_
          ),
          .model_variant_from_id
        )
      } else {
        .model_variant_from_id
      },
      .base_model_id = stringr::str_remove(
        .model_id_normalized,
        "_(completion|prompt)$"
      )
    ) |>
    dplyr::left_join(
      model_label_lookup(),
      by = c(".base_model_id" = "base_model_id")
    ) |>
    dplyr::mutate(
      .fallback_model_name = if ("model_label" %in% names(data)) {
        stringr::str_remove(
          as.character(model_label),
          stringr::regex(
            "\\s*\\(?\\s*(completion|prompt)\\s*\\)?$",
            ignore_case = TRUE
          )
        )
      } else {
        .base_model_id
      },
      model_readable = stringr::str_c(
        dplyr::coalesce(model_readable_base, .fallback_model_name),
        dplyr::if_else(
          is.na(.model_variant),
          "",
          stringr::str_c(" (", .model_variant, ")")
        )
      )
    ) |>
    dplyr::select(
      -dplyr::starts_with(".model_"),
      -.base_model_id,
      -.fallback_model_name,
      -model_readable_base
    )
}

analysis_contrast_spec <- function() {
  tibble::tribble(
    ~contrast, ~subdirectory, ~reference_label,
    "real_general_vs_biomedical", "", "real general",
    "real_biomedical_vs_biomedical",
    "real_biomedical_vs_biomedical",
    "real biomedical"
  )
}

read_rds_safe <- function(path) {
  if (!file.exists(path)) {
    return(NULL)
  }
  
  tryCatch(
    readRDS(path),
    error = function(error) {
      warning(
        "Could not read ", path, ": ", conditionMessage(error),
        call. = FALSE
      )
      NULL
    }
  )
}

discover_result_index <- function(
    result_root,
    real_results_dir = file.path(result_root, "real"),
    contrast_spec = analysis_contrast_spec()
) {
  if (!dir.exists(result_root)) {
    stop("Result root does not exist: ", result_root)
  }
  
  model_paths <- list.dirs(
    result_root,
    recursive = FALSE,
    full.names = TRUE
  )
  
  purrr::map_dfr(model_paths, function(model_path) {
    contrast_spec |>
      dplyr::mutate(
        analysis_model = basename(model_path),
        model_path = model_path,
        result_dir = dplyr::if_else(
          subdirectory == "",
          model_path,
          file.path(model_path, subdirectory)
        ),
        stable_result_dir = file.path(
          result_dir,
          "stable_significant_clusters"
        ),
        has_scores = file.exists(
          file.path(result_dir, "per_lang_scores.rds")
        ),
        has_completion_marker = file.exists(
          file.path(result_dir, "postprocessing_complete.rds")
        ),
        has_stable_results = dir.exists(stable_result_dir)
      )
  }) |>
    dplyr::filter(
      has_scores,
      normalizePath(model_path, mustWork = FALSE) !=
        normalizePath(real_results_dir, mustWork = FALSE)
    ) |>
    dplyr::arrange(contrast, analysis_model)
}

bind_result_file <- function(
    result_index,
    file_name,
    result_dir_column = "result_dir"
) {
  if (nrow(result_index) == 0L) {
    return(tibble::tibble())
  }
  
  purrr::map_dfr(seq_len(nrow(result_index)), function(row_index) {
    metadata <- result_index[row_index, ]
    path <- file.path(
      metadata[[result_dir_column]][[1]],
      paste0(file_name, ".rds")
    )
    object <- read_rds_safe(path)
    
    if (is.null(object)) {
      return(tibble::tibble())
    }
    if (!is.data.frame(object)) {
      warning("Skipping non-tabular object: ", path, call. = FALSE)
      return(tibble::tibble())
    }
    
    tibble::as_tibble(object) |>
      dplyr::mutate(
        analysis_model = metadata$analysis_model[[1]],
        contrast = metadata$contrast[[1]],
        reference_label = metadata$reference_label[[1]],
        .before = 1
      )
  })
}

load_analysis_results <- function(
    result_root,
    files,
    stable_files = character(),
    real_results_dir = file.path(result_root, "real")
) {
  result_index <- discover_result_index(
    result_root = result_root,
    real_results_dir = real_results_dir
  )
  
  if (nrow(result_index) == 0L) {
    stop("No model result files were discovered below: ", result_root)
  }
  
  primary <- stats::setNames(
    purrr::map(
      files,
      ~ bind_result_file(result_index, .x)
    ),
    files
  )
  
  stable_index <- result_index |>
    dplyr::filter(has_stable_results)
  
  stable <- stats::setNames(
    purrr::map(
      stable_files,
      ~ bind_result_file(
        stable_index,
        .x,
        result_dir_column = "stable_result_dir"
      )
    ),
    stable_files
  )
  
  list(
    index = result_index,
    primary = primary,
    stable_index = stable_index,
    stable = stable
  )
}

read_real_table <- function(real_results_dir, file_name, required = TRUE) {
  path <- file.path(real_results_dir, paste0(file_name, ".rds"))
  object <- read_rds_safe(path)
  
  if (is.null(object)) {
    if (required) {
      stop("Required authentic reference result is unavailable: ", path)
    }
    return(tibble::tibble())
  }
  if (!is.data.frame(object)) {
    stop("Authentic reference object is not tabular: ", path)
  }
  
  tibble::as_tibble(object)
}

add_missing_columns <- function(data, columns, value = NA_real_) {
  for (column in setdiff(columns, names(data))) {
    data[[column]] <- rep(value, nrow(data))
  }
  data
}

safe_ratio <- function(numerator, denominator) {
  dplyr::if_else(
    is.finite(denominator) & denominator > 0,
    numerator / denominator,
    NA_real_
  )
}

safe_f1 <- function(precision, recall) {
  dplyr::case_when(
    !is.finite(precision) | !is.finite(recall) ~ NA_real_,
    precision + recall == 0 ~ 0,
    TRUE ~ 2 * precision * recall / (precision + recall)
  )
}

safe_spearman <- function(x, y, minimum_n = 3L) {
  keep <- is.finite(x) & is.finite(y)
  if (sum(keep) < minimum_n) {
    return(NA_real_)
  }
  suppressWarnings(
    stats::cor(x[keep], y[keep], method = "spearman")
  )
}

normalize_direction <- function(direction) {
  dplyr::case_when(
    stringr::str_detect(direction, ">") ~ ">",
    stringr::str_detect(direction, "<") ~ "<",
    TRUE ~ NA_character_
  )
}

split_cluster_members <- function(members) {
  if (length(members) == 0L || is.na(members) || !nzchar(members)) {
    return(character())
  }
  
  members |>
    as.character() |>
    stringr::str_split(",\\s*") |>
    purrr::pluck(1) |>
    stringr::str_trim() |>
    unique()
}

jaccard_similarity <- function(first, second) {
  combined <- union(first, second)
  if (length(combined) == 0L) {
    return(NA_real_)
  }
  length(intersect(first, second)) / length(combined)
}

# Maximum-cardinality one-to-one matching for a thresholded similarity matrix.
# This avoids allowing one cluster to "recover" several reference clusters.
maximum_threshold_matching <- function(similarity, threshold = 0.50) {
  if (nrow(similarity) == 0L || ncol(similarity) == 0L) {
    return(0L)
  }
  
  eligible <- is.finite(similarity) & similarity >= threshold
  matched_reference <- rep(NA_integer_, ncol(eligible))
  
  augment <- function(reference_row, seen_model) {
    candidates <- which(eligible[reference_row, ] & !seen_model)
    if (length(candidates) == 0L) {
      return(FALSE)
    }
    
    for (model_column in candidates) {
      seen_model[[model_column]] <- TRUE
      previous_reference <- matched_reference[[model_column]]
      if (
        is.na(previous_reference) ||
        augment(previous_reference, seen_model)
      ) {
        matched_reference[[model_column]] <<- reference_row
        return(TRUE)
      }
    }
    FALSE
  }
  
  matched <- 0L
  for (reference_row in seq_len(nrow(eligible))) {
    if (augment(reference_row, rep(FALSE, ncol(eligible)))) {
      matched <- matched + 1L
    }
  }
  matched
}

compare_cluster_sets <- function(
    reference_members,
    model_members,
    threshold = 0.50
) {
  reference_sets <- purrr::map(reference_members, split_cluster_members)
  model_sets <- purrr::map(model_members, split_cluster_members)
  
  n_reference <- length(reference_sets)
  n_model <- length(model_sets)
  if (n_reference == 0L || n_model == 0L) {
    return(tibble::tibble(
      n_reference_clusters = n_reference,
      n_model_clusters = n_model,
      n_recovered_clusters = 0L,
      recall = ifelse(n_reference > 0L, 0, NA_real_),
      precision = ifelse(n_model > 0L, 0, NA_real_),
      f1 = NA_real_,
      mean_best_jaccard = NA_real_
    ))
  }
  
  similarity <- outer(
    seq_along(reference_sets),
    seq_along(model_sets),
    Vectorize(function(reference_index, model_index) {
      jaccard_similarity(
        reference_sets[[reference_index]],
        model_sets[[model_index]]
      )
    })
  )
  
  n_recovered <- maximum_threshold_matching(similarity, threshold)
  recall <- n_recovered / n_reference
  precision <- n_recovered / n_model
  
  tibble::tibble(
    n_reference_clusters = n_reference,
    n_model_clusters = n_model,
    n_recovered_clusters = n_recovered,
    recall = recall,
    precision = precision,
    f1 = ifelse(
      precision + recall > 0,
      2 * precision * recall / (precision + recall),
      NA_real_
    ),
    mean_best_jaccard = mean(apply(similarity, 1, max), na.rm = TRUE)
  )
}
