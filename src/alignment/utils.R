# Internal configuration and graph helpers for the alignment analyses.

alignment_script_path <- function() {
  file_args <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_args) == 0) return(NA_character_)
  normalizePath(sub("^--file=", "", file_args[[1]]), mustWork = FALSE)
}

ALIGNMENT_SOURCE_DIR <- trimws(Sys.getenv("ALIGNMENT_SOURCE_DIR", unset = ""))
if (!nzchar(ALIGNMENT_SOURCE_DIR)) {
  script_path <- alignment_script_path()
  ALIGNMENT_SOURCE_DIR <- if (is.na(script_path)) getwd() else dirname(script_path)
}
ALIGNMENT_SOURCE_DIR <- normalizePath(ALIGNMENT_SOURCE_DIR, mustWork = FALSE)

alignment_env_value <- function(name, default, alias = NULL) {
  value <- trimws(Sys.getenv(name, unset = ""))
  if (!nzchar(value) && !is.null(alias)) {
    value <- trimws(Sys.getenv(alias, unset = ""))
  }
  if (nzchar(value)) value else default
}

alignment_env_flag <- function(name, default, alias = NULL) {
  fallback <- if (isTRUE(default)) "true" else "false"
  tolower(alignment_env_value(name, fallback, alias)) %in%
    c("1", "true", "t", "yes", "y")
}

alignment_env_csv <- function(name, default = character(), alias = NULL) {
  value <- alignment_env_value(name, paste(default, collapse = ","), alias)
  values <- trimws(strsplit(value, ",", fixed = TRUE)[[1]])
  unique(values[nzchar(values)])
}

ALIGNMENT_PROJECT_ROOT <- normalizePath(
  alignment_env_value(
    "ALIGNMENT_PROJECT_ROOT",
    file.path(ALIGNMENT_SOURCE_DIR, "..", "..")
  ),
  mustWork = FALSE
)

alignment_validate_model_id <- function(model_id, variable = "ALIGNMENT_MODEL_ID",
                                        allow_empty = TRUE) {
  if (!nzchar(model_id) && allow_empty) return(invisible(model_id))
  if (!nzchar(model_id) || !grepl("^[A-Za-z0-9._-]+$", model_id)) {
    stop("Invalid ", variable, ": ", model_id)
  }
  invisible(model_id)
}

alignment_require_columns <- function(x, required, object_name) {
  missing <- setdiff(required, names(x))
  if (length(missing) > 0) {
    stop(
      object_name,
      " is missing required columns: ",
      paste(missing, collapse = ", ")
    )
  }
  invisible(x)
}

# Return connected-component membership for an undirected feature graph.
alignment_components <- function(nodes, edges_df) {
  nodes <- unique(as.character(nodes))
  parent <- setNames(nodes, nodes)
  rank_value <- setNames(integer(length(nodes)), nodes)

  find_root <- function(x) {
    while (parent[[x]] != x) {
      parent[[x]] <<- parent[[parent[[x]]]]
      x <- parent[[x]]
    }
    x
  }

  union_sets <- function(a, b) {
    root_a <- find_root(a)
    root_b <- find_root(b)
    if (root_a == root_b) return(invisible(NULL))

    if (rank_value[[root_a]] < rank_value[[root_b]]) {
      parent[[root_a]] <<- root_b
    } else if (rank_value[[root_a]] > rank_value[[root_b]]) {
      parent[[root_b]] <<- root_a
    } else {
      parent[[root_b]] <<- root_a
      rank_value[[root_a]] <<- rank_value[[root_a]] + 1L
    }
    invisible(NULL)
  }

  if (nrow(edges_df) > 0) {
    for (i in seq_len(nrow(edges_df))) {
      union_sets(as.character(edges_df[[1]][i]), as.character(edges_df[[2]][i]))
    }
  }

  roots <- vapply(nodes, find_root, character(1))
  tibble::tibble(feature = nodes, comp = match(roots, unique(roots)))
}
