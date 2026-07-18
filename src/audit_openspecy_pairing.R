suppressPackageStartupMessages({
  library(OpenSpecy)
  library(hdf5r)
})

args <- commandArgs(trailingOnly = TRUE)
library_dir <- if (length(args) >= 1) args[[1]] else ".tooling/openspecy_lib"
pair_dir <- if (length(args) >= 2) args[[2]] else "datasets/openspecy_rruff_576"

normalize_text <- function(x) {
  x <- tolower(trimws(as.character(x)))
  x[is.na(x) | x %in% c("", "na", "n/a", "null", "unknown")] <- NA_character_
  x
}

field <- function(data, name) {
  if (name %in% names(data)) data[[name]] else rep(NA_character_, nrow(data))
}

count_shared_ids <- function(metadata, id_field) {
  values <- normalize_text(field(metadata, id_field))
  modality <- metadata$audit_modality
  valid <- !is.na(values) & !is.na(modality)
  if (!any(valid)) return(c(rows = 0L, ids = 0L, dual_ids = 0L))
  modalities <- split(modality[valid], values[valid])
  dual <- vapply(
    modalities,
    function(x) all(c("ir", "raman") %in% unique(x)),
    logical(1)
  )
  c(rows = sum(valid), ids = length(modalities), dual_ids = sum(dual))
}

audit_pair_file <- function(path, metadata, sample_lookup) {
  pairs <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  ir_row <- unname(sample_lookup[as.character(pairs$ir_sample_name)])
  raman_row <- unname(sample_lookup[as.character(pairs$raman_sample_name)])
  valid_rows <- !is.na(ir_row) & !is.na(raman_row)

  ir_meta <- metadata[ir_row[valid_rows], , drop = FALSE]
  raman_meta <- metadata[raman_row[valid_rows], , drop = FALSE]
  expected_id <- normalize_text(pairs$identity_key[valid_rows])
  ir_rruff <- normalize_text(field(ir_meta, "rruffid"))
  raman_rruff <- normalize_text(field(raman_meta, "rruffid"))
  ir_identity <- normalize_text(field(ir_meta, "spectrum_identity"))
  raman_identity <- normalize_text(field(raman_meta, "spectrum_identity"))
  ir_locality <- normalize_text(field(ir_meta, "locality"))
  raman_locality <- normalize_text(field(raman_meta, "locality"))

  result <- data.frame(
    pair_file = basename(path),
    total_pairs = nrow(pairs),
    samples_found = sum(valid_rows),
    ir_modality_correct = sum(ir_meta$audit_modality == "ir", na.rm = TRUE),
    raman_modality_correct = sum(raman_meta$audit_modality == "raman", na.rm = TRUE),
    same_rruffid = sum(!is.na(ir_rruff) & ir_rruff == raman_rruff),
    expected_rruffid = sum(
      !is.na(ir_rruff) &
        ir_rruff == expected_id &
        raman_rruff == expected_id
    ),
    same_material_name = sum(
      !is.na(ir_identity) & ir_identity == raman_identity
    ),
    same_nonempty_locality = sum(
      !is.na(ir_locality) &
        !is.na(raman_locality) &
        ir_locality == raman_locality
    )
  )

  mismatch <- which(
    is.na(ir_rruff) |
      is.na(raman_rruff) |
      ir_rruff != expected_id |
      raman_rruff != expected_id |
      ir_meta$audit_modality != "ir" |
      raman_meta$audit_modality != "raman"
  )

  if (length(mismatch) > 0) {
    cat("\nFirst strict-pair mismatches in", basename(path), ":\n")
    print(utils::head(data.frame(
      expected_id = expected_id[mismatch],
      ir_sample = pairs$ir_sample_name[valid_rows][mismatch],
      ir_rruffid = ir_rruff[mismatch],
      ir_modality = ir_meta$audit_modality[mismatch],
      raman_sample = pairs$raman_sample_name[valid_rows][mismatch],
      raman_rruffid = raman_rruff[mismatch],
      raman_modality = raman_meta$audit_modality[mismatch]
    ), 10), row.names = FALSE)
  }

  result
}

read_h5_spectra <- function(path, expected_rows) {
  handle <- H5File$new(path, mode = "r")
  on.exit(handle$close_all())
  spectra <- handle[["spectra"]][, ]
  if (nrow(spectra) == expected_rows) return(spectra)
  if (ncol(spectra) == expected_rows) return(t(spectra))
  stop(
    "Unexpected HDF5 shape for ", path, ": ",
    paste(dim(spectra), collapse = " x "),
    "; expected ", expected_rows, " rows"
  )
}

resample_like_python <- function(y, source_x, target_x) {
  output <- approx(source_x, y, xout = target_x, rule = 1)$y
  span <- max(output) - min(output)
  if (span < 1e-8) return(output - min(output))
  (output - min(output)) / span
}

audit_derived_h5 <- function(
    pair_path, h5_path, sample_column, metadata, sample_lookup,
    lib, source_x, target_x, max_checks = 25L) {
  pairs <- read.csv(pair_path, stringsAsFactors = FALSE, check.names = FALSE)
  sample_names <- as.character(pairs[[sample_column]])
  stored <- read_h5_spectra(h5_path, nrow(pairs))
  checks <- seq_len(min(max_checks, nrow(pairs)))

  correlation <- numeric(length(checks))
  max_abs_error <- numeric(length(checks))
  for (j in seq_along(checks)) {
    i <- checks[[j]]
    metadata_row <- unname(sample_lookup[sample_names[[i]]])
    if (is.na(metadata_row)) stop("Sample absent from raw metadata: ", sample_names[[i]])
    raw_column <- match(sample_names[[i]], names(lib$spectra))
    if (is.na(raw_column)) stop("Sample absent from raw spectra: ", sample_names[[i]])
    raw <- as.numeric(lib$spectra[[raw_column]])[lib$wavenumber >= 500 & lib$wavenumber <= 4000]
    expected <- resample_like_python(raw, source_x, target_x)
    correlation[[j]] <- suppressWarnings(cor(expected, stored[i, ]))
    max_abs_error[[j]] <- max(abs(expected - stored[i, ]))
  }

  data.frame(
    h5_file = basename(h5_path),
    checked_rows = length(checks),
    minimum_correlation = min(correlation, na.rm = TRUE),
    maximum_abs_error = max(max_abs_error, na.rm = TRUE),
    exact_within_1e_5 = sum(max_abs_error < 1e-5)
  )
}

cat("Loading raw OpenSpecy library from:", library_dir, "\n")
lib <- load_lib("raw", path = library_dir)
metadata <- as.data.frame(lib$metadata, stringsAsFactors = FALSE)

type <- normalize_text(field(metadata, "spectrum_type"))
metadata$audit_modality <- ifelse(
  type == "ftir",
  "ir",
  ifelse(type == "raman", "raman", NA_character_)
)

cat("\nRaw OpenSpecy rows:", nrow(metadata), "\n")
cat("FTIR rows:", sum(metadata$audit_modality == "ir", na.rm = TRUE), "\n")
cat("Raman rows:", sum(metadata$audit_modality == "raman", na.rm = TRUE), "\n")

id_fields <- c(
  "rruffid", "sample_id", "product_id", "polymer_id",
  "cas_number", "spectrum_identity"
)
id_summary <- do.call(
  rbind,
  lapply(id_fields, function(name) {
    counts <- count_shared_ids(metadata, name)
    data.frame(
      identifier = name,
      nonempty_rows = unname(counts["rows"]),
      distinct_ids = unname(counts["ids"]),
      dual_modality_ids = unname(counts["dual_ids"])
    )
  })
)
cat("\nCandidate identity fields in the raw R metadata:\n")
print(id_summary, row.names = FALSE)

sample_names <- as.character(field(metadata, "sample_name"))
sample_lookup <- setNames(seq_len(nrow(metadata)), sample_names)
pair_paths <- file.path(pair_dir, c("train_pairs.csv", "test_pairs.csv"))
missing_pair_files <- pair_paths[!file.exists(pair_paths)]
if (length(missing_pair_files) > 0) {
  stop("Missing pair files: ", paste(missing_pair_files, collapse = ", "))
}

pair_summary <- do.call(
  rbind,
  lapply(pair_paths, audit_pair_file, metadata = metadata, sample_lookup = sample_lookup)
)
cat("\nAudit of Python-generated pair files against raw R metadata:\n")
print(pair_summary, row.names = FALSE)

rruff <- normalize_text(field(metadata, "rruffid"))
shared_rruff <- names(Filter(
  function(x) all(c("ir", "raman") %in% unique(x)),
  split(metadata$audit_modality[!is.na(rruff)], rruff[!is.na(rruff)])
))
shared_rows <- metadata[rruff %in% shared_rruff, , drop = FALSE]

organization <- normalize_text(field(shared_rows, "organization"))
librarytype <- normalize_text(field(shared_rows, "librarytype"))
cat("\nOrganizations among strict shared-RRUFF rows:\n")
print(sort(table(organization, useNA = "ifany"), decreasing = TRUE))
cat("\nLibrary types among strict shared-RRUFF rows:\n")
print(sort(table(librarytype, useNA = "ifany"), decreasing = TRUE))

cat("\nExample strict shared-RRUFF records:\n")
example_columns <- intersect(
  c(
    "sample_name", "spectrum_type", "rruffid", "spectrum_identity",
    "locality", "organization", "instrument_used", "laser_light_used"
  ),
  names(shared_rows)
)
print(utils::head(shared_rows[, example_columns, drop = FALSE], 12), row.names = FALSE)

axis_mask <- lib$wavenumber >= 500 & lib$wavenumber <= 4000
source_x <- as.numeric(lib$wavenumber[axis_mask])
target_x <- seq(min(source_x), max(source_x), length.out = 576L)
h5_checks <- rbind(
  audit_derived_h5(
    pair_paths[[1]], file.path(pair_dir, "train_ir.h5"), "ir_sample_name",
    metadata, sample_lookup, lib, source_x, target_x
  ),
  audit_derived_h5(
    pair_paths[[1]], file.path(pair_dir, "train_raman.h5"), "raman_sample_name",
    metadata, sample_lookup, lib, source_x, target_x
  ),
  audit_derived_h5(
    pair_paths[[2]], file.path(pair_dir, "test_ir.h5"), "ir_sample_name",
    metadata, sample_lookup, lib, source_x, target_x
  ),
  audit_derived_h5(
    pair_paths[[2]], file.path(pair_dir, "test_raman.h5"), "raman_sample_name",
    metadata, sample_lookup, lib, source_x, target_x
  )
)
cat("\nNumerical audit of HDF5 rows against raw R spectra:\n")
print(h5_checks, row.names = FALSE)

write.csv(
  pair_summary,
  file.path(pair_dir, "r_openspecy_pair_audit.csv"),
  row.names = FALSE
)
write.csv(
  h5_checks,
  file.path(pair_dir, "r_openspecy_h5_audit.csv"),
  row.names = FALSE
)
cat("\nSaved audit summary to:", file.path(pair_dir, "r_openspecy_pair_audit.csv"), "\n")
