suppressPackageStartupMessages({
  library(OpenSpecy)
  library(hdf5r)
})

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1) args[[1]] else ".tooling/openspecy_lib"
output_dir <- if (length(args) >= 2) args[[2]] else "data/openspecy"
min_coverage <- if (length(args) >= 3) as.numeric(args[[3]]) else 0.80

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

normalize_text <- function(x) {
  x <- iconv(as.character(x), from = "", to = "ASCII//TRANSLIT")
  x <- tolower(trimws(x))
  x <- gsub("[^a-z0-9]+", " ", x)
  x <- gsub("\\s+", " ", x)
  x[is.na(x) | x %in% c("", "na", "n a", "null", "unknown")] <- NA_character_
  x
}

lib <- load_lib("raw", path = input_dir)
metadata <- as.data.frame(lib$metadata, stringsAsFactors = FALSE)

modality_raw <- normalize_text(metadata$spectrum_type)
metadata$modality <- ifelse(
  modality_raw == "ftir",
  "ir",
  ifelse(modality_raw == "raman", "raman", NA_character_)
)

identity <- normalize_text(metadata$spectrum_identity)
cas <- normalize_text(metadata$cas_number)
metadata$identity_key <- ifelse(
  !is.na(cas),
  paste0("cas:", cas),
  paste0("name:", identity)
)
metadata$identity_key[is.na(identity) & is.na(cas)] <- NA_character_

axis_mask <- lib$wavenumber >= 500 & lib$wavenumber <= 4000
wavenumber <- lib$wavenumber[axis_mask]
if (length(wavenumber) < 2) stop("No Open Specy points found in 500--4000 cm^-1")

candidate <- which(!is.na(metadata$modality) & !is.na(metadata$identity_key))
candidate_names <- metadata$sample_name[candidate]
spectra_columns <- match(candidate_names, names(lib$spectra))
if (anyNA(spectra_columns)) {
  stop("Some metadata sample names do not match spectrum columns")
}

coverage <- vapply(
  spectra_columns,
  function(column) mean(is.finite(lib$spectra[[column]][axis_mask])),
  numeric(1)
)
candidate <- candidate[coverage >= min_coverage]
metadata <- metadata[candidate, , drop = FALSE]
metadata$finite_coverage_500_4000 <- coverage[coverage >= min_coverage]

modality_by_identity <- split(metadata$modality, metadata$identity_key)
dual_identity <- names(Filter(
  function(x) all(c("ir", "raman") %in% unique(x)),
  modality_by_identity
))
metadata <- metadata[metadata$identity_key %in% dual_identity, , drop = FALSE]

spectra_columns <- match(metadata$sample_name, names(lib$spectra))
if (anyNA(spectra_columns)) stop("Internal spectrum-column alignment error")

metadata$benchmark_index <- seq_len(nrow(metadata)) - 1L
metadata_out <- metadata[, intersect(
  c(
    "benchmark_index", "sample_name", "modality", "identity_key",
    "spectrum_identity", "cas_number", "rruffid", "sample_id",
    "product_id", "polymer_id", "locality", "organization", "citation",
    "librarytype", "instrument_used", "instrument_mode", "intensity_units",
    "spectral_resolution", "laser_light_used", "wavenumber_range",
    "finite_coverage_500_4000"
  ),
  names(metadata)
), drop = FALSE]

metadata_path <- file.path(output_dir, "metadata.csv")
utils::write.csv(metadata_out, metadata_path, row.names = FALSE, na = "")

h5_path <- file.path(output_dir, "spectra.h5")
if (file.exists(h5_path)) file.remove(h5_path)
h5 <- H5File$new(h5_path, mode = "w")
on.exit(h5$close_all(), add = TRUE)

h5[["wavenumber"]] <- as.numeric(wavenumber)
chunk_rows <- min(256L, nrow(metadata))
spectra_ds <- h5$create_dataset(
  "spectra",
  dtype = h5types$H5T_IEEE_F32LE,
  dims = c(nrow(metadata), length(wavenumber)),
  chunk_dims = c(chunk_rows, length(wavenumber)),
  gzip_level = 4
)

for (start in seq.int(1L, nrow(metadata), by = chunk_rows)) {
  end <- min(start + chunk_rows - 1L, nrow(metadata))
  batch_columns <- spectra_columns[start:end]
  batch <- t(as.matrix(lib$spectra[axis_mask, ..batch_columns]))
  spectra_ds[start:end, ] <- batch
}

h5$close_all()

counts <- table(metadata$modality)
cat("Open Specy benchmark prepared\n")
cat("  Output:", normalizePath(output_dir), "\n")
cat("  Wavenumber points:", length(wavenumber), "(500--4000 cm^-1)\n")
cat("  Minimum finite coverage:", min_coverage, "\n")
cat("  Dual-modality identities:", length(unique(metadata$identity_key)), "\n")
cat("  FTIR spectra:", unname(counts[["ir"]]), "\n")
cat("  Raman spectra:", unname(counts[["raman"]]), "\n")
cat("  Total spectra:", nrow(metadata), "\n")
