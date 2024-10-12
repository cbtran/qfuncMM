#' Estimate functional connectivity from voxel-level BOLD signals. Run stage 1 intra-regional analysis for a single region.
#'   Results are saved to a JSON file.
#'
#' @param subject_id Identifies the subject/exam.
#' @param region_uniqud Uniquely identifies the region for a given subject.
#' @param region_name Name of the region.
#' @param region_data \eqn{M\times L_j} matrix
#' @param region_coords \eqn{L_j \times 3} matrix of spatial coordinates.
#' @param kernel_type Choice of spatial kernel.
#' @param cov_setting Choice of covariance structure. Default option 'auto' chooses between 'noisy' and 'noiseless' based on model fit.
#' @param out_file Output file path.
#' @param overwrite Overwrite output file if it exists.
#' @param num_init Number of initializations for multi-start optimiization.
#' @param verbose Print progress messages.
#'
#' @useDynLib qfuncMM
#' @importFrom Rcpp sourceCpp
#' @importFrom jsonlite toJSON
#' @export
qfuncMM_stage1_intra <- function(subject_id, region_uniqid, region_name, region_data, region_coords,
                                 kernel_type = "matern_5_2",
                                 cov_setting = c("auto", "noisy", "noiseless"),
                                 out_file = NULL,
                                 overwrite = FALSE,
                                 num_init = 10L,
                                 verbose = FALSE) {
  region_uniqid <- as.integer(region_uniqid)
  region_name <- as.character(region_name)
  subject_id <- as.character(subject_id)
  start_time <- Sys.time()
  kernel_type_id <- kernel_dict(kernel_type)
  cov_setting <- match.arg(cov_setting)
  log_var_ratio_threshold <- 5
  psi_threshold <- 0.5

  if (is.null(out_file)) {
    out_file <- file.path(
      ".",
      sprintf("qfuncMM_stage1_intra_region_%s_%d_%s.json", subject_id, region_uniqid, format(start_time, "%Y%m%d_%H%M%S"))
    )
  }
  out_dir <- dirname(out_file)
  if (!file.info(out_dir)$isdir || file.access(out_dir, mode = 2) != 0) {
    stop(sprintf("The specified output location '%s' is not in a valid, writeable directory.", out_file))
  }
  if (file.exists(out_file) && !overwrite) {
    stop(sprintf("File '%s' already exists. Set overwrite = TRUE to overwrite.", out_file))
  }

  n_timept <- nrow(region_data)
  n_voxel <- ncol(region_data)
  stopifnot(n_voxel == nrow(region_coords))

  message(sprintf(
    "Running QFunCMM stage 1 intra-regional for region %d-'%s' with %d voxels and %d time points using %d inits.",
    region_uniqid, region_name, n_voxel, n_timept, num_init
  ))

  # Standardize the data matrices
  region_data_std <- (region_data - mean(region_data)) / stats::sd(region_data)

  if (cov_setting %in% c("auto", "noisy")) {
    message("Fitting noisy model...")
    intra <- fit_intra_model(
      region_data_std,
      region_coords,
      kernel_type_id,
      "noisy",
      num_init,
      verbose = verbose
    )
    if (all(is.infinite(intra$results_by_init[, "nll"]))) {
      stop("The optimization has failed for every initialization. Try increasing the number of initializations or checking your data.")
    }
    if (cov_setting == "auto") {
      message("Checking model fit")
      var_ratio <- log(intra$intra_param["k_gamma"] + intra$intra_param["nugget_gamma"])
      if (var_ratio > log_var_ratio_threshold || intra$intra_param["psi"] >= psi_threshold) {
        cov_setting <- "noiseless"
      } else {
        cov_setting <- "noisy"
      }
      message(sprintf(
        "Model fit: var_ratio = %.3f, psi = %.3f. Choosing %s model.",
        var_ratio, intra$intra_param["psi"], cov_setting
      ))
    }
  }
  if (cov_setting == "noiseless") {
    message("Fitting noiseless model...")
    intra <- fit_intra_model(
      region_data_std,
      region_coords,
      kernel_type_id,
      "noiseless",
      num_init,
      verbose = verbose
    )
    if (all(is.infinite(intra$results_by_init[, "nll"]))) {
      stop("The optimization has failed for every initialization. Try increasing the number of initializations or checking your data.")
    }
  }

  outlist <- list(
    region_uniqid = region_uniqid, region_name = region_name, subject_id = subject_id,
    cov_setting = cov_setting, kernel_type = kernel_type,
    start_time = format(start_time, "%Y-%m-%dT%H:%M:%OS3Z"),
    end_time = format(Sys.time(), "%Y-%m-%dT%H:%M:%OS3Z")
  )

  outlist$inits <- intra$initializations
  outlist$results_by_init <- intra$results_by_init
  outlist$stage1 <- as.list(intra$intra_param)
  outlist$eblue <- as.numeric(intra$eblue)
  outlist$data_std <- region_data_std
  outlist$coords <- region_coords

  out_json <- jsonlite::toJSON(outlist, auto_unbox = TRUE, pretty = TRUE, digits = I(10))
  write(out_json, out_file)
  message(
    sprintf("Subject %s region %d: Finished stage 1 intra-regional. \nResults saved to ", subject_id, region_uniqid),
    normalizePath(out_file)
  )
}
