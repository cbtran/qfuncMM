#' Estimate functional connectivity from voxel-level BOLD signals. Run stage 1 intra-regional analysis for a single region.
#'   Results are saved to a JSON file.
#'
#' @param subject_id Identifies the subject/exam.
#' @param region_uniqid Uniquely identifies the region for a given subject.
#' @param region_name Name of the region.
#' @param region_data \eqn{M\times L_j} matrix
#' @param region_coords \eqn{L_j \times 3} matrix of spatial coordinates.
#' @param out_dir Output directory.
#' @param kernel_type Choice of spatial kernel.
#' @param cov_setting Choice of covariance structure. Default option 'auto' chooses between 'noisy' and 'noiseless' based on model fit.
#' @param num_init Number of initializations for multi-start optimiization.
#' @param noisy_num_init_tries Number of initializations to try for the noisy model before switching to noiseless.
#' @param log_var_ratio_threshold Threshold for log(var_ratio) to switch between 'noisy' and 'noiseless' models.
#' @param psi_threshold Threshold for psi to switch between 'noisy' and 'noiseless' models.
#' @param overwrite Overwrite existing output file.
#' @param verbose Print progress messages.
#'
#' @useDynLib qfuncMM
#' @importFrom Rcpp sourceCpp
#' @importFrom jsonlite toJSON
#' @export
qfuncMM_stage1_intra <- function(
    subject_id, region_uniqid, region_name, region_data, region_coords, out_dir,
    kernel_type = "matern_5_2",
    cov_setting = c("auto", "noisy", "noiseless"),
    num_init = 10L,
    noisy_num_init_tries = as.integer(num_init / 2),
    log_var_ratio_threshold = 5,
    psi_threshold = 0.5,
    overwrite = FALSE,
    verbose = FALSE) {
  region_uniqid <- as.integer(region_uniqid)
  region_name <- as.character(region_name)
  subject_id <- as.character(subject_id)
  kernel_type_id <- kernel_dict(kernel_type)
  cov_setting <- match.arg(cov_setting)
  stopifnot(num_init >= 1L)
  if (noisy_num_init_tries > num_init || cov_setting != "auto" || num_init == 1L) {
    noisy_num_init_tries <- num_init
  }

  bad_noisy_fit <- function(intra_result) {
    log_var_ratio <- log(intra_result$intra_param["k_gamma"] + intra_result$intra_param["nugget_gamma"])
    psi <- intra_result$intra_param["psi"]
    noiseless_better <- log_var_ratio > log_var_ratio_threshold || psi >= psi_threshold
    return(c(noiseless_better, log_var_ratio, psi))
  }

  if (!dir.exists(out_dir)) {
    dir.create(out_dir)
  }
  if (!file.info(out_dir)$isdir || file.access(out_dir, mode = 2) != 0) {
    stop(sprintf("The specified output location '%s' is not in a valid, writeable directory.", out_dir))
  }
  out_file <- file.path(
    out_dir,
    sprintf("qfuncMM_stage1_intra_region_%s_%d.json", subject_id, region_uniqid)
  )
  if (file.exists(out_file) && !overwrite) {
    stop(sprintf("Output file '%s' already exists. Set 'overwrite' to TRUE to overwrite.", out_file))
  }

  n_timept <- nrow(region_data)
  n_voxel <- ncol(region_data)
  stopifnot(n_voxel == nrow(region_coords))

  message(sprintf(
    "Running QFunCMM stage 1 intra-regional for region %d-'%s' with %d voxels and %d time points using %d inits...",
    region_uniqid, region_name, n_voxel, n_timept, num_init
  ))

  start_time <- Sys.time()
  # Standardize the data matrices
  region_data_std <- (region_data - mean(region_data)) / stats::sd(region_data)
  inits <- stage1_init(region_data_std, region_coords, num_init, FALSE)

  if (cov_setting %in% c("auto", "noisy")) {
    message("Fitting noisy model...")
    intra <- fit_intra_model(
      region_data_std,
      region_coords,
      inits[seq(1, noisy_num_init_tries), , drop = FALSE],
      kernel_type_id,
      "noisy",
      verbose = verbose
    )
    if (all(is.infinite(intra$results_by_init[, "nll"]))) {
      stop("The optimization has failed for every initialization. Try increasing the number of initializations or checking your data.")
    }
    if (cov_setting == "auto") {
      message(sprintf("Checking model fit over %d noisy inits...", noisy_num_init_tries))
      init_check <- bad_noisy_fit(intra)
      if (init_check[1]) {
        cov_setting <- "noiseless"
        message(sprintf(
          "Model fit: log(var_ratio) = %.3f, psi = %.3f. Choosing %s model.",
          init_check[2], init_check[3], cov_setting
        ))
      } else if (num_init == 1L) {
        cov_setting <- "noisy"
        message(sprintf(
          "Model fit: log(var_ratio) = %.3f, psi = %.3f. Choosing %s model.",
          init_check[2], init_check[3], cov_setting
        ))
      } else {
        message(sprintf("Running remaining %d noisy inits...", num_init - noisy_num_init_tries))
        intra2 <- fit_intra_model(
          region_data_std,
          region_coords,
          inits[seq(noisy_num_init_tries + 1, num_init), ],
          kernel_type_id,
          "noisy",
          verbose = verbose
        )
        intra$initializations <- rbind(intra$initializations, intra2$initializations)
        intra$results_by_init <- rbind(intra$results_by_init, intra2$results_by_init)
        intra$intra_param <- `if`(intra$objval < intra2$objval, intra$intra_param, intra2$intra_param)
        intra$eblue <- `if`(intra$objval < intra2$objval, intra$eblue, intra2$eblue)
        intra$objval <- min(intra$objval, intra2$objval)

        init_check <- bad_noisy_fit(intra)
        if (init_check[1]) {
          cov_setting <- "noiseless"
        } else {
          cov_setting <- "noisy"
        }
        message(sprintf(
          "Model fit: log(var_ratio) = %.3f, psi = %.3f. Choosing %s model.",
          init_check[2], init_check[3], cov_setting
        ))
      }
    }
  }
  if (cov_setting == "noiseless") {
    message("Fitting noiseless model...")
    intra <- fit_intra_model(
      region_data_std,
      region_coords,
      inits,
      kernel_type_id,
      "noiseless",
      verbose = verbose
    )
    if (all(is.infinite(intra$results_by_init[, "nll"]))) {
      stop("The optimization has failed for every initialization. Try increasing the number of initializations or checking your data.")
    }
  }

  outlist <- list(
    region_uniqid = region_uniqid, region_name = region_name, subject_id = subject_id,
    num_voxels = n_voxel, num_timepoints = n_timept,
    cov_setting = cov_setting, kernel_type = kernel_type,
    start_time = format(start_time, "%Y-%m-%dT%H:%M:%OS3Z"),
    run_time_minutes = round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 5),
    inits = intra$initializations,
    results_by_init = intra$results_by_init,
    stage1 = as.list(intra$intra_param),
    eblue = as.numeric(intra$eblue),
    data_std = region_data_std,
    coords = region_coords
  )

  out_json <- jsonlite::toJSON(outlist, auto_unbox = TRUE, pretty = TRUE, digits = I(10))
  write(out_json, out_file)
  message(
    sprintf("Subject %s region %d: Finished stage 1 intra-regional. \nResults saved to ", subject_id, region_uniqid),
    normalizePath(out_file)
  )
}
