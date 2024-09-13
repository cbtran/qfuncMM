#' Estimate functional connectivity from voxel-level BOLD signals. Run stage 1 intra-regional analysis.
#'   Results are saved to a JSON file.
#'
#' @param region_list List of \eqn{M\times L_j} region matrices.
#' @param voxel_coords List specifying voxels for each region. Each item
#'   in the list is a \eqn{L_j \times 3} matrix of spatial coordinates.
#' @param kernel_type Choice of spatial kernel.
#' @param cov_setting Choice of covariance structure.
#' @param out_file Output file path.
#' @param overwrite Overwrite output file if it exists.
#' @param num_init Number of initializations for multi-start optimiization.
#' @param verbose Print progress messages.
#'
#' @useDynLib qfuncMM
#' @importFrom Rcpp sourceCpp
#' @importFrom jsonlite toJSON
#' @export
qfuncMM_stage1_intra <- function(region_list, voxel_coords,
                                 kernel_type = "matern_5_2",
                                 cov_setting = c("standard", "diag_time", "noiseless", "noiseless_profiled"),
                                 out_file = NULL,
                                 overwrite = FALSE,
                                 num_init = 10L,
                                 verbose = TRUE) {
  start_time <- Sys.time()
  if (is.null(out_file)) {
    out_file <- file.path(".", sprintf("qfuncMM_stage1_intra_%s.json", format(start_time, "%Y%m%d_%H%M%S")))
  }
  out_dir <- dirname(out_file)
  if (!file.info(out_dir)$isdir || file.access(out_dir, mode = 2) != 0) {
    stop(sprintf("The specified output location '%s' is not in a valid, writeable directory.", out_file))
  }
  if (file.exists(out_file) && !overwrite) {
    stop(sprintf("File '%s' already exists. Set overwrite = TRUE to overwrite.", out_file))
  }

  kernel_type_id <- kernel_dict(kernel_type)
  cov_setting <- match.arg(cov_setting)

  # TODO: split validation into a separate function
  if (length(region_list) == 0) {
    stop("Must specify at least one region.")
  }
  if (length(region_list) != length(voxel_coords)) {
    stop("Length of region_list and region_coords must be the same.")
  }

  n_region <- length(region_list)
  n_timept <- nrow(region_list[[1]])
  n_voxel <- vector(length = n_region, mode = "integer")
  for (i in seq_along(region_list)) {
    if (!is.matrix(region_list[[i]])) {
      stop(sprintf("Region %d is not a matrix.", i))
    }
    if (!is.matrix(voxel_coords[[i]])) {
      stop(sprintf("Region coordinates %d is not a matrix.", i))
    }
    if (!is.numeric(voxel_coords[[i]]) || ncol(voxel_coords[[i]]) != 3) {
      stop(sprintf("Region %d: voxels must have three numeric coordinates.", i))
    }
    if (nrow(region_list[[i]]) != n_timept) {
      stop(sprintf("Region %d: inconsistent number of time points (rows)", i))
    }
    n_voxel[i] <- ncol(region_list[[i]])
    if (n_voxel[i] != nrow(voxel_coords[[i]])) {
      stop(sprintf("Region %d: Inconsistent number of voxels (columns)", i))
    }
  }

  message(sprintf(
    "Running QFunCMM stage 1 intra-regional for %d regions and %d time points using %d initializations per region.",
    n_region, n_timept, num_init
  ))

  outlist <- vector("list", length = n_region + 1)
  names(outlist) <- c("info", paste0("r", seq_len(n_region)))
  outlist[["info"]] <- list(
    n_region = n_region,
    n_timept = n_timept,
    kernel_type = kernel_type,
    cov_setting = cov_setting,
    num_init = num_init,
    timestamp = format(start_time, "%Y-%m-%dT%H:%M:%OS3Z")
  )

  # Standardize the data matrices
  region_list_std <- lapply(region_list, \(reg) (reg - mean(reg)) / stats::sd(reg))

  for (regid in seq_along(region_list_std)) {
    intra <- fit_intra_model(
      region_list_std[[regid]],
      voxel_coords[[regid]],
      kernel_type_id,
      cov_setting,
      num_init
    )

    outlist[[paste0("r", regid)]] <- c(
      as.list(intra$intra_param),
      list(eblue = as.numeric(intra$eblue))
    )
  }

  out_json <- jsonlite::toJSON(outlist, auto_unbox = TRUE, pretty = TRUE, digits = I(10))
  write(out_json, out_file)
  message("Finished stage 1 intra-regional for all regions.\nResults saved to ", normalizePath(out_file))
}
