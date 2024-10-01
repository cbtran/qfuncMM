#' Estimate functional connectivity from voxel-level BOLD signals.
#'
#' @param region_list List of \eqn{M\times L_j} region matrices.
#' @param voxel_coords List specifying voxels for each region. Each item
#'   in the list is a \eqn{L_j \times 3} matrix of spatial coordinates.
#' @param kernel_type Choice of spatial kernel. Default "matern_5_2".
#' @param cov_setting Choice of covariance structure.
#' @param verbose Print progress messages.
#'
#' @useDynLib qfuncMM
#' @importFrom Rcpp sourceCpp
#' @importFrom stats cor dist
#' @export
qfuncMM <- function(region_list, voxel_coords,
                    kernel_type = "matern_5_2",
                    cov_setting = c("standard", "diag_time", "noiseless", "noiseless_profiled"),
                    verbose = TRUE) {
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

  if (verbose) {
    message(
      "Running QFunCMM with ", n_region, " regions and ",
      n_timept, " time points."
    )
  }

  time_sqrd_mat <- outer(seq_len(n_timept), seq_len(n_timept), `-`)^2

  # stage1_regional <- matrix(
  #   nrow = n_region, ncol = 5,
  #   dimnames = list(
  #     paste0("r", seq_len(n_region)),
  #     c(
  #       "phi_gamma", "tau_gamma",
  #       "k_gamma", "nugget_gamma",
  #       "var_noise"
  #     )
  #   )
  # )
  # stage1_eblue <- matrix(nrow = n_region, ncol = n_timept)

  if (verbose) {
    message("Stage 1: estimating intra-regional parameters...")
  }

  # Standardize the data matrices
  # TODO: Should we keep the raw data matrices around?
  region_list_std <- lapply(region_list, \(reg) (reg - mean(reg)) / stats::sd(reg))
  stage1_info <- vector("list", length = n_region)

  for (regid in seq_along(region_list_std)) {
    intra_out <- fit_intra_model(
      region_list_std[[regid]],
      voxel_coords[[regid]],
      kernel_type_id,
      cov_setting,
      num_init = 10
    )

    stage1_region_info <- list()
    stage1_region_info$intra_param <- intra_out$intra_param
    stage1_region_info$eblue <- intra_out$eblue
    stage1_region_info$data <- region_list_std[[regid]]
    stage1_region_info$coords <- voxel_coords[[regid]]
    stage1_region_info$cov_setting <- cov_setting
    stage1_info[[regid]] <- stage1_region_info
  }

  if (verbose) {
    message("Finished stage 1.\n")
  }

  # Matrix of asymptotic variances for region pairs
  if (verbose) {
    message("Stage 2: estimating inter-regional correlations...")
  }

  region_dimnames <- list(
    paste0("r", seq_len(n_region)),
    paste0("r", seq_len(n_region))
  )
  rho <- matrix(1,
    nrow = n_region, ncol = n_region,
    dimnames = list(
      paste0("r", seq_len(n_region)),
      paste0("r", seq_len(n_region))
    )
  )
  stage2_inter <- array(
    dim = c(n_region, n_region, 4),
    dimnames = c(
      region_dimnames,
      list(c("k_eta1", "k_eta2", "tau_eta", "nugget_eta"))
    )
  )
  rho_eblue <- matrix(1,
    nrow = n_region, ncol = n_region,
    dimnames = region_dimnames
  )
  rho_ca <- matrix(1,
    nrow = n_region, ncol = n_region,
    dimnames = region_dimnames
  )

  # Run stage 2 for each pair of regions
  for (reg1 in seq_len(n_region - 1)) {
    for (reg2 in seq(reg1 + 1, n_region)) {
      eblue_r12 <- stats::cor(stage1_info[[reg1]]$eblue, stage1_info[[reg2]]$eblue)
      rho_eblue[reg1, reg2] <- eblue_r12
      rho_eblue[reg2, reg1] <- eblue_r12

      ca <- cor(rowMeans(stage1_info[[reg1]]$data), rowMeans(stage1_info[[reg2]]$data))
      rho_ca[reg1, reg2] <- ca
      rho_ca[reg2, reg1] <- ca

      stage2_result <- fit_inter_model(
        stage1_info[[reg1]], stage1_info[[reg2]], time_sqrd_mat, kernel_type_id, eblue_r12
      )
      rho[reg1, reg2] <- stage2_result["rho"]
      rho[reg2, reg1] <- stage2_result["rho"]
      stage2_inter[reg1, reg2, ] <- stage2_result[-1]
      stage2_inter[reg2, reg1, ] <- stage2_inter[reg1, reg2, ]
      message("Finished region pair ", reg1, " - ", reg2, "\n")
    }
  }

  if (verbose) {
    message("Finished stage 2.")
  }
  stage1_regional <- do.call(rbind, lapply(stage1_info, \(x) x$intra_param))
  list(rho = rho, rho_eblue = rho_eblue, rho_ca = rho_ca, stage1 = stage1_regional, stage2 = stage2_inter)
}
