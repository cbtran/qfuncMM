#' Fit the mixed model
#'
#' @param region_list List of region matrices.
#'   Each region matrix contains signals of the voxels in that region for each
#'   time point. The number of rows is the number of time points and the number
#'   of columns is the number of voxels.
#' @param voxel_coords List of region coordinates. Each row is a voxel and
#'   the three columns are x, y, and z coordinates.
#' @param n_basis Number of B-spline basis.
#' @param kernel_type Choice of spatial kernel. Default "matern_5_2".
#' @param verbose Print progress messages.
#'
#' @useDynLib qfuncMM
#' @importFrom splines bs
#' @importFrom Rcpp sourceCpp
#' @export
qfuncmm <- function(region_list, voxel_coords,
                  n_basis = 45, kernel_type = "matern_5_2", verbose = TRUE) {
  kernel_type_id <- kernel_dict(kernel_type)

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
      stop(sprintf("Region %d: Inconsistent number of voxels (cols)", i))
    }
  }

  if (verbose) {
    cat("Running Qfunc with", n_region, "regions and",
      n_timept, "time points.\n")
  }

  time_sqrd_mat <- outer(seq_len(n_timept), seq_len(n_timept), "-")^2

  stage1_regional <- matrix(nrow = n_region, ncol = 3)
  stage1_fixed <- vector(length = n_region, mode = "list")
  stage1_bspline <- vector(length = n_region, mode = "list")

  if (verbose) {
    cat("Stage 1: estimating intra-regional parameters...\n")
  }

  for (regid in seq_along(region_list)) {
    intra <- fit_intra_model(
      region_list[[regid]], voxel_coords[[regid]],
      n_basis, kernel_type_id, time_sqrd_mat)

    stage1_regional[regid, ] <- intra$intra_param
    stage1_fixed[[regid]] <- intra$fixed
    stage1_bspline[[regid]] <- intra$bspline_pred
  }

  if (verbose) {
    cat("Finished stage 1.\n")
  }

  # Result matrix of correlations between regions
  qfunc_result <- matrix(0, nrow = n_region, ncol = n_region)
  diag(qfunc_result) <- 1

  # Matrix of asymptotic variances for region pairs
  asymp_var <- matrix(0, nrow = n_region, ncol = n_region)
  diag(asymp_var) <- 1

  ## These correlations are for comparison only.
  # Correlation between stage 1 fixed effects
  stage1_fixed_cor <- matrix(0, nrow = n_region, ncol = n_region)
  diag(stage1_fixed_cor) <- 1

  # Correlation between stage 1 B-spline predictions
  stage1_bspline_cor <- matrix(0, nrow = n_region, ncol = n_region)
  diag(stage1_bspline_cor) <- 1

  if (verbose) {
    cat("Stage 2: estimating inter-regional correlations...\n")
  }

  # Run stage 2 for each pair of regions
  for (reg1 in seq_len(n_region)) {
    for (reg2 in seq_len(reg1 - 1)) {
      fixed_cor <- stats::cor(stage1_fixed[[reg1]], stage1_fixed[[reg2]])
      stage1_fixed_cor[reg1, reg2] <- fixed_cor
      stage1_fixed_cor[reg2, reg1] <- fixed_cor

      bspline_cor <- stats::cor(stage1_bspline[[reg1]], stage1_bspline[[reg2]])
      stage1_bspline_cor[reg1, reg2] <- bspline_cor
      stage1_bspline_cor[reg2, reg1] <- bspline_cor

      stage2_result <- fit_inter_model(
        region_list[[reg1]], voxel_coords[[reg1]],
        region_list[[reg2]], voxel_coords[[reg2]],
        time_sqrd_mat,
        c(stage1_regional[reg1, ], stage1_regional[reg2, ]),
        kernel_type_id)

      qfunc_result[reg1, reg2] <- stage2_result$theta[1]
      qfunc_result[reg2, reg1] <- stage2_result$theta[1]
      asymp_var[reg1, reg2] <- stage2_result$asymptotic_var
      asymp_var[reg2, reg1] <- stage2_result$asymptotic_var
    }
  }

  if (verbose) {
    cat("Finished stage 2.\n")
  }

  list(
    cor = qfunc_result,
    asymp_var = asymp_var,
    cor_fixed_intra = stage1_fixed_cor,
    cor_fixed_bspline = stage1_bspline_cor
  )
}
