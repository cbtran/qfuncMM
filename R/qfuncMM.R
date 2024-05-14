#' Estimate functional connectivity from voxel-level BOLD signals.
#'
#' @param region_list List of \eqn{M\times L_j} region matrices.
#' @param voxel_coords List specifying voxels for each region. Each item
#'   in the list is a \eqn{L_j \times 3} matrix of spatial coordinates.
#' @param kernel_type Choice of spatial kernel. Default "matern_5_2".
#' @param verbose Print progress messages.
#'
#' @useDynLib qfuncMM
#' @importFrom Rcpp sourceCpp
#' @export
qfuncMM <- function(region_list, voxel_coords,
                    kernel_type = "matern_5_2", verbose = TRUE) {
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
    cat("Running QFunCMM with", n_region, "regions and",
        n_timept, "time points.\n")
  }

  time_sqrd_mat <- outer(seq_len(n_timept), seq_len(n_timept), `-`)^2

  stage1_regional <- matrix(nrow = n_region, ncol = 5,
                            dimnames = list(paste0("r", seq_len(n_region)),
                                            c("phi_gamma", "tau_gamma",
                                              "k_gamma", "nugget_gamma",
                                              "var_noise")))
  stage1_eblue <- matrix(nrow = n_region, ncol = n_timept)

  if (verbose) {
    cat("Stage 1: estimating intra-regional parameters...\n")
  }

  for (regid in seq_along(region_list)) {
    intra <- fit_intra_model(region_list[[regid]],
                             voxel_coords[[regid]],
                             kernel_type_id,
                             time_sqrd_mat)

    stage1_regional[regid, ] <- intra$intra_param
    stage1_eblue[regid, ] <- intra$eblue
  }

  if (verbose) {
    cat("Finished stage 1.\n")
  }

  # Matrix of asymptotic variances for region pairs
  if (verbose) {
    cat("Stage 2: estimating inter-regional correlations...\n")
  }

  cor_mx <- matrix(1, nrow = n_region, ncol = n_region,
                   dimnames = list(paste0("r", seq_len(n_region)),
                                   paste0("r", seq_len(n_region))))
  stage2_inter <- array(dim = c(n_region, n_region, 4),
                        dimnames = list(paste0("r", seq_len(n_region)),
                                        paste0("r", seq_len(n_region)),
                                        c("k_eta1", "k_eta2",
                                          "tau_eta", "nugget_eta")))
  stage2_eblue <- matrix(1, nrow = n_region, ncol = n_region)
  rho_ca <- matrix(1, nrow = n_region, ncol = n_region)

  # Run stage 2 for each pair of regions
  for (reg1 in seq_len(n_region - 1)) {
    for (reg2 in seq(reg1 + 1, n_region)) {
      stage2_eblue[reg1, reg2] <- cor(stage1_eblue[reg1, ], stage1_eblue[reg2, ]) * (n_timept - 1) / n_timept
      stage2_eblue[reg2, reg1] <- stage2_eblue[reg1, reg2]

      stage2_result <- fit_inter_model(region_list[[reg1]],
                                       voxel_coords[[reg1]],
                                       region_list[[reg2]],
                                       voxel_coords[[reg2]],
                                       time_sqrd_mat,
                                       stage1_regional[reg1, ],
                                       stage1_regional[reg2, ],
                                       kernel_type_id)
      cor_mx[reg1, reg2] <- stage2_result$params["rho"]
      cor_mx[reg2, reg1] <- stage2_result$params["rho"]
      stage2_inter[reg1, reg2, ] <- stage2_result$params[-1]
      stage2_inter[reg2, reg1, ] <- stage2_inter[reg1, reg2, ]

      rho_ca[reg1, reg2] <- stage2_result$rho_ca
      rho_ca[reg2, reg1] <- stage2_result$rho_ca
      cat("Finished region pair", reg1, "-", reg2, "\n")
    }
  }

  if (verbose) {
    cat("Finished stage 2.\n")
  }

  list(rho = cor_mx, rho_eblue = stage2_eblue, rho_ca = rho_ca, stage1 = stage1_regional, stage2 = stage2_inter)
}
