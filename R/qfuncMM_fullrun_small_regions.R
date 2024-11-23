#' Run both stage 1 and stage 2 for QFuncMM. This is a convenience function for small brain regions and testing.
#' Larger applications should use `qfuncMM_stage1_intra` and `qfuncMM_stage2_inter` separately.
#'
#' @param region_list List of \eqn{M\times L_j} region matrices.
#' @param voxel_coords List specifying voxels for each region. Each item
#'   in the list is a \eqn{L_j \times 3} matrix of spatial coordinates.
#' @param kernel_type Choice of spatial kernel.
#' @param stage1_cov_setting Choice of noisy or noiseless stage 1 model.
#' @param num_init Number of stage 1 initializations to use.
#' @param verbose Print progress messages.
#'
#' @useDynLib qfuncMM
#' @importFrom Rcpp sourceCpp
#' @importFrom stats cor dist
#' @export
qfuncMM_fullrun_small_regions <- function(region_list, voxel_coords,
                                          kernel_type = "matern_5_2",
                                          stage1_cov_setting = c("noisy", "noiseless"),
                                          num_init = 10L,
                                          verbose = FALSE) {
  kernel_type_id <- kernel_dict(kernel_type)
  stage1_cov_setting <- match.arg(stage1_cov_setting)

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

  message("Running QFunCMM with ", n_region, " regions and ", n_timept, " time points.")
  message("Stage 1: estimating intra-regional parameters...")

  # Standardize the data matrices
  # TODO: Should we keep the raw data matrices around?
  region_list_std <- lapply(region_list, \(reg) (reg - mean(reg)) / stats::sd(reg))
  stage1_info <- vector("list", length = n_region)

  for (regid in seq_along(region_list_std)) {
    inits <- stage1_init(region_list_std[[regid]], voxel_coords[[regid]], num_init, FALSE)
    intra_out <- fit_intra_model(
      region_list_std[[regid]], voxel_coords[[regid]], inits, kernel_type_id, stage1_cov_setting, verbose
    )

    stage1_region_info <- list()
    stage1_region_info$stage1 <- intra_out$intra_param
    stage1_region_info$eblue <- intra_out$eblue
    stage1_region_info$data_std <- region_list_std[[regid]]
    stage1_region_info$coords <- voxel_coords[[regid]]
    stage1_region_info$cov_setting <- stage1_cov_setting
    stage1_info[[regid]] <- stage1_region_info
  }

  if (verbose) {
    message("Finished stage 1.\nStage 2: estimating inter-regional correlations...")
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
      list(get("stage2_paramlist_components", qfuncMM_pkg_env))
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
        stage1_info[[reg1]], stage1_info[[reg2]], kernel_type_id, eblue_r12, verbose
      )
      theta <- stage2_result$theta
      rho[reg1, reg2] <- theta["rho"]
      rho[reg2, reg1] <- theta["rho"]
      stage2_inter[reg1, reg2, ] <- theta[get("stage2_paramlist_components", qfuncMM_pkg_env)]
      stage2_inter[reg2, reg1, ] <- stage2_inter[reg1, reg2, ]
      message("Finished region pair ", reg1, " - ", reg2, "\n")
    }
  }

  message("Finished stage 2.")
  stage1_regional <- do.call(rbind, lapply(stage1_info, \(x) x$stage1))
  list(rho = rho, rho_eblue = rho_eblue, rho_ca = rho_ca, stage1 = stage1_regional, stage2 = stage2_inter)
}
