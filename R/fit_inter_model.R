#' Stage 2: Fit inter-regional model given pair of regions
#'
#' @param region1_mx Data matrix of signals of region 1
#' @param voxel_coords_1 coordinates of voxels in region 1
#' @param region2_mx Data matrix of signals of region 2
#' @param voxel_coords_2 coordinates of voxels in region 2
#' @param time_sqrd_mat Temporal squared distance matrix
#' @param region1_stage1, Estimated stage 1 parameters for region 1
#' @param region2_stage1, Estimated stage 1 parameters for region 2
#' @param rho_eblue, EBLUE from stage 1 for initialization
#' @param kernel_type_id Choice of spatial kernel
#' @param cov_setting Choice of covariance structure
#' @return Estimated stage 2 parameters
#' @noRd

fit_inter_model <- function(
    region1_mx,
    voxel_coords_1,
    region2_mx,
    voxel_coords_2,
    time_sqrd_mat,
    region1_stage1,
    region2_stage1,
    rho_eblue,
    kernel_type_id,
    cov_setting) {
  # Parameter list: rho, kEta1, kEta2, tauEta, nugget
  softminus <- function(x) {
    log(exp(x) - 1)
  }

  logit <- function(x, lower, upper) {
    x <- (x - lower) / (upper - lower)
    return(log(x) - log(1 - x))
  }

  # Use the EBLUE as a reasonable initialization.
  init <- c(logit(rho_eblue, -1, 1), softminus(1), softminus(1), 0, softminus(0.1))
  if (cov_setting == "diag_time") {
    init <- c(logit(rho_eblue, -1, 1), softminus(1), softminus(1))
  }

  result <- opt_inter(
    theta_init = init,
    dataRegion1 = region1_mx,
    dataRegion2 = region2_mx,
    voxel_coords_1 = voxel_coords_1,
    voxel_coords_2 = voxel_coords_2,
    time_sqrd_mat = time_sqrd_mat,
    stage1ParamsRegion1 = region1_stage1,
    stage1ParamsRegion2 = region2_stage1,
    kernel_type_id = kernel_type_id,
    setting = cov_setting
  )

  params <- result$theta
  names(params) <- c("rho", "k_eta1", "k_eta2", "tau_eta", "nugget_eta")
  return(params = params)
}
