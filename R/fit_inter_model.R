#' Stage 2: Fit inter-regional model given pair of regions
#' @param parameters_init unrestricted initialization of parameters \eqn{\rho, \tau_\eta, k_\eta, nugget}
#' @param region1_mx Data matrix of signals of region 1
#' @param voxel_coords_1 coordinates of voxels in region 1
#' @param region2_mx Data matrix of signals of region 2
#' @param voxel_coords_2 coordinates of voxels in region 2
#' @param time_sqrd_mat Temporal squared distance matrix
#' @param stage1_regional Estimated parameters of intra-regional models from 2 regions
#' @param kernel_type_id Choice of spatial kernel
#' @return List of 2 components:
#' \item{theta}{estimated inter-regional parameters \eqn{\hat{\rho}, \hat{\tau}_\eta, \hat{k}_\eta, \hat{nugget}, \hat{\mu}_1, \hat{\mu}_2}}
#' \item{asymptotic_var}{asymptotic variance of transformed correlation coefficient}
#' \item{rho_transformed}{Fisher transformation of correlation coefficient}
#' @noRd
fit_inter_model <- function(
    region1_mx,
    voxel_coords_1,
    region2_mx,
    voxel_coords_2,
    time_sqrd_mat,
    stage1_regional,
    kernel_type_id) {

  num_timept <- nrow(region2_mx)

  # TODO: Move this inside opt_inter. Better yet, get rid of it.
  # All this does is "select" the signal for each region.
  fixedfx_design <- rbind(
    matrix(rep(1:0, each = ncol(region1_mx) * num_timept), ncol = 2),
    matrix(rep(0:1, each = ncol(region2_mx) * num_timept), ncol = 2)
  )

  init <- c(0, 0, 0, -2, mean(region1_mx), mean(region2_mx))

  result <- opt_inter(
    theta_init = init,
    X = matrix(c(region1_mx, region2_mx), ncol = 1),
    Z = fixedfx_design,
    voxel_coords_1 = voxel_coords_1,
    voxel_coords_2 = voxel_coords_2,
    time_sqrd_mat = time_sqrd_mat,
    stage1_regional = stage1_regional,
    kernel_type_id = kernel_type_id)

  result
}
