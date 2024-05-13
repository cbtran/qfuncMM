#' Stage 1: Fit intra-regional model for a single region
#'
#' @param region_mx Data matrix of signals of 1 region
#' @param voxel_coords coordinates of voxels in the region
#' @param time_sqrd_mat Temporal squared distance matrix
#' @param kernel_type_id Choice of spatial kernel. Default "matern_5_2".
#' @return Esimated intra parameters and noise variance
#'
#' @noRd

fit_intra_model <- function(
    region_mx,
    voxel_coords,
    kernel_type_id,
    time_sqrd_mat) {

  # Param list: phi, tau_gamma, k_gamma, nugget_gamma
  param_init <- c(0, 0, 0, 0)

  intra <- opt_intra(param_init, matrix(region_mx, ncol = 1),
                     voxel_coords, time_sqrd_mat, kernel_type_id)

  list(intra_param = c(intra$theta, intra$var_noise), eblue = intra$eblue)
}