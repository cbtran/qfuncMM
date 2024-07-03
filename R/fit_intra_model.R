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
    kernel_type_id = 3L,
    # time_sqrd_mat,
    nugget_only = FALSE,
    init = c(0, 0, 0, 0)) {
  # Param list: phi, tau_gamma, k_gamma, nugget_gamma
  param_init <- init
  if (nugget_only) {
    param_init <- c(0, 0)
  }

  m <- nrow(region_mx)
  time_sqrd_mat <- outer(seq_len(m), seq_len(m), `-`)^2

  intra <- opt_intra(
    param_init, matrix(region_mx, ncol = 1),
    voxel_coords, time_sqrd_mat, kernel_type_id, nugget_only
  )

  intra_param <- c(intra$theta, intra$var_noise)
  if (nugget_only) {
    intra_param <- c(intra_param[1], 1, 0, intra_param[2], intra$var_noise)
  }
  names(intra_param) <- c("phi", "tau_gamma", "k_gamma", "nugget_gamma", "var_noise")
  list(intra_param = intra_param, eblue = intra$eblue, objval = intra$objval)
}
