#' Stage 1: Fit intra-regional model for a single region
#'
#' @param region_mx Data matrix of signals of 1 region
#' @param voxel_coords coordinates of voxels in the region
#' @param num_basis Number of B-spline basis.
#' @param time_sqrd_mat Temporal squared distance matrix
#' @param kernel_type_id Choice of spatial kernel. Defaul "matern_5_2".
#' @return List of 2 components:
#' \item{theta}{estimated intra-regional parameters \eqn{\hat{\phi}_\gamma, \hat{\tau}_\gamma, \hat{k}_\gamma}}
#' \item{nu}{fixed-effect estimate \eqn{\hat{\nu}}}
#'
#' @noRd

fit_intra_model <- function(
    region_mx,
    voxel_coords,
    num_basis,
    kernel_type_id,
    time_sqrd_mat) {

  num_timept <- nrow(region_mx)
  num_voxel <- ncol(region_mx)

  regiondf <- data.frame(
    signal = as.numeric(region_mx),
    time = rep(seq_len(num_timept), times = num_voxel))

  regionfit <- stats::lm(
    signal ~ -1 + splines::bs(time, df = num_basis, intercept = TRUE),
    data = regiondf)

  # Used for comparison only
  bspline_pred <- stats::predict(
    regionfit, data.frame(time = seq_len(num_timept)))

  bspline_design <-
    splines::bs(
      rep(seq_len(num_timept), num_voxel), df = num_basis, intercept = TRUE)
  param_init <- c(0, 0, 0, stats::coef(regionfit))

  intra <- opt_intra(
    param_init, matrix(region_mx, ncol = 1), bspline_design,
    voxel_coords, time_sqrd_mat, num_voxel, num_timept, kernel_type_id)

  list(intra_param = intra$theta,
       fixed = bspline_design[seq_len(num_timept), ] %*% intra$nu,
       bspline_pred = bspline_pred)
}