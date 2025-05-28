#' Compute Asymptotic Confidence Interval for Rho
#'
#' Get Asymptotic Confidence Interval for Rho Parameter
#'
#' Computes asymptotic confidence intervals for the rho parameter using
#' asymptotic variance estimates. The function supports different methods
#' for variance calculation based on provided region information.
#'
#' @param theta Named numeric vector containing "rho".
#' @param level The confidence level for the interval (e.g., 0.95 for
#'   95\% confidence interval).
#' @param asympvar_rho Numeric value or NULL. Pre-computed asymptotic variance
#'   for rho. If NULL, variance will be computed using region information.
#' @param region1_info List or NULL. Information about the first region used
#'   for variance calculation when asympvar_rho is NULL.
#' @param region2_info List or NULL. Information about the second region used
#'   for variance calculation when asympvar_rho is NULL.
#'
#' @return A numeric vector containing the lower and upper bounds of the
#'   confidence interval for the rho parameter. The confidence interval is computed
#'  on the Fisher Z scale using the delta method and then transformed back to the correlation scale.
#'
#' @seealso \code{\link{get_asymp_var_rho}} for computing asymptotic variance
#' @export
get_asymp_ci_rho <- function(theta, level, asympvar_rho = NULL, region1_info = NULL, region2_info = NULL) {
  if (is.null(asympvar_rho)) {
    if (is.null(region1_info) || is.null(region2_info)) {
      stop("Either 'asympvar_rho' must be provided or 'region1_info' and 'region2_info' must be specified.")
    }
    asympvar_rho <- get_asymp_var_rho(theta, region1_info, region2_info)
  }
  rho <- theta["rho"]
  if (is.na(rho) || rho < -1 || rho > 1) {
    stop("Invalid or missing named value for 'rho'. It must be between -1 and 1.")
  }
  z_rho <- fisher_z(rho)
  intlen <- stats::qnorm(1 - (1 - level) / 2) * d_fisher_z(rho) * sqrt(asympvar_rho)
  ci <- inv_fisher_z(c(z_rho - intlen, z_rho + intlen))
  names(ci) <- c("lower", "upper")
  return(ci)
}


#' Get Asymptotic Variance for Rho
#'
#' Computes the asymptotic variance for rho based on the Fisher information matrix.
#'
#' @param theta Numeric. The estimated stage 2 parameter vector
#' @param region1_info Stage 1 info for region 1.
#' @param region2_info Stage 1 info for region 2.
#' @param method Which method was used in stage 2? Either "reml" or "vecchia".
#'
#' @return A scalar value representing the asymptotic variance.
#'
#' @export
get_asymp_var_rho <- function(theta, region1_info, region2_info, method = c("reml", "vecchia")) {
  method <- match.arg(method)
  m <- region1_info$num_timepoints
  time_sqrd_mat <- outer(seq_len(m), seq_len(m), `-`)^2

  fisher_info_mx <- get_fisher_info(
    theta = theta, # paramlist: "rho", "k_eta1", "k_eta2", "tau_eta", "nugget_eta"
    data_r1 = region1_info$data_std,
    data_r2 = region2_info$data_std,
    coords_r1 = region1_info$coords,
    coords_r2 = region2_info$coords,
    time_sqrd_mat = time_sqrd_mat,
    stage1_r1 = unlist(region1_info$stage1),
    stage1_r2 = unlist(region2_info$stage1),
    cov_setting_id1 = cov_setting_dict(region1_info$cov_setting),
    cov_setting_id2 = cov_setting_dict(region2_info$cov_setting),
    kernel_type_id = kernel_dict("matern_5_2"),
    reml = method == "reml"
  )

  inv_fisher_info <- solve(fisher_info_mx)
  asympvar_rho <- inv_fisher_info["rho", "rho"]
  return(asympvar_rho)
}
