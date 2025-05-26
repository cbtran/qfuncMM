#' Get Asymptotic Confidence Interval for Rho Parameter
#'
#' Computes asymptotic confidence intervals for the rho (correlation) parameter
#' using region-specific information matrices.
#'
#' @param theta Numeric. The estimated stage 2 parameter vector
#' @param level Numeric. Confidence level (e.g., 0.95 for 95\% confidence interval)
#' @param region1_info Stage 1 info for region 1.
#' @param region2_info Stage 1 info for region 2.
#'
#' @return A numeric vector containing the lower and upper bounds of the
#'   confidence interval for the rho parameter
#'
#' @export
get_asymp_ci_rho <- function(theta, level, region1_info, region2_info) {
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
    kernel_type_id = kernel_dict("matern_5_2")
  )

  rho <- theta["rho"]
  inv_fisher_info <- solve(fisher_info_mx)
  asympvar_rho <- inv_fisher_info["rho", "rho"]

  z_rho <- fisher_z(rho)
  intlen <- stats::qnorm(1 - (1 - level) / 2) * d_fisher_z(rho) * sqrt(asympvar_rho)
  ci <- inv_fisher_z(c(z_rho - intlen, z_rho + intlen))
  names(ci) <- c("lower", "upper")
  return(ci)
}
