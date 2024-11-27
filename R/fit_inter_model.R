#' Stage 2: Fit inter-regional model given pair of regions
#'
#' @noRd

fit_inter_model <- function(region1_info, region2_info, kernel_type_id, init, verbose) {
  softminus <- function(x) {
    ifelse(x > 10, x, log(exp(x) - 1))
  }

  logit <- function(x, lower, upper) {
    x <- (x - lower) / (upper - lower)
    return(log(x / (1 - x)))
  }

  m <- length(region1_info$eblue)
  time_sqrd_mat <- outer(seq_len(m), seq_len(m), `-`)^2

  init <- c(
    logit(init["rho"], -1, 1),
    softminus(init[get("stage2_paramlist_components", qfuncMM_pkg_env)])
  )
  if (region1_info$cov_setting == "diag_time" || region2_info$cov_setting == "diag_time") {
    init <- init[1:3]
  }

  result <- opt_inter(
    theta_init = init,
    data_r1 = region1_info$data_std,
    data_r2 = region2_info$data_std,
    coords_r1 = region1_info$coords,
    coords_r2 = region2_info$coords,
    time_sqrd_mat = time_sqrd_mat,
    stage1_r1 = unlist(region1_info$stage1),
    stage1_r2 = unlist(region2_info$stage1),
    cov_setting_id1 = cov_setting_dict(region1_info$cov_setting),
    cov_setting_id2 = cov_setting_dict(region2_info$cov_setting),
    kernel_type_id = kernel_type_id,
    verbose = verbose
  )

  theta <- result$theta
  names(theta) <- get("stage2_paramlist", qfuncMM_pkg_env)
  return(list(theta = theta, objval = result$objval))
}
