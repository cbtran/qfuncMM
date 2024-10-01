#' Stage 2: Fit inter-regional model given pair of regions

fit_inter_model <- function(
    region1_info,
    region2_info,
    time_sqrd_mat,
    kernel_type_id,
    rho_init) {
  softminus <- function(x) {
    log(exp(x) - 1)
  }

  logit <- function(x, lower, upper) {
    x <- (x - lower) / (upper - lower)
    return(log(x) - log(1 - x))
  }

  # Use the EBLUE as a reasonable initialization.
  init <- c(logit(rho_init, -1, 1), softminus(1), softminus(1), 0, softminus(0.1))
  if (region1_info$cov_setting == "diag_time" || region2_info$cov_setting == "diag_time") {
    init <- c(logit(rho_init, -1, 1), softminus(1), softminus(1))
  }

  result <- opt_inter(
    theta_init = init,
    dataRegion1 = region1_info$data,
    dataRegion2 = region2_info$data,
    voxel_coords_1 = region1_info$coords,
    voxel_coords_2 = region2_info$coords,
    time_sqrd_mat = time_sqrd_mat,
    stage1ParamsRegion1 = region1_info$intra_param,
    stage1ParamsRegion2 = region2_info$intra_param,
    cov_setting_id1 = cov_setting_dict(region1_info$cov_setting),
    cov_setting_id2 = cov_setting_dict(region2_info$cov_setting),
    kernel_type_id = kernel_type_id
  )

  params <- result$theta
  names(params) <- c("rho", "k_eta1", "k_eta2", "tau_eta", "nugget_eta")
  return(params = params)
}
