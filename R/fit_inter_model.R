#' Stage 2: Fit inter-regional model given pair of regions
#'
#' @noRd

fit_inter_model <- function(region1_info, region2_info, kernel_type_id, rho_init, verbose) {
  m <- length(region1_info$eblue)
  time_sqrd_mat <- outer(seq_len(m), seq_len(m), `-`)^2

  # Use the EBLUE as a reasonable initialization.
  init <- c(rho_init, 1, 1, 0.5, 0.1)
  if (region1_info$cov_setting == "diag_time" || region2_info$cov_setting == "diag_time") {
    init <- c(logit(rho_init, -1, 1), softminus(1), softminus(1))
  }

  r1_distmat <- as.matrix(stats::dist(region1_info$coords))
  c1 <- matern(region1_info$stage1$phi, r1_distmat)
  b1 <- region1_info$stage1$k_gamma * rbf(region1_info$stage1$tau_gamma, time_sqrd_mat)
  diag(b1) <- diag(b1) + region1_info$stage1$nugget_gamma
  lambda1 <- kronecker(c1, b1)

  r2_distmat <- as.matrix(stats::dist(region2_info$coords))
  c2 <- matern(region2_info$stage1$phi, r2_distmat)
  b2 <- region2_info$stage1$k_gamma * rbf(region2_info$stage1$tau_gamma, time_sqrd_mat)
  diag(b2) <- diag(b2) + region2_info$stage1$nugget_gamma
  lambda2 <- kronecker(c2, b2)

  # tictoc::tic("Stage 2 Inter")
  # eval <- stage2_inter_nll(
  #   init, region1_info$data_std, region2_info$data_std,
  #   region1_info$coords, region2_info$coords, time_sqrd_mat,
  #   unlist(region1_info$stage1), unlist(region2_info$stage1),
  #   lambda1, lambda2, 2L, 2L, 3L
  # )
  # tictoc::toc()

  tictoc::tic("Stage 2 Inter opt")
  result <- minqa::bobyqa(init, stage2_inter_nll,
    lower = c(-1, 0, 0, 0, 0), upper = c(1, Inf, Inf, Inf, Inf),
    region1 = region1_info$data_std, region2 = region2_info$data_std,
    r1_coords = region1_info$coords, r2_coords = region2_info$coords,
    time_sqrd_mat = time_sqrd_mat, r1_stage1 = unlist(region1_info$stage1),
    r2_stage1 = unlist(region2_info$stage1),
    lambda1 = lambda1, lambda2 = lambda2,
    cov_setting_id1 = cov_setting_dict(region1_info$cov_setting),
    cov_setting_id2 = cov_setting_dict(region2_info$cov_setting),
    kernel_type_id = 3L,
    control = list(npt = 9, iprint = 3, rhobeg = 0.2, rhoend = 1e-6, maxfun = 250)
  )
  tictoc::toc()

  result <- opt_inter(
    theta_init = init,
    dataRegion1 = region1_info$data_std,
    dataRegion2 = region2_info$data_std,
    voxel_coords_1 = region1_info$coords,
    voxel_coords_2 = region2_info$coords,
    time_sqrd_mat = time_sqrd_mat,
    stage1ParamsRegion1 = unlist(region1_info$stage1),
    stage1ParamsRegion2 = unlist(region2_info$stage1),
    cov_setting_id1 = cov_setting_dict(region1_info$cov_setting),
    cov_setting_id2 = cov_setting_dict(region2_info$cov_setting),
    kernel_type_id = kernel_type_id,
    verbose = verbose
  )

  theta <- result$theta
  names(theta) <- c("rho", "k_eta1", "k_eta2", "tau_eta", "nugget_eta")
  return(list(theta = theta, objective = result$objective))
}
