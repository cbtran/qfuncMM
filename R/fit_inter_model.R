#' Stage 2: Fit inter-regional model given pair of regions
#'
#' @noRd

fit_inter_model <- function(region1_info, region2_info, kernel_type_id, init, verbose, max_iter = 100) {
  m <- length(region1_info$eblue)
  time_sqrd_mat <- outer(seq_len(m), seq_len(m), `-`)^2

  r1_num_voxel <- ncol(region1_info$data_std)
  r2_num_voxel <- ncol(region2_info$data_std)
  if (r1_num_voxel < r2_num_voxel) {
    # It is faster to put the smaller region second
    temp <- region1_info
    region1_info <- region2_info
    region2_info <- temp
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

  result <- minqa::bobyqa(init, stage2_inter_reml,
    lower = c(-1, 0, 0, 0, 0), upper = c(1, Inf, Inf, Inf, Inf),
    region1 = region1_info$data_std, region2 = region2_info$data_std,
    r1_coords = region1_info$coords, r2_coords = region2_info$coords,
    time_sqrd_mat = time_sqrd_mat, r1_stage1 = unlist(region1_info$stage1),
    r2_stage1 = unlist(region2_info$stage1),
    lambda1 = lambda1, lambda2 = lambda2,
    cov_setting_id1 = cov_setting_dict(region1_info$cov_setting),
    cov_setting_id2 = cov_setting_dict(region2_info$cov_setting),
    kernel_type_id = 3L,
    control = list(npt = 9, iprint = 3, rhobeg = 0.2, rhoend = 1e-6, maxfun = max_iter)
  )

  theta <- result$par
  names(theta) <- c("rho", "k_eta1", "k_eta2", "tau_eta", "nugget_eta")
  if (r1_num_voxel < r2_num_voxel) {
    temp <- theta["k_eta1"]
    theta["k_eta1"] <- theta["k_eta2"]
    theta["k_eta2"] <- temp
  }
  return(list(theta = theta, objective = result$fval))
}
