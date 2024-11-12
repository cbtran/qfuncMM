stage1_nll <- function(phi, tau_gamma, k_gamma, nugget_gamma, region_mx, voxel_coords) {
  m <- nrow(region_mx)
  time_sqrd_mat <- outer(seq_len(m), seq_len(m), `-`)^2
  theta <- c(phi, tau_gamma, k_gamma, nugget_gamma)
  s1 <- eval_stage1_nll(theta, matrix(region_mx, ncol = 1), voxel_coords, time_sqrd_mat, 3L)
  grad <- as.numeric(s1$grad)
  names(grad) <- c("dPhi", "dTau_gamma", "dK_gamma", "dNugget_gamma")
  list(nll = s1$nll, grad = grad)
}

stage1_lambda <- function(region1_info, region2_info) {
  m <- length(region1_info$eblue)
  time_sqrd_mat <- outer(seq_len(m), seq_len(m), `-`)^2
  r1_stage1 <- unlist(region1_info$stage1)
  r2_stage1 <- unlist(region2_info$stage1)

  r1_distmat <- as.matrix(stats::dist(region1_info$coords))
  c1 <- matern(r1_stage1["phi"], r1_distmat)
  b1 <- r1_stage1["k_gamma"] * rbf(r1_stage1["tau_gamma"], time_sqrd_mat)
  diag(b1) <- diag(b1) + r1_stage1["nugget_gamma"]
  lambda1 <- kronecker(c1, b1)

  r2_distmat <- as.matrix(stats::dist(region2_info$coords))
  c2 <- matern(r2_stage1["phi"], r2_distmat)
  b2 <- r2_stage1["k_gamma"] * rbf(r2_stage1["tau_gamma"], time_sqrd_mat)
  diag(b2) <- diag(b2) + r2_stage1["nugget_gamma"]
  lambda2 <- kronecker(c2, b2)

  list(lambda1 = lambda1, lambda2 = lambda2, time_sqrd_mat = time_sqrd_mat)
}

stage2_reml <- function(theta, region1, region2, r1_coords, r2_coords, time_sqrd_mat, r1_stage1, r2_stage1, lambda1, lambda2) {
  reml <- stage2_inter_reml(
    theta,
    region1, region2, r1_coords, r2_coords,
    time_sqrd_mat, r1_stage1, r2_stage1,
    lambda1, lambda2, 2L, 2L, 3L
  )
  reml
}
