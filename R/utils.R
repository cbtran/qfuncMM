stage1_nll <- function(phi, tau_gamma, k_gamma, nugget_gamma, region_mx, voxel_coords) {
  m <- nrow(region_mx)
  time_sqrd_mat <- outer(seq_len(m), seq_len(m), `-`)^2
  theta <- c(phi, tau_gamma, k_gamma, nugget_gamma)
  s1 <- eval_stage1_nll(theta, matrix(region_mx, ncol = 1), voxel_coords, time_sqrd_mat, 3L)
  grad <- as.numeric(s1$grad)
  names(grad) <- c("dPhi", "dTau_gamma", "dK_gamma", "dNugget_gamma")
  list(nll = s1$nll, grad = grad)
}
