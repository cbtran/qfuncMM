qfuncMM_pkg_env <- new.env(parent = emptyenv())
assign("stage1_paramlist", c("phi_gamma", "tau_gamma", "k_gamma", "nugget_gamma", "sigma2_ep"), envir = qfuncMM_pkg_env)
assign("stage2_paramlist_components", c("k_eta1", "k_eta2", "tau_eta", "nugget_eta"), envir = qfuncMM_pkg_env)
assign("stage2_paramlist", c("rho", get("stage2_paramlist_components", qfuncMM_pkg_env)), envir = qfuncMM_pkg_env)

stage1_nll <- function(phi, tau_gamma, k_gamma, nugget_gamma, region_mx, voxel_coords) {
  m <- nrow(region_mx)
  time_sqrd_mat <- outer(seq_len(m), seq_len(m), `-`)^2
  theta <- c(phi, tau_gamma, k_gamma, nugget_gamma)
  s1 <- eval_stage1_nll(theta, matrix(region_mx, ncol = 1), voxel_coords, time_sqrd_mat, 3L)
  grad <- as.numeric(s1$grad)
  names(grad) <- c("dPhi", "dTau_gamma", "dK_gamma", "dNugget_gamma")
  list(nll = s1$nll, grad = grad)
}

fisher_z <- function(r) {
  atanh(r)
}

d_fisher_z <- function(r) {
  if (r < -1 || r > 1) {
    stop("r must be between -1 and 1")
  }
  return(1 / (1 - r^2))
}

inv_fisher_z <- function(z) {
  tanh(z)
}
