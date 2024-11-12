xi_tau_opt <- function(theta, data_cross, time_sqrd) {
  chi <- theta[1]
  tau <- theta[2]
  xi_tau_rbf <- chi * rbf(tau, time_sqrd)
  diag(xi_tau_rbf) <- 0
  sum(sapply(data_cross, \(slice) {
    sum((slice - xi_tau_rbf)^2)
  }))
}

init_region <- function(region_info) {
  s1 <- region_info$stage1
  xi_ot <- s1$k_gamma + s1$nugget_gamma
  xi_im <- s1$k_gamma
  if (region_info$cov_setting == "noisy") {
    xi_ot <- s1$var_noise * (xi_ot + 1)
    xi_im <- s1$var_noise * xi_im
  }
  chi_ot <- max(0, 1 - xi_ot)

  n_timept <- nrow(region_info$data_std)
  n_voxel <- ncol(region_info$data_std)
  time_sqrd <- outer(1:n_timept, 1:n_timept, `-`)^2
  xi_im_rbf <- xi_im * rbf(s1$tau_gamma, time_sqrd)
  data_cross <- lapply(1:n_voxel, \(l) {
    slice <- tcrossprod(region_info$data_std[, l]) - xi_im_rbf
    diag(slice) <- 0
    slice
  })

  # May have to multistart this optimization!!
  optim_result <- stats::optim(c(1, 1), xi_tau_opt,
    data_cross = data_cross, time_sqrd = time_sqrd,
    method = "L-BFGS-B", lower = c(0.0001, 0.0001)
  )
  # message(optim_result$value)
  (par <- optim_result$par)
  chi_im <- par[1]
  k_eta <- chi_im
  if (region_info$cov_setting == "noisy") {
    k_eta <- k_eta / s1$var_noise
  }
  nugget_eta <- max(0, chi_ot / chi_im - 1)

  c(k_eta = k_eta, tau_eta = par[2], nugget_eta = nugget_eta)
}

stage2_init <- function(r1_info, r2_info) {
  r1k <- init_region(r1_info)
  k_eta1 <- r1k["k_eta"]
  tau_eta1 <- r1k["tau_eta"]
  nugget_eta1 <- r1k["nugget_eta"]

  r2k <- init_region(r2_info)
  k_eta2 <- r2k["k_eta"]
  tau_eta2 <- r2k["tau_eta"]
  nugget_eta2 <- r2k["nugget_eta"]

  tau_eta <- mean(c(tau_eta1, tau_eta2))
  nugget_eta <- max(nugget_eta1, nugget_eta2)
  rho_eblue <- cor(r1_info$eblue, r2_info$eblue)
  init <- c(
    rho = rho_eblue,
    k_eta1 = k_eta1,
    k_eta2 = k_eta2,
    tau_eta = tau_eta,
    nugget_eta = nugget_eta
  )
  names(init)[2] <- "k_eta1"
  names(init)[3] <- "k_eta2"
  return(init)
}
