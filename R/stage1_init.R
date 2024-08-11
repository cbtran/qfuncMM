rbf <- function(tau, time_sqrd_mat) {
  exp(-tau^2 / 2 * time_sqrd_mat)
}

opt_init <- function(theta, time_sqrd_mat, raw_cov_numeric) {
  k <- theta[1]
  tau <- theta[2]
  kh <- k * rbf(tau, time_sqrd_mat)
  diag(kh) <- 0
  sum((raw_cov_numeric - as.numeric(kh))^2)
}

# Produces a list of initializations for the intra-regional model fitting.
stage1_init <- function(region_mx, voxel_coords, profiled_k) {
  num_voxel <- ncol(region_mx)
  num_timept <- nrow(region_mx)
  time_sqrd_mat <- outer(seq_len(num_timept), seq_len(num_timept), `-`)^2

  xbar <- rowMeans(region_mx)
  centered <- region_mx - xbar
  raw_cov <- array(NA, dim = c(num_voxel, num_timept, num_timept))
  for (l in 1:num_voxel) {
    ctc <- centered[, l] %*% t(centered[, l])
    diag(ctc) <- 0
    raw_cov[l, , ] <- ctc
  }

  par <- stats::optim(c(0.1, 0.1), opt_init,
    time_sqrd_mat = time_sqrd_mat, raw_cov_numeric = as.numeric(raw_cov),
    method = "L-BFGS-B", lower = c(0.0001, 0.0001)
  )$par
  k <- par[1]
  if (k < .Machine$double.eps) {
    k <- 1 # Avoid division by zero
  }
  phi <- 1
  tau <- par[2]
  nugget <- sum(centered^2) / (num_voxel * num_timept - 1) - k
  nugget <- max(0.01, nugget)
  if (profiled_k) {
    nugget_over_k <- nugget / k

    theta <- c(tau, nugget_over_k)
    diffs <- theta / 2
    inits <- rbind(theta - diffs, theta, theta + diffs)
    inits <- cbind(phi, as.matrix(expand.grid(inits[, 1], inits[, 2])))
    colnames(inits) <- c("phi", "tau", "nugget_over_k")
  } else {
    inits <- matrix(c(phi, tau, k, nugget), nrow = 1)
    colnames(inits) <- c("phi", "tau", "k", "nugget")
  }
  return(inits)
}
