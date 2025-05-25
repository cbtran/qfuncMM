rbf <- function(tau, time_sqrd_mat) {
  exp(-tau^2 / 2 * time_sqrd_mat)
}

matern <- function(phi, dist_mat) {
  (1 + phi * sqrt(5) * dist_mat + phi^2 * (5 / 3) * dist_mat^2) *
    exp(-phi * sqrt(5) * dist_mat)
}

# Nonlinear least squares objective for k and tau
opt_init_temporal <- function(theta, time_sqrd_mat, raw_cov_numeric) {
  k <- theta[1]
  tau <- theta[2]
  kh <- k * rbf(tau, time_sqrd_mat) # M x M matrix
  diag(kh) <- 0
  sum((raw_cov_numeric - as.numeric(kh))^2)
}

# Nonlinear least squares objective for phi
opt_init_spatial <- function(phi, dist_mat, raw_cov_numeric) {
  m <- matern(phi, dist_mat) # Lj x Lj matrix
  diag(m) <- 0
  obj <- sum((raw_cov_numeric - as.numeric(m))^2)
  return(obj)
}

# Returns initializations for temporal parameters k, tau, and nugget.
init_temporal <- function(region_mx) {
  num_voxel <- ncol(region_mx)
  num_timept <- nrow(region_mx)
  time_sqrd_mat <- outer(seq_len(num_timept), seq_len(num_timept), `-`)^2
  xbar <- rowMeans(region_mx)
  centered <- region_mx - xbar # M x Lj matrix with centered columns
  raw_cov <- array(NA, dim = c(num_timept, num_timept, num_voxel))
  for (l in seq_len(num_voxel)) {
    cct <- centered[, l] %*% t(centered[, l]) # M x M matrix
    diag(cct) <- 0
    raw_cov[, , l] <- cct
  }

  par <- stats::optim(c(0.1, 0.1), opt_init_temporal,
    time_sqrd_mat = time_sqrd_mat, raw_cov_numeric = as.numeric(raw_cov),
    method = "L-BFGS-B", lower = c(0.0001, 0.0001)
  )$par
  k <- par[1]
  tau <- par[2]
  nugget <- sum(centered^2) / (num_voxel * num_timept - 1) - k
  nugget <- max(0.01, nugget)
  return(c(k = k, tau = tau, nugget = nugget))
}

init_spatial <- function(region_mx, voxel_coords) {
  num_voxel <- nrow(voxel_coords)
  num_timept <- nrow(region_mx)
  dist_mat <- as.matrix(stats::dist(voxel_coords))
  xbar <- colMeans(region_mx) # length Lj vector
  centered <- t(region_mx) - xbar # Lj x M matrix with centered columns
  raw_cov <- array(NA, dim = c(num_voxel, num_voxel, num_timept))
  for (m in seq_len(num_timept)) {
    cct <- centered[, m] %*% t(centered[, m])
    diag(cct) <- 0
    raw_cov[, , m] <- cct
  }
  par <- stats::optim(0.1, opt_init_spatial,
    dist_mat = dist_mat, raw_cov_numeric = as.numeric(raw_cov),
    method = "L-BFGS-B", lower = 0.0001
  )$par
  return(c(phi = par[1]))
}

# Produces a list of initializations for the intra-regional model fitting.
stage1_init <- function(region_mx, voxel_coords, num_init, profiled_k) {
  if (num_init < 1) {
    num_init <- 1
  }
  init_temp <- init_temporal(region_mx)
  k <- init_temp["k"]
  tau <- init_temp["tau"]
  nugget <- init_temp["nugget"]

  init_space <- init_spatial(region_mx, voxel_coords)
  phi <- init_space["phi"]

  if (profiled_k) {
    nugget_over_k <- nugget / k
    theta <- c(tau, nugget_over_k)
    diffs <- theta / 2
    inits <- rbind(theta - diffs, theta, theta + diffs)
    inits <- cbind(phi, as.matrix(expand.grid(inits[, 1], inits[, 2])))
    colnames(inits) <- c("phi", "tau", "nugget_over_k")
  } else {
    theta <- c(phi, tau, k, nugget)
    inits <- matrix(nrow = num_init, ncol = length(theta))
    inits[1, ] <- theta
    colnames(inits) <- c("phi", "tau", "k", "nugget")

    if (num_init > 1) {
      # Randomly perturb the initializations
      for (it in seq_along(theta)) {
        t <- theta[it]
        t_inits <- stats::rlnorm(
          num_init - 1,
          meanlog = log(t), sdlog = ifelse(t < 0.01, 3, 1)
        )
        inits[-1, it] <- t_inits
      }
    }
  }
  return(inits)
}
