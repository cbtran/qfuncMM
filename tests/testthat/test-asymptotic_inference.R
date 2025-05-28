test_that("noisy vs noiseless fisher information matrix", {
  set.seed(10315)
  m <- 10
  l1 <- 5
  l2 <- 5

  c1 <- matrix(rpois(3 * l1, lambda = 10), nrow = l1, ncol = 3)
  c2 <- matrix(rpois(3 * l2, lambda = 10), nrow = l2, ncol = 3)

  theta <- c(
    rho = 0.1,
    k_eta1 = 2.7,
    k_eta2 = 2.9,
    tau_eta = 0.3,
    nugget_eta = 0.1
  )

  time_sqrd_mat <- outer(seq_len(m), seq_len(m), `-`)^2
  stage1_r1 <- c(
    phi_gamma = 0.76,
    tau_gamma = 0.52,
    k_gamma = 2.05,
    nugget_gamma = 0.14,
    sigma2_ep = 0.16
  )

  stage1_r2 <- c(
    phi_gamma = 1.25,
    tau_gamma = 0.48,
    k_gamma = 2.09,
    nugget_gamma = 0.36,
    sigma2_ep = 0.18
  )

  # tictoc::tic("Fisher Information Matrix")
  fisher_info <- get_fisher_info(
    theta = theta,
    coords_r1 = c1,
    coords_r2 = c2,
    time_sqrd_mat = time_sqrd_mat,
    stage1_r1 = stage1_r1,
    stage1_r2 = stage1_r2,
    cov_setting_id1 = cov_setting_dict("noisy"),
    cov_setting_id2 = cov_setting_dict("noisy"),
    kernel_type_id = kernel_dict("matern_5_2"),
    reml = TRUE
  )
  # tictoc::toc()
  # print(sqrt(sum(fisher_info^2)))
  # print(solve(fisher_info)["rho", "rho"])
  expect_equal(dim(fisher_info), c(15, 15))
  expect_names <- c(
    "phi_gamma1", "tau_gamma1", "k_gamma1", "nugget_gamma1",
    "phi_gamma2", "tau_gamma2", "k_gamma2", "nugget_gamma2",
    names(theta),
    "sigma2_ep1", "sigma2_ep2"
  )
  expect_equal(colnames(fisher_info), expect_names)
  expect_equal(rownames(fisher_info), expect_names)

  # tictoc::tic("Fisher Information Matrix - Noiseless")
  fisher_info_noiseless <- get_fisher_info(
    theta = theta,
    coords_r1 = c1,
    coords_r2 = c2,
    time_sqrd_mat = time_sqrd_mat,
    stage1_r1 = stage1_r1,
    stage1_r2 = stage1_r2,
    cov_setting_id1 = cov_setting_dict("noiseless"),
    cov_setting_id2 = cov_setting_dict("noiseless"),
    kernel_type_id = kernel_dict("matern_5_2"),
    reml = TRUE
  )
  # tictoc::toc()
  # print(sqrt(sum(fisher_info_noiseless^2)))
  # print(solve(fisher_info_noiseless)["rho", "rho"])
  expect_equal(dim(fisher_info_noiseless), c(13, 13))
  expect_names <- c(
    "phi_gamma1", "tau_gamma1", "k_gamma1", "nugget_gamma1",
    "phi_gamma2", "tau_gamma2", "k_gamma2", "nugget_gamma2",
    names(theta)
  )
  expect_equal(colnames(fisher_info_noiseless), expect_names)
  expect_equal(rownames(fisher_info_noiseless), expect_names)

  # tictoc::tic("Fisher Information Matrix - mixed")
  fisher_info_mixed <- get_fisher_info(
    theta = theta,
    coords_r1 = c1,
    coords_r2 = c2,
    time_sqrd_mat = time_sqrd_mat,
    stage1_r1 = stage1_r1,
    stage1_r2 = stage1_r2,
    cov_setting_id1 = cov_setting_dict("noiseless"),
    cov_setting_id2 = cov_setting_dict("noisy"),
    kernel_type_id = kernel_dict("matern_5_2"),
    reml = TRUE
  )
  # tictoc::toc()
  # print(sqrt(sum(fisher_info_mixed^2)))
  # print(solve(fisher_info_mixed)["rho", "rho"])
  expect_equal(dim(fisher_info_mixed), c(14, 14))
  expect_names <- c(
    "phi_gamma1", "tau_gamma1", "k_gamma1", "nugget_gamma1",
    "phi_gamma2", "tau_gamma2", "k_gamma2", "nugget_gamma2",
    names(theta), "sigma2_ep2"
  )
  expect_equal(colnames(fisher_info_mixed), expect_names)
  expect_equal(rownames(fisher_info_mixed), expect_names)
})

test_that("ReML vs ML fisher information matrix", {
  set.seed(10315)
  m <- 10
  l1 <- 5
  l2 <- 5

  c1 <- matrix(rpois(3 * l1, lambda = 10), nrow = l1, ncol = 3)
  c2 <- matrix(rpois(3 * l2, lambda = 10), nrow = l2, ncol = 3)

  theta <- c(
    rho = 0.1,
    k_eta1 = 2.7,
    k_eta2 = 2.9,
    tau_eta = 0.3,
    nugget_eta = 0.1
  )

  time_sqrd_mat <- outer(seq_len(m), seq_len(m), `-`)^2
  stage1_r1 <- c(
    phi_gamma = 0.76,
    tau_gamma = 0.52,
    k_gamma = 2.05,
    nugget_gamma = 0.14,
    sigma2_ep = 0.16
  )

  stage1_r2 <- c(
    phi_gamma = 1.25,
    tau_gamma = 0.48,
    k_gamma = 2.09,
    nugget_gamma = 0.36,
    sigma2_ep = 0.18
  )

  fisher_info_reml <- get_fisher_info(
    theta = theta,
    coords_r1 = c1,
    coords_r2 = c2,
    time_sqrd_mat = time_sqrd_mat,
    stage1_r1 = stage1_r1,
    stage1_r2 = stage1_r2,
    cov_setting_id1 = cov_setting_dict("noisy"),
    cov_setting_id2 = cov_setting_dict("noisy"),
    kernel_type_id = kernel_dict("matern_5_2"),
    reml = TRUE
  )

  fisher_info_ml <- get_fisher_info(
    theta = theta,
    coords_r1 = c1,
    coords_r2 = c2,
    time_sqrd_mat = time_sqrd_mat,
    stage1_r1 = stage1_r1,
    stage1_r2 = stage1_r2,
    cov_setting_id1 = cov_setting_dict("noiseless"),
    cov_setting_id2 = cov_setting_dict("noiseless"),
    kernel_type_id = kernel_dict("matern_5_2"),
    reml = FALSE
  )

  var_rho_reml <- solve(fisher_info_reml)["rho", "rho"]
  var_rho_ml <- solve(fisher_info_ml)["rho", "rho"]

  # Expect that the variance estimates are different
  expect_true(!isTRUE(all.equal(var_rho_reml, var_rho_ml)))
})

test_that("asymptotic ci from variance", {
  theta <- 0.1
  expect_error(get_asymp_ci_rho(theta, 0.95, asympvar_rho = 0.1), "Invalid or missing named value for 'rho'")
  names(theta) <- "rho"
  ci <- get_asymp_ci_rho(theta, 0.95, asympvar_rho = 0.1)
  expect_length(ci, 2)
  expect_named(ci, c("lower", "upper"))
  expect_true(all(ci >= -1 & ci <= 1))
})

test_that("asymptotic var rho", {
  set.seed(10315)
  m <- 10
  l1 <- 5
  l2 <- 5

  c1 <- matrix(rpois(3 * l1, lambda = 10), nrow = l1, ncol = 3)
  c2 <- matrix(rpois(3 * l2, lambda = 10), nrow = l2, ncol = 3)

  theta <- c(
    rho = 0.1,
    k_eta1 = 2.7,
    k_eta2 = 2.9,
    tau_eta = 0.3,
    nugget_eta = 0.1
  )

  time_sqrd_mat <- outer(seq_len(m), seq_len(m), `-`)^2
  stage1_r1 <- c(
    phi_gamma = 0.76,
    tau_gamma = 0.52,
    k_gamma = 2.05,
    nugget_gamma = 0.14,
    sigma2_ep = 0.16
  )

  stage1_r2 <- c(
    phi_gamma = 1.25,
    tau_gamma = 0.48,
    k_gamma = 2.09,
    nugget_gamma = 0.36,
    sigma2_ep = 0.18
  )

  region1_info <- list(
    coords = c1,
    stage1 = stage1_r1,
    cov_setting = "noisy",
    num_timepoints = m
  )

  region2_info <- list(
    coords = c2,
    stage1 = stage1_r2,
    cov_setting = "noisy",
    num_timepoints = m
  )

  avar_reml <- get_asymp_var_rho(theta, region1_info, region2_info, "reml")
  avar_vecchia <- get_asymp_var_rho(theta, region1_info, region2_info, "vecchia")

  expect_true(!isTRUE(all.equal(avar_reml, avar_vecchia)))
})
