test_that("fisher information matrix", {
  d1 <- qfunc_sim_data$data[[1]]
  d2 <- qfunc_sim_data$data[[2]]

  c1 <- qfunc_sim_data$coords[[1]]
  c2 <- qfunc_sim_data$coords[[2]]

  theta <- c(
    rho = 0.1,
    k_eta1 = 2.7,
    k_eta2 = 2.9,
    tau_eta = 0.3,
    nugget_eta = 0.1
  )

  m <- 60
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

  fisher_info <- get_fisher_info(
    theta = theta,
    data_r1 = d1,
    data_r2 = d2,
    coords_r1 = c1,
    coords_r2 = c2,
    time_sqrd_mat = time_sqrd_mat,
    stage1_r1 = stage1_r1,
    stage1_r2 = stage1_r2,
    cov_setting_id1 = 0L,
    cov_setting_id2 = 0L,
    kernel_type_id = kernel_dict("matern_5_2")
  )

  expect_equal(dim(fisher_info), c(5, 5))
  expect_equal(colnames(fisher_info), names(theta))
  expect_equal(rownames(fisher_info), names(theta))
})
