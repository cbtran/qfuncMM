test_that("Mapped IDs are correct", {
  expect_equal(kernel_dict("rbf"), 0L)
  expect_equal(kernel_dict("matern_1_2"), 1L)
  expect_equal(kernel_dict("matern_3_2"), 2L)
  expect_equal(kernel_dict("matern_5_2"), 3L)
})

test_that("Invalid kernel", {
  expect_error(kernel_dict("invalid_kernel"),
    "Invalid covariance kernel: invalid_kernel")
})
