test_that("get_dist_sqrd_mat", {
  # Create squared distance matrix from coordinates
  coords <- matrix(1:30, nrow = 10, ncol = 3)
  expected <- matrix(c(
    0, 3, 12, 27, 48, 75, 108, 147, 192, 243,
    3, 0, 3, 12, 27, 48, 75, 108, 147, 192,
    12, 3, 0, 3, 12, 27, 48, 75, 108, 147,
    27, 12, 3, 0, 3, 12, 27, 48, 75, 108,
    48, 27, 12, 3, 0, 3, 12, 27, 48, 75,
    75, 48, 27, 12, 3, 0, 3, 12, 27, 48,
    108, 75, 48, 27, 12, 3, 0, 3, 12, 27,
    147, 108, 75, 48, 27, 12, 3, 0, 3, 12,
    192, 147, 108, 75, 48, 27, 12, 3, 0, 3,
    243, 192, 147, 108, 75, 48, 27, 12, 3, 0),
    nrow = 10, ncol = 10)
  actual <- get_dist_sqrd_mat(coords)
  expect_equal(actual, expected)
})
