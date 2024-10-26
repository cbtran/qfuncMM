rbf <- function(tau, time_sqrd_mat) {
  exp(-tau^2 / 2 * time_sqrd_mat)
}

matern <- function(phi, dist_mat) {
  (1 + phi * sqrt(5) * dist_mat + phi^2 * (5 / 3) * dist_mat^2) *
    exp(-phi * sqrt(5) * dist_mat)
}
