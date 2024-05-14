# Compute the correlation of averages for two regions.
# Regions are given as M x Lj matrices.
compute_ca <- function(region1_mx, region2_mx) {
  m <- nrow(region1_mx)
  cor(apply(region1_mx, 1, mean), apply(region2_mx, 1, mean)) * (m - 1) / m
}
