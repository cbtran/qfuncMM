#' Stage 2: Fit inter-regional model given pair of regions
#'
#' @param region1_mx Data matrix of signals of region 1
#' @param voxel_coords_1 coordinates of voxels in region 1
#' @param region2_mx Data matrix of signals of region 2
#' @param voxel_coords_2 coordinates of voxels in region 2
#' @param time_sqrd_mat Temporal squared distance matrix
#' @param stage1_regional Estimated parameters of intra-regional models from 2 regions
#' @param kernel_type_id Choice of spatial kernel
#' @return Estimated stage 2 parameters
#' @noRd

fit_inter_model <- function(
    region1_mx,
    voxel_coords_1,
    region2_mx,
    voxel_coords_2,
    time_sqrd_mat,
    region1_stage1,
    region2_stage1,
    kernel_type_id) {

  ca <- compute_ca(region1_mx, region2_mx)
  # Parameter list: rho, kEta1, kEta2, tauEta, nugget
  softminus <- function(x) {
    log(exp(x) - 1)
  }
  # Reasonable initiliazation
  init <- c(ca, softminus(1), softminus(1), 0, softminus(0.1))

  result <- opt_inter(theta_init = init,
                      dataRegion1 = matrix(region1_mx, ncol = 1),
                      dataRegion2 = matrix(region2_mx, ncol = 1),
                      voxel_coords_1 = voxel_coords_1,
                      voxel_coords_2 = voxel_coords_2,
                      time_sqrd_mat = time_sqrd_mat,
                      stage1ParamsRegion1 = region1_stage1,
                      stage1ParamsRegion2 = region2_stage1,
                      kernel_type_id = kernel_type_id)

  result <- as.list(c(result$theta, result$var_noise))
  names(result) <- c("rho", "k_eta1", "k_eta2", "tau_eta", "nugget_eta")
  return(result)
}

# Compute the correlation of averages for two regions
compute_ca <- function(region1_mx, region2_mx) {
  r1avg <- apply(region1_mx, 1, mean)
  r2avg <- apply(region2_mx, 1, mean)
  r1avgavg <- r1avg - mean(r1avg)
  r2avgavg <- r2avg - mean(r2avg)
  sum(r1avgavg * r2avgavg) / (stats::sd(r1avg) * stats::sd(r2avg) * length(r1avg))
}
