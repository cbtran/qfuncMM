#' Stage 1: Fit intra-regional model for a single region
#'
#' @param region_mx Data matrix of signals of 1 region
#' @param voxel_coords coordinates of voxels in the region
#' @param time_sqrd_mat Temporal squared distance matrix
#' @param kernel_type_id Choice of spatial kernel. Default "matern_5_2".
#' @return Esimated intra parameters and noise variance
#'
#' @noRd

fit_intra_model <- function(
    region_mx,
    voxel_coords,
    kernel_type_id = 3L,
    # time_sqrd_mat,
    nugget_only = FALSE,
    noiseless = FALSE,
    noiseless_profiled = FALSE,
    init = NULL) {
  # Param list: phi, tau_gamma, k_gamma, nugget_gamma
  # param_init <- init
  # if (nugget_only) {
  #   param_init <- c(0.1, 0.1)
  # }

  m <- nrow(region_mx)
  time_sqrd_mat <- outer(seq_len(m), seq_len(m), `-`)^2
  inits <- NULL
  if (is.null(init)) {
    inits <- stage1_init(region_mx, voxel_coords, noiseless_profiled)
  } else {
    inits <- matrix(init, nrow = 1)
  }
  n_init <- nrow(inits)

  best_intra <- NULL
  best_obj <- Inf
  for (init_num in seq_len(n_init)) {
    intra <- tryCatch(
      {
        intra <- opt_intra(
          inits[init_num, ], matrix(region_mx, ncol = 1),
          voxel_coords, time_sqrd_mat, kernel_type_id,
          nugget_only, noiseless, noiseless_profiled
        )
      },
      error = function(e) {
        warning(e)
        list(theta = rep(NA, 4), var_noise = NA, eblue = rep(NA, m), objval = Inf)
      }
    )
    if (intra$objval < best_obj) {
      best_obj <- intra$objval
      best_intra <- intra
    }
  }

  intra_param <- c(best_intra$theta, best_intra$var_noise)
  # if (nugget_only) {
  #   intra_param <- c(intra_param[1], 1, 0, intra_param[2], best_intra$var_noise)
  # }
  names(intra_param) <- c("phi", "tau_gamma", "k_gamma", "nugget_gamma", "var_noise")
  list(intra_param = intra_param, eblue = intra$eblue, objval = intra$objval)
}
