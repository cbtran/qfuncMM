#' Stage 1: Fit intra-regional model for a single region over a list of initializations
#'
#' @noRd

fit_intra_model <- function(
    region_mx,
    voxel_coords,
    inits,
    kernel_type_id,
    cov_setting = c("noisy", "noiseless"),
    verbose = FALSE) {
  cov_setting <- match.arg(cov_setting)
  m <- nrow(region_mx)
  time_sqrd_mat <- outer(seq_len(m), seq_len(m), `-`)^2
  n_init <- nrow(inits)

  best_intra <- NULL
  best_obj <- Inf
  out_names <- c(get("stage1_paramlist", qfuncMM_pkg_env), "psi", "nll")
  results_by_init <- matrix(nrow = n_init, ncol = length(out_names))
  colnames(results_by_init) <- out_names
  for (init_num in seq_len(n_init)) {
    intra <- tryCatch(
      {
        intra <- opt_intra(
          inits[init_num, ], matrix(region_mx, ncol = 1),
          voxel_coords, time_sqrd_mat, kernel_type_id, cov_setting_dict(cov_setting), verbose
        )
      },
      error = function(e) {
        warning(e)
        list(theta = rep(NA, 4), sigma2_ep = NA, psi = NA, eblue = rep(NA, m), objval = Inf)
      }
    )
    results_by_init[init_num, ] <- c(intra$theta, intra$sigma2_ep, intra$psi, intra$objval)
    if (intra$objval < best_obj) {
      best_obj <- intra$objval
      best_intra <- intra
    }
  }

  intra_param <- c(best_intra$theta, best_intra$sigma2_ep, best_intra$psi)
  names(intra_param) <- c(get("stage1_paramlist", qfuncMM_pkg_env), "psi")
  list(
    intra_param = intra_param, eblue = best_intra$eblue, objval = best_intra$objval,
    initializations = inits, results_by_init = results_by_init
  )
}
