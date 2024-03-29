#' Simulated voxel-level BOLD signals
#'
#' Simulated data from three regions using real voxel coordinates from rats.
#' The regions have 41, 25, and 77 voxels with 60 time points.
#' Data are generated from the mixed model strong inter-regional signal
#' and weak intra-regional spatial correlation.
#'
#' @format ## `qfunc_sim_data`
#' A list with two elements:
#' \describe{
#'   \item{data}{List of data matrices for each region}
#'   \item{coords}{Matrix giving voxel coordinates for each region}
#' }
"qfunc_sim_data"