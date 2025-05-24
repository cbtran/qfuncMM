#' Estimate functional connectivity from voxel-level BOLD signals. Run stage 2 inter-regional analysis for a pair of regions.
#'   Results are saved to a JSON file.
#'
#' @param stage1_region1_outfile JSON file from stage 1 for region 1.
#' @param stage1_region2_outfile JSON file from stage 1 for region 2.
#' @param out_dir Output directory.
#' @param data_and_coords Optional list containing standardized data and coordinates for both regions,
#'   if not provided in stage 1 files.
#' @param overwrite Overwrite existing output file.
#' @param verbose Print progress messages.
#'
#' @useDynLib qfuncMM
#' @importFrom Rcpp sourceCpp
#' @importFrom jsonlite toJSON read_json
#' @export
qfuncMM_stage2_reml <- function(
    stage1_region1_outfile, stage1_region2_outfile, out_dir,
    data_and_coords = NULL, overwrite = FALSE, verbose = FALSE) {
  region1_data <- jsonlite::read_json(stage1_region1_outfile, simplifyVector = TRUE)
  region2_data <- jsonlite::read_json(stage1_region2_outfile, simplifyVector = TRUE)

  if (region1_data$stage1$sigma2_ep == "NA") {
    region1_data$stage1$sigma2_ep <- NA
  }
  if (region2_data$stage1$sigma2_ep == "NA") {
    region2_data$stage1$sigma2_ep <- NA
  }

  # Set data and coordinates if provided
  if (!is.null(data_and_coords)) {
    region1_data$data_std <- data_and_coords$data_std1
    region2_data$data_std <- data_and_coords$data_std2
    region1_data$coords <- data_and_coords$coords1
    region2_data$coords <- data_and_coords$coords2
  }

  run_stage2(
    region1_data,
    region2_data,
    out_dir = out_dir,
    method = "reml",
    overwrite = overwrite,
    verbose = verbose
  )
}
