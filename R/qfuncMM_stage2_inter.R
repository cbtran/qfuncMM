#' Estimate functional connectivity from voxel-level BOLD signals. Run stage 2 inter-regional analysis for a pair of regions.
#'   Results are saved to a JSON file.
#'
#' @param stage1_region1_outfile JSON file from stage 1 for region 1.
#' @param stage1_region2_outfile JSON file from stage 1 for region 2.
#' @param out_dir Output directory.
#' @param m_seq Number of neighbors to use in Vecchia approximation.
#' @param st_scale Scaling factor for spatial and temporal dimensions.
#' @param kernel_type Choice of spatial kernel.
#' @param overwrite Overwrite existing output file.
#' @param verbose Print progress messages.
#'
#' @useDynLib qfuncMM
#' @importFrom Rcpp sourceCpp
#' @importFrom jsonlite toJSON read_json
#' @export
qfuncMM_stage2_inter <- function(
    stage1_region1_outfile, stage1_region2_outfile, out_dir, m_seq, st_scale,
    kernel_type = "matern_5_2", overwrite = FALSE, verbose = FALSE) {
  kernel_type_id <- kernel_dict(kernel_type)

  if (!dir.exists(out_dir)) {
    dir.create(out_dir)
  }
  if (!file.info(out_dir)$isdir || file.access(out_dir, mode = 2) != 0) {
    stop(sprintf("The specified output location '%s' is not in a valid, writeable directory.", out_dir))
  }

  j1 <- jsonlite::read_json(stage1_region1_outfile, simplifyVector = TRUE)
  if (j1$stage1$sigma2_ep == "NA") {
    j1$stage1$sigma2_ep <- NA
  }
  j2 <- jsonlite::read_json(stage1_region2_outfile, simplifyVector = TRUE)
  if (j2$stage1$sigma2_ep == "NA") {
    j2$stage1$sigma2_ep <- NA
  }
  if (j1$subject_id != j2$subject_id) {
    stop(sprintf(
      "Mismatched subject IDs '%s' and '%s'. The two regions must be from the same subject and exam.",
      j1$subject_id, j2$subject_id
    ))
  }
  if (j1$region_uniqid == j2$region_uniqid) {
    stop(sprintf("Both regions have unique id '%d'. Regions must be different.", j1$region_uniqid))
  }

  out_file <- file.path(
    out_dir,
    sprintf("qfuncMM_stage2_inter_%s_%d-%d.json", j1$subject_id, j1$region_uniqid, j2$region_uniqid)
  )

  if (file.exists(out_file) && !overwrite) {
    stop(sprintf("Output file '%s' already exists. Set 'overwrite' to TRUE to overwrite.", out_file))
  }

  start_time <- Sys.time()
  rho_eblue <- stats::cor(j1$eblue, j2$eblue)
  rho_ca <- stats::cor(rowMeans(j1$data_std), rowMeans(j2$data_std))

  message(sprintf(
    "Running QFunCMM stage 2 inter-regional for subject '%s' region pair (%d, %d)...",
    j1$subject_id, j1$region_uniqid, j2$region_uniqid
  ))

  # init <- stage2_init(j1, j2)
  # inter_result <- fit_inter_model(j1, j2, kernel_type_id, init, verbose)
  inter_result <- GpGp::fit_qfuncmm(j1, j2, st_scale = st_scale, m_seq = m_seq)
  theta <- inter_result$covparms

  outlist <- list(
    subject_id = j1$subject_id,
    region1_uniqid = j1$region_uniqid, region1_name = j1$region_name,
    region2_uniqid = j2$region_uniqid, region2_name = j2$region_name,
    region1_num_voxel = ncol(j1$data_std), region2_num_voxel = ncol(j2$data_std),
    num_timept = length(j1$eblue),
    spatial_kernel = kernel_type,
    start_time = format(start_time, "%Y-%m-%dT%H:%M:%OS3Z"),
    run_time_minutes = round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 5)
  )
  outlist$stage2 <-
    c(
      list(rho = theta["rho"], rho_eblue = rho_eblue, rho_ca = rho_ca),
      as.list(theta[get("stage2_paramlist_components", qfuncMM_pkg_env)])
    )
  outlist$loglik <- inter_result$loglik
  outlist$mu <- inter_result$betahat
  out_json <- jsonlite::toJSON(outlist, auto_unbox = TRUE, pretty = TRUE, digits = I(10))
  write(out_json, out_file)
  message(
    sprintf(
      "Subject '%s' region pair (%d, %d): Finished stage 2 inter-regional. \nResults saved to ",
      j1$subject_id, j1$region_uniqid, j2$region_uniqid
    ),
    normalizePath(out_file)
  )
}
