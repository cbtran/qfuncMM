# Internal function for stage 2 inter-regional analysis
# Not exported nor documented
run_stage2 <- function(
    region1_data, region2_data, out_dir,
    method = c("vecchia", "reml"),
    m_seq = 100, st_scale = c(10, 1),
    overwrite = FALSE, verbose = FALSE) {
  method <- match.arg(method)

  required_components <- c(
    "subject_id", "region_uniqid", "region_name",
    "stage1", "cov_setting", "data_std", "coords", "eblue"
  )
  missing_r1 <- setdiff(required_components, names(region1_data))
  missing_r2 <- setdiff(required_components, names(region2_data))

  if (length(missing_r1) > 0) {
    stop(sprintf("Region 1 data is missing required components: %s", paste(missing_r1, collapse = ", ")))
  }
  if (length(missing_r2) > 0) {
    stop(sprintf("Region 2 data is missing required components: %s", paste(missing_r2, collapse = ", ")))
  }

  j1 <- region1_data
  j2 <- region2_data

  if (j1$subject_id != j2$subject_id) {
    stop(sprintf(
      "Mismatched subject IDs '%s' and '%s'. The two regions must be from the same subject and exam.",
      j1$subject_id, j2$subject_id
    ))
  }
  if (j1$region_uniqid == j2$region_uniqid) {
    stop(sprintf("Both regions have unique id '%d'. Regions must be different.", j1$region_uniqid))
  }

  out_file <- NULL
  if (!is.null(out_dir)) {
    if (!dir.exists(out_dir)) {
      dir.create(out_dir)
    }
    if (!file.info(out_dir)$isdir || file.access(out_dir, mode = 2) != 0) {
      stop(sprintf("The specified output location '%s' is not in a valid, writeable directory.", out_dir))
    }

    out_file <- file.path(
      out_dir,
      sprintf("qfuncMM_stage2_inter_%s_%d-%d.json", j1$subject_id, j1$region_uniqid, j2$region_uniqid)
    )

    if (file.exists(out_file) && !overwrite) {
      stop(sprintf("Output file '%s' already exists. Set 'overwrite' to TRUE to overwrite.", out_file))
    }
  }

  start_time <- Sys.time()
  rho_eblue <- stats::cor(j1$eblue, j2$eblue)
  rho_ca <- stats::cor(rowMeans(j1$data_std), rowMeans(j2$data_std))

  message(sprintf(
    "Running QFunCMM stage 2 with %s for subject '%s' region pair (%d, %d)...",
    ifelse(method == "vecchia", "Vecchia's approximation", "REML"),
    j1$subject_id, j1$region_uniqid, j2$region_uniqid
  ))

  # Fit model based on method
  if (method == "vecchia") {
    inter_result <- GpGpQFuncMM::fit_qfuncmm(j1, j2, st_scale = st_scale, m_seq = m_seq)
  } else { # method == "reml"
    init <- stage2_init(j1, j2)
    inter_result <- fit_inter_model(j1, j2, kernel_dict("matern_5_2"), init, verbose)
    inter_result$betahat <- c(NA, NA) # Mean not estimated for ReML yet
  }

  theta <- inter_result$covparms

  outlist <- list(
    subject_id = j1$subject_id,
    region1_uniqid = j1$region_uniqid, region1_name = j1$region_name,
    region2_uniqid = j2$region_uniqid, region2_name = j2$region_name,
    region1_num_voxel = ncol(j1$data_std), region2_num_voxel = ncol(j2$data_std),
    num_timept = length(j1$eblue),
    spatial_kernel = "matern_5_2",
    start_time = format(start_time, "%Y-%m-%dT%H:%M:%OS3Z"),
    run_time_minutes = round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 5)
  )
  outlist$stage2 <-
    c(
      list(rho = theta[["rho"]], rho_eblue = rho_eblue, rho_ca = rho_ca),
      as.list(theta[get("stage2_paramlist_components", qfuncMM_pkg_env)])
    )
  outlist$loglik <- inter_result$loglik
  outlist$mu <- inter_result$betahat

  message(
    sprintf(
      "Subject '%s' region pair (%d, %d): Finished stage 2 inter-regional.",
      j1$subject_id, j1$region_uniqid, j2$region_uniqid
    )
  )
  if (!is.null(out_file)) {
    out_json <- jsonlite::toJSON(outlist, auto_unbox = TRUE, pretty = TRUE, digits = I(10))
    write(out_json, out_file)
    message("Results saved to ", normalizePath(out_file))
  }
  return(outlist)
}
