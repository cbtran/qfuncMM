# library(Thpc)
RhpcBLASctl::blas_set_num_threads(15)
RhpcBLASctl::omp_set_num_threads(5)
library(RcppArmadillo)
library(Rcpp)
devtools::load_all()

softminus <- function(x) {
  log(exp(x) - 1)
}
# Reasonable initiliazation
init <- c(0.5, softminus(1), softminus(1), 0, softminus(0.1))

file1 <- "scratch/hcp/s108828/qfuncMM_stage1_intra_region_108828_19.json"
file2 <- "scratch/hcp/s108828/qfuncMM_stage1_intra_region_108828_20.json"
j1 <- jsonlite::read_json(file1, simplifyVector = TRUE)
if (j1$stage1$var_noise == "NA") {
  j1$stage1$var_noise <- NA
}
j2 <- jsonlite::read_json(file2, simplifyVector = TRUE)
if (j2$stage1$var_noise == "NA") {
  j2$stage1$var_noise <- NA
}

n_timept <- nrow(j1$data_std)
time_sqrd_mat <- outer(seq_len(n_timept), seq_len(n_timept), `-`)^2
prof <- qfuncMM::profile_inter(
  init, j1$data_std, j2$data_std, j1$coords, j2$coords, time_sqrd_mat,
  unlist(j1$stage1), unlist(j2$stage1), 3
)

library(RcppClock)
summary(naptimes, units = "s")
plot(naptimes)

nt_times <- naptimes$timer / 1e9
names(nt_times) <- naptimes$ticker
cat("reml", nt_times["reml"], "\n")
cat("gradients", nt_times["gradients"], "\n")
cat("total", nt_times["reml"] + nt_times["gradients"], "\n")

# print(rbind(c(prof$val, nt_orig$val$val), cbind(prof$grad, nt_orig$val$grad)))
# saveRDS(list(nt = naptimes, vals = prof), "scratch/profile/inter/naptimes_hcp_avoid_join.rds")
nt_orig <- readRDS("scratch/profile/inter/naptimes_hcp_avoid_join.rds")
plot(nt_orig$nt)
