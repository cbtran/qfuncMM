% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fit_inter_mode.R
\name{fit_inter_model}
\alias{fit_inter_model}
\title{Fit inter-regional model}
\usage{
fit_inter_model(
  parameters_init = c(rep(0, 3), -2),
  X_Region1,
  X_Region2,
  dist_sqrd_mat_1,
  dist_sqrd_mat_2,
  time_sqrd_mat,
  gamma_vec,
  kernel_type = "matern_5_2"
)
}
\arguments{
\item{parameters_init}{unrestricted initialization of parameters \eqn{\rho, \tau_\eta, k_\eta, nugget}}

\item{X_Region1}{Data matrix of signals of region 1}

\item{X_Region2}{Data matrix of signals of region 2}

\item{dist_sqrd_mat_1}{Spatial squared distance matrix of region 1}

\item{dist_sqrd_mat_2}{Spatial squared distance matrix of region 2}

\item{time_sqrd_mat}{Temporal squared distance matrix}

\item{gamma_vec}{Estimated parameters of intra-regional models from 2 regions}

\item{kernel_type}{Choice of spatial kernel. Defaul "matern_5_2".}
}
\value{
List of 2 components:
\item{theta}{estimated inter-regional parameters \eqn{\hat{\rho}, \hat{\tau}_\eta, \hat{k}_\eta, \hat{nugget}, \hat{\mu}_1, \hat{\mu}_2}}
\item{asymptotic_var}{asymptotic variance of transformed correlation coefficient}
\item{rho_transformed}{Fisher transformation of correlation coefficient}
}
\description{
Fit inter-regional model
}
\examples{
L <- 20 # Numbers of voxels
side_length <- 7 # Side length of each region
M <- 30 # Numbers of timepoints
num_sim <- 1 # Numbers of simulation
# Generate voxels for each region
set.seed(1)
vxlID_1 <- sample(1:(side_length^3), L, replace = FALSE)
set.seed(2)
vxlID_2 <- sample(1:(side_length^3), L, replace = FALSE)
set.seed(4)
vxlID_3 <- sample(1:(side_length^3), L, replace = FALSE)

# rho
rho_vec <- data.frame(rho12=0.1, rho13=0.35, rho23=0.6)
# Parameters
parameters_true <- data.frame(tau_eta = 1/4, k_eta = 0.5,
                              phi_gamma_1=1, tau_gamma_1=1/2, k_gamma_1=2,
                              phi_gamma_2=1, tau_gamma_2=1/2, k_gamma_2=2,
                              phi_gamma_3=1, tau_gamma_3=1/2, k_gamma_3=2,
                              nugget=0.1, mu_1 = 1, mu_2 = 10, mu_3 = 20)

simulated_data = simulate_3_region(L1=L, L2=L, L3=L,
                                   side_length=side_length, 
                                   M=M, 
                                   theta=parameters_true, 
                                   rho_vec=rho_vec, 
                                   sigma_sqrd=1,
                                   mu_1=rep(parameters_true$mu_1, M),
                                   mu_2=rep(parameters_true$mu_2, M),
                                   mu_3=rep(parameters_true$mu_3, M),
                                   vxlID_1=vxlID_1, 
                                   vxlID_2=vxlID_2, 
                                   vxlID_3=vxlID_3,
                                   random_seed=1, num_sim=1,
                                   C_kernel_type="matern_5_2")

X_Region1 <- simulated_data[[1]]$X_Region1
X_Region2 <- simulated_data[[1]]$X_Region2
X_Region3 <- simulated_data[[1]]$X_Region3

dist_sqrd_mat_region1 <- get_dist_sqrd_mat(L, side_length, vxlID_1)
dist_sqrd_mat_region2 <- get_dist_sqrd_mat(L, side_length, vxlID_2)
dist_sqrd_mat_region3 <- get_dist_sqrd_mat(L, side_length, vxlID_3)
time_sqrd_mat <- (outer(1:M, 1:M, "-"))^2

fit_region1 <- fit_intra_model(X=X_Region1,
                               degree = 3,
                               nbasis = M/2,
                               dist_sqrd_mat=dist_sqrd_mat_region1,
                               time_sqrd_mat=time_sqrd_mat,
                               L=L, M=M,
                               kernel_type = "matern_5_2")

fit_region2 <- fit_intra_model(X=X_Region2,
                               degree = 3,
                               nbasis = M/2,
                               dist_sqrd_mat=dist_sqrd_mat_region2,
                               time_sqrd_mat=time_sqrd_mat,
                               L=L, M=M,
                               kernel_type = "matern_5_2")

fit_region3 <- fit_intra_model(X=X_Region3,
                               degree = 3,
                               nbasis = M/2,
                               dist_sqrd_mat=dist_sqrd_mat_region3,
                               time_sqrd_mat=time_sqrd_mat,
                               L=L, M=M,
                               kernel_type = "matern_5_2")

gamma_vec_12 <- c(fit_region1$theta, fit_region2$theta)
gamma_vec_13 <- c(fit_region1$theta, fit_region3$theta)
gamma_vec_23 <- c(fit_region2$theta, fit_region3$theta)

inter12 <- fit_inter_model(X_Region1=X_Region1,
                           X_Region2=X_Region2, 
                           dist_sqrd_mat_1=dist_sqrd_mat_region1, 
                           dist_sqrd_mat_2=dist_sqrd_mat_region2,
                           time_sqrd_mat=time_sqrd_mat,
                           gamma_vec=gamma_vec_12,
                           kernel_type="matern_5_2")


inter13 <- fit_inter_model(X_Region1=X_Region1,
                           X_Region2=X_Region3, 
                           dist_sqrd_mat_1=dist_sqrd_mat_region1, 
                           dist_sqrd_mat_2=dist_sqrd_mat_region3,
                           time_sqrd_mat=time_sqrd_mat,
                           gamma_vec=gamma_vec_13,
                           kernel_type="matern_5_2")


inter23 <- fit_inter_model(X_Region1=X_Region2,
                           X_Region2=X_Region3, 
                           dist_sqrd_mat_1=dist_sqrd_mat_region2, 
                           dist_sqrd_mat_2=dist_sqrd_mat_region3,
                           time_sqrd_mat=time_sqrd_mat,
                           gamma_vec=gamma_vec_23,
                           kernel_type="matern_5_2")

c(rho_12 = inter12$theta[1], rho_13 = inter13$theta[1], rho_23 = inter23$theta[1])
}
