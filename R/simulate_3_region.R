#' Simulate signals from 3 regions
#'
#'
#' @param L1 number of voxels in region 1
#' @param L2 number of voxels in region 2
#' @param L3 number of voxels in region 3
#' @param side_length side length of 3D lattice where voxels are sampled
#' @param M number of timepoints
#' @param theta true covariance matrices parameters
#' @param rho_vec vector of inter-regional correlations \eqn{(\rho_{12}, \rho_{13}, \rho_{23})}
#' @param sigma_sqrd noise variance. Default 1.
#' @param mu_1 \code{M}-by-\code{1} vector of mean of signal from region 1
#' @param mu_2 \code{M}-by-\code{1} vector of mean of signal from region 2
#' @param mu_3 \code{M}-by-\code{1} vector of mean of signal from region 3
#' @param vxlID_1 \code{L_1}-by-\code{1} vector of voxel ID of region 1
#' @param vxlID_2 \code{L_2}-by-\code{1} vector of voxel ID of region 2
#' @param vxlID_3 \code{L_3}-by-\code{1} vector of voxel ID of region 3
#' @param random_seed random seed
#' @param num_sim number of simulation
#' @param out_folder_name folder to save simulated data
#' @param plot_it If True, plot of simulated data is saved in \code{out_folder_name}. Default True.
#' @param C_kernel_type Choice of spatial kernel. Defaul "matern_5_2".
#' @return obs_signal Simulated signal
#'
#' @useDynLib qfuncMM
#' @importFrom MASS mvrnorm
#' @importFrom graphics matplot
#' @importFrom Rcpp sourceCpp
#'
#' @examples
#' L <- 20 # Numbers of voxels
#' side_length <- 7 # Side length of each region
#' M <- 30 # Numbers of timepoints
#' num_sim <- 1 # Numbers of simulation

#' # Generate voxels for each region
#' set.seed(1)
#' vxlID_1 <- sample(1:(side_length^3), L, replace = FALSE)
#' set.seed(2)
#' vxlID_2 <- sample(1:(side_length^3), L, replace = FALSE)
#' set.seed(4)
#' vxlID_3 <- sample(1:(side_length^3), L, replace = FALSE)
#' 
#' # rho
#' rho_vec <- data.frame(rho12=0.1, rho13=0.35, rho23=0.6)
#' # Parameters
#' parameters_true <- data.frame(tau_eta = 1/4, k_eta = 0.5,
#' phi_gamma_1=1, tau_gamma_1=1/2, k_gamma_1=2,
#' phi_gamma_2=1, tau_gamma_2=1/2, k_gamma_2=2,
#' phi_gamma_3=1, tau_gamma_3=1/2, k_gamma_3=2,
#' nugget=0.1, mu_1 = 1, mu_2 = 10, mu_3 = 20)
#'
#' simulated_data = simulate_3_region(L1=L, L2=L, L3=L,
#'                                    side_length=side_length, 
#'                                    M=M, 
#'                                    theta=parameters_true, 
#'                                    rho_vec=rho_vec, 
#'                                    sigma_sqrd=1,
#'                                    mu_1=rep(parameters_true$mu_1, M),
#'                                    mu_2=rep(parameters_true$mu_2, M),
#'                                    mu_3=rep(parameters_true$mu_3, M),
#'                                    vxlID_1=vxlID_1, 
#'                                    vxlID_2=vxlID_2, 
#'                                    vxlID_3=vxlID_3,
#'                                    random_seed=1, num_sim=1,
#'                                    C_kernel_type="matern_5_2")
#' plot_mat <- matrix(c(simulated_data[[1]]$X_Region1, simulated_data[[1]]$X_Region2, simulated_data[[1]]$X_Region3), 3*L, M, byrow=TRUE)
#' matplot(t(plot_mat), type="l", xlab="Time points", ylab="Signal")
#' @export
simulate_3_region <- function(L1,
                              L2,
                              L3,
                              side_length,
                              M,
                              theta,
                              rho_vec,
                              sigma_sqrd=1,
                              mu_1,
                              mu_2,
                              mu_3,
                              vxlID_1,
                              vxlID_2,
                              vxlID_3,
                              random_seed=1,
                              num_sim,
                              out_folder_name,
                              plot_it=T,
                              C_kernel_type="matern_5_2") {
                            
  
  # Get distance and time matrices
  dist_sqrd_mat_region1 <- get_dist_sqrd_mat(L1, side_length, vxlID_1)
  dist_sqrd_mat_region2 <- get_dist_sqrd_mat(L2, side_length, vxlID_2)
  dist_sqrd_mat_region3 <- get_dist_sqrd_mat(L3, side_length, vxlID_3)
  timeSqrd_mat <- (outer(1:M, 1:M, "-"))^2
  
  # Create covariance matrices
  
  # Covariance of eta effect
  rhoMat <- matrix(c(1,rho_vec$rho12,rho_vec$rho13, 
                     rho_vec$rho12,1,rho_vec$rho23, 
                     rho_vec$rho13,rho_vec$rho23,1),3,3)
  A <- theta$k_eta * get_cor_mat("rbf", timeSqrd_mat, theta$tau_eta) + theta$nugget * diag(M)
  etaSigma <- kronecker(rhoMat, A)
  
  # Matrices of gamma effects. C is spatial correlation matrix, B is temporal covariance matrix
  
  ## Region 1
  C1 <- get_cor_mat(C_kernel_type, dist_sqrd_mat_region1, theta$phi_gamma_1)
  B1 <- theta$k_gamma_1 * get_cor_mat("rbf", timeSqrd_mat, theta$tau_gamma_1)
  gammaSigma1 <- kronecker(C1, B1)
  
  ## Region 2
  C2 <- get_cor_mat(C_kernel_type, dist_sqrd_mat_region2, theta$phi_gamma_2)
  B2 <- theta$k_gamma_2 * get_cor_mat("rbf", timeSqrd_mat, theta$tau_gamma_2)
  gammaSigma2 <- kronecker(C2, B2)
  
  ## Region 3
  C3 <- get_cor_mat(C_kernel_type, dist_sqrd_mat_region3, theta$phi_gamma_3)
  B3 <- theta$k_gamma_3 * get_cor_mat("rbf", timeSqrd_mat, theta$tau_gamma_3)
  gammaSigma3 <- kronecker(C3, B3)
  
  if(!missing(random_seed)) {
    set.seed(random_seed)
  }
  eta <- mvrnorm(num_sim, mu = rep(0, 3*M), Sigma = sigma_sqrd*etaSigma)
  gammaR1 <- mvrnorm(num_sim, mu = rep(0, L1*M), Sigma = sigma_sqrd*gammaSigma1)
  gammaR2 <- mvrnorm(num_sim, mu = rep(0, L2*M), Sigma = sigma_sqrd*gammaSigma2)
  gammaR3 <- mvrnorm(num_sim, mu = rep(0, L3*M), Sigma = sigma_sqrd*gammaSigma3)
  

  obs_signal <- list()
  for(i in 1:num_sim) {
    # Generate iid noise
    set.seed(1000+i)
    epsilon <- rep(0, (L1+L2+L3)*M) + diag(rep(sqrt(sigma_sqrd), (L1+L2+L3)*M)) %*% rnorm((L1+L2+L3)*M)
    
    # Combine eta and gamma effects
    obs_signal[[i]] = list(X_Region1 = matrix(rep(mu_1, L1) + rep(eta[i,1:M], L1) + gammaR1[i,] + epsilon[1:(L1*M)], ncol=1),
                           X_Region2 = matrix(rep(mu_2, L2) + rep(eta[i,(M+1):(2*M)], L2) + gammaR2[i,] + epsilon[(L1*M+1):((L1+L2)*M)], ncol=1),
                           X_Region3 = matrix(rep(mu_3, L3) + rep(eta[i,(2*M+1):(3*M)], L3) + gammaR3[i,] + epsilon[((L1+L2)*M+1):((L1+L2+L3)*M)], ncol=1)
    )
  }
  
  # Write out data if folder name is provided
  if(!missing(out_folder_name)) {
    saveRDS(obs_signal, file=paste0(out_folder_name, "/sim-data.rds"))
    write.csv(eta, paste0(out_folder_name, "/sim-eta.csv"))
    if (plot_it) {
      for(i in 1:num_sim) {
        pdf(paste0(out_folder_name, "/sim-", i, ".pdf"))
        plot_mat <- matrix(c(obs_signal[[i]]$X_Region1, obs_signal[[i]]$X_Region2, obs_signal[[i]]$X_Region3), L1+L2+L3, M, byrow=TRUE)
        matplot(t(plot_mat), type="l", xlab="Time points", ylab="Signal")
        dev.off()
      }
    }
  }
  
  # Return data
  return(obs_signal)
}














