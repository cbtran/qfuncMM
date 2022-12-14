#' @title Fit intra-regional model
#'
#' @description Fit intra-regional model
#'
#' @details For an order \code{q} B-splines (\code{q = degree + 1} since an intercept is used) with \code{k} internal knots 0 < \code{t_1} <...< \code{t_k} < T,
#' the number of B-spline basis equals \code{q + k}.
#'
#' @param parameters_init unrestricted initialization of parameters \eqn{{\phi}_{\gamma}^{(0)}, {\tau}_{\gamma}^{(0)}, {k}_{\gamma}^{(0)}} for 1 region 
#' @param X Data matrix of signals of 1 region
#' @param degree Degree of the piecewise polynomial. Default 3 for cubic splines.
#' @param nbasis Number of B-spline basis.
#' If \code{knots} is unspecified, the function choose \code{nbasis - degree - 1} \strong{internal} knots at suitable quantiles of \code{grid}.
#' If \code{knots} is specified, the value of \code{nbasis} will be \strong{ignored}.
#' @param knots \code{k} \strong{internal} breakpoints that define that spline.
#' @param dist_sqrd_mat Spatial squared distance matrix
#' @param time_sqrd_mat Temporal squared distance matrix
#' @param L Number of voxels 
#' @param M Number of time points
#' @param kernel_type Choice of spatial kernel. Defaul "matern_5_2".
#' @return List of 2 components:
#' \item{theta}{estimated intra-regional parameters \eqn{\hat{\phi}_\gamma, \hat{\tau}_\gamma, \hat{k}_\gamma}}
#' \item{nu}{fixed-effect estimate \eqn{\hat{\nu}}}
#'                                    
#' @useDynLib qfuncMM
#' @importFrom splines bs
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
#' dist_sqrd_mat_region1 <- get_dist_sqrd_mat(L, side_length, vxlID_1)
#' time_sqrd_mat <- (outer(1:M, 1:M, "-"))^2
#' fit_intra_model(X=simulated_data[[1]]$X_Region1,
#'                degree = 3,
#'                nbasis = 10,
#'                dist_sqrd_mat=dist_sqrd_mat_region1,
#'                time_sqrd_mat=time_sqrd_mat,
#'                L=L, M=M,
#'                kernel_type = "matern_5_2")
#' @export

fit_intra_model <- function(
    parameters_init = rep(0,3),
    X, 
    degree = 3,
    nbasis=NULL,
    knots=NULL,
    dist_sqrd_mat, 
    time_sqrd_mat,
    L, 
    M, 
    kernel_type="matern_5_2") {
  
  if (degree<2) {
    stop("Method requires degree to be at least 2.\n")
  }
  
  if (is.null(knots) & is.null(nbasis)) {
    stop("Either knots or nbasis needs to be specified.\n")
  } else if (is.null(knots) & !is.null(nbasis)) {
    if (nbasis <= degree) {
      stop("nbasis is too small.\n")
    }
  } else if (!is.null(knots)) {
    k = length(knots)
    nbasis = k + degree + 1
  }
  
  X_df <- data.frame(signal=X, time=rep(1:M, times=L))
  fitR <- lm(signal ~  -1 + bs(time, df=nbasis, knots=knots, degree=degree, intercept=T), data=X_df)
  G <- bs(rep(1:M, L), df=nbasis, knots=knots, degree=degree, intercept=T)
  
  parameters_init_c <- c(parameters_init, coef(fitR))
  result <- opt_intra(parameters_init_c,
                      X, 
                      G,
                      dist_sqrd_mat, 
                      time_sqrd_mat,
                      L, M, 
                      kernel_type)
                                      
  region_para <- result$theta
  nu <- result$nu
  list(theta=region_para,
       nu=nu)
                                              
}