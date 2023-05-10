#include <RcppEnsmallen.h>
#include "helper.h"
#include "OptIntra.h"
#include "OptInter.h"
// [[Rcpp::depends(RcppEnsmallen)]]

/*****************************************************************************
 Intra-regional model
*****************************************************************************/

//' @title Fit intra-regional model using L-BFGS
//' @param theta_init unrestricted initialization of parameters for 1 region
//' @param X_region Data matrix of signals of 1 region
//' @param Z_region fixed-effects design matrix of 1 region
//' @param dist_sqrd_mat Spatial squared distance matrix
//' @param time_sqrd_mat Temporal squared distance matrix
//' @param L Number of voxels
//' @param M Number of time points
//' @param kernel_type Choice of spatial kernel
//' @return List of 2 components:
//' \item{theta}{estimated intra-regional parameters}
//' \item{nu}{fixed-effect estimate}
//' @noRd
// [[Rcpp::export]]
Rcpp::List opt_intra(const arma::vec& theta_init,
                     const arma::mat& X_region,
                     const arma::mat& Z_region,
                     const arma::mat& dist_sqrd_mat,
                     const arma::mat& time_sqrd_mat,
                     int L, int M,
                     std::string kernel_type) {


  // Read in parameters inits
  arma::mat nu = theta_init.tail_rows(theta_init.n_elem - 3);

  // Update basis coefficents
  arma::mat theta_vec(theta_init.n_elem, 1);
  theta_vec.col(0) = theta_init;

  // Construct the objective function.
  OptIntra opt_intra(X_region,
                     Z_region,
                     dist_sqrd_mat,
                     time_sqrd_mat,
                     L, M, nu,
                     kernel_type);


  // Create the L_BFGS optimizer with default parameters.
  ens::L_BFGS optimizer(20); // L-BFGS optimizer with 10 memory points
  // Maximum number of iterations
  optimizer.MaxIterations() = 100;
  optimizer.MaxLineSearchTrials() = 10;
  // Relative error
  optimizer.MinGradientNorm() = 1e-4;

  // Run the optimization
  optimizer.Optimize(opt_intra, theta_vec);
  arma::vec theta = softplus(theta_vec.head_rows(3));
  nu = theta_vec.tail_rows(Z_region.n_cols);

  // Return
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("nu") = nu);
}



/*****************************************************************************
 Inter-regional model
*****************************************************************************/

//' @title Fit inter-regional model using L-BFGS
//' @param theta_init unrestricted initialization of parameters  for inter-regional model
//' @param X Data matrix of signals of 2 regions
//' @param Z fixed-effects design matrix of 2 regions
//' @param dist_sqrdMat_1 Block component for that region 1
//' @param dist_sqrdMat_2 Block component for that region 2
//' @param kernel_type Choice of spatial kernel
//' @param stage1_regional Regional parameters from stage 1
//' @return List of 3 components:
//' \item{theta}{estimated inter-regional parameters}
//' \item{asymptotic_var}{asymptotic variance of transformed correlation coefficient}
//' \item{rho_transformed}{Fisher transformation of correlation coefficient}
//' @noRd
// [[Rcpp::export]]
Rcpp::List opt_inter(const arma::vec& theta_init,
                     const arma::mat& X,
                     const arma::mat& Z,
                     const arma::mat& dist_sqrdMat_1,
                     const arma::mat& dist_sqrdMat_2,
                     const arma::mat& time_sqrd_mat,
                     const arma::vec& stage1_regional,
                     std::string kernel_type) {

  // Read in parameters inits
  arma::mat theta_vec(6, 1);
  theta_vec.col(0) = theta_init;

  const arma::mat block_region_1 = arma::kron(get_cor_mat(kernel_type, dist_sqrdMat_1, stage1_regional(0)),
                                      stage1_regional(2) * get_cor_mat("rbf", time_sqrd_mat, stage1_regional(1)));

  const arma::mat block_region_2 = arma::kron(get_cor_mat(kernel_type, dist_sqrdMat_2, stage1_regional(3)),
                                      stage1_regional(5) * get_cor_mat("rbf", time_sqrd_mat, stage1_regional(4)));

  int L1 = dist_sqrdMat_1.n_cols;
  int L2 = dist_sqrdMat_2.n_cols;
  int M = time_sqrd_mat.n_cols;
  // Construct the objective function.
  OptInter opt_schur_rho_f(X, Z, L1, L2, M, block_region_1, block_region_2, time_sqrd_mat);

  // Create the L_BFGS optimizer with default parameters.
  ens::L_BFGS optimizer(10); // L-BFGS optimizer with 10 memory points
  // Maximum number of iterations
  optimizer.MaxIterations() = 50;
  optimizer.MaxLineSearchTrials() = 10;
  // Relative error
  optimizer.MinGradientNorm() = 1e-4;

  // Run the optimization
  optimizer.Optimize(opt_schur_rho_f, theta_vec);

  //Return rho
  theta_vec(0) = sigmoid_inv(theta_vec(0), -1 ,1);
  theta_vec(1) = softplus(theta_vec(1));
  theta_vec(2) = softplus(theta_vec(2));
  theta_vec(3) = softplus(theta_vec(3));

  Rcpp::List asymp_var = asymptotic_variance(block_region_1,
                                             block_region_2,
                                             time_sqrd_mat,
                                             L1, L2, M,
                                             theta_vec(2),
                                             theta_vec(1),
                                             theta_vec(3),
                                             theta_vec(0),
                                             Z,
                                             kernel_type);

  return Rcpp::List::create(Rcpp::Named("theta") = theta_vec,
                            Rcpp::Named("asymptotic_var") = asymp_var[0],
                            Rcpp::Named("rho_transformed") = asymp_var[1]);

}