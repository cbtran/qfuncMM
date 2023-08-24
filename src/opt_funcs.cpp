#include <RcppEnsmallen.h>

#include "OptInter.h"
#include "OptInterOld.h"
#include "OptIntra.h"
#include "OptIntraOld.h"
#include "OptIntraFixed.h"
#include "get_cor_mat.h"
#include "helper.h"
// [[Rcpp::depends(RcppEnsmallen)]]

/*****************************************************************************
 Intra-regional model
*****************************************************************************/

//' @title Fit intra-regional model using L-BFGS
//' @param theta_init unrestricted initialization of parameters for 1 region
//' @param X_region Data matrix of signals of 1 region
//' @param Z_region fixed-effects design matrix of 1 region
//' @param voxel_coords Voxel coordinates for the region
//' @param time_sqrd_mat Temporal squared distance matrix
//' @param num_voxel Number of voxels
//' @param num_timept Number of time points
//' @param kernel_type_id Choice of spatial kernel
//' @return List of 2 components:
//' \item{theta}{estimated intra-regional parameters}
//' \item{nu}{fixed-effect estimate}
//' @noRd
// [[Rcpp::export]]
Rcpp::List opt_intra(const arma::vec& theta_init, const arma::mat& X_region,
                     const arma::mat& Z_region, const arma::mat& voxel_coords,
                     const arma::mat& time_sqrd_mat, int num_voxel,
                     int num_timept, int kernel_type_id) {
  // Necessary evil since we can't easily expose enums to R
  KernelType kernel_type = static_cast<KernelType>(kernel_type_id);

  // Read in parameters inits
  arma::mat nu = theta_init.tail_rows(theta_init.n_elem - 3);

  // Update basis coefficents
  arma::mat theta_vec(theta_init.n_elem, 1);
  theta_vec.col(0) = theta_init;

  arma::mat dist_sqrd_mat = squared_distance(voxel_coords);

  // Construct the objective function.
  OptIntraOld opt_intra(X_region, Z_region, dist_sqrd_mat, time_sqrd_mat,
                        num_voxel, num_timept, nu, kernel_type);

  // Create the L_BFGS optimizer with default parameters.
  ens::L_BFGS optimizer(20);  // L-BFGS optimizer with 10 memory points
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

// [[Rcpp::export]]
Rcpp::List opt_intra_new(const arma::vec& theta_init, const arma::mat& X_region,
                         const arma::mat& voxel_coords,
                         const arma::mat& time_sqrd_mat, int num_voxel,
                         int num_timept, int kernel_type_id) {
  // Necessary evil since we can't easily expose enums to R
  KernelType kernel_type = static_cast<KernelType>(kernel_type_id);

  // Update basis coefficents
  arma::mat theta = theta_init;

  arma::mat dist_sqrd_mat = squared_distance(voxel_coords);

  // Construct the objective function.
  OptIntra opt_intra(X_region, dist_sqrd_mat, time_sqrd_mat, num_voxel,
                     num_timept, kernel_type);

  // Create the L_BFGS optimizer with default parameters.
  ens::L_BFGS optimizer(20);
  optimizer.MaxIterations() = 100;
  optimizer.MaxLineSearchTrials() = 10;
  optimizer.MinGradientNorm() = 1e-4;
//   optimizer.MaxStep() = 0.5;

  // // Run the optimization
  optimizer.Optimize(opt_intra, theta);
  theta = softplus(theta);

  return Rcpp::List::create(
    Rcpp::Named("theta") = theta,
    Rcpp::Named("var_noise") = opt_intra.GetNoiseVarianceEstimate());
}

// For testing: optimize one parameter holding others fixed
Rcpp::List opt_intra_fixed(const arma::vec& theta_init,
                           const arma::vec& theta_fixed,
                           const arma::mat& X_region,
                           const arma::mat& voxel_coords,
                           const arma::mat& time_sqrd_mat, int num_voxel,
                           int num_timept, int kernel_type_id) {
  // Necessary evil since we can't easily expose enums to R
  KernelType kernel_type = static_cast<KernelType>(kernel_type_id);

  // Update basis coefficents
  arma::mat theta = theta_init;

  arma::mat dist_sqrd_mat = squared_distance(voxel_coords);

  // Construct the objective function.
  OptIntraFixed opt_intra(X_region, dist_sqrd_mat, time_sqrd_mat, num_voxel,
                          num_timept, kernel_type, theta_fixed);

  // Create the L_BFGS optimizer with default parameters.
  ens::L_BFGS optimizer(20);
  optimizer.MaxIterations() = 100;
  optimizer.MaxLineSearchTrials() = 10;
  optimizer.MinGradientNorm() = 1e-4;
//   optimizer.MaxStep() = 0.5;

  // // Run the optimization
  optimizer.Optimize(opt_intra, theta);
  theta = softplus(theta);

  return Rcpp::List::create(
    Rcpp::Named("theta") = theta,
    Rcpp::Named("var_noise") = opt_intra.GetNoiseVarianceEstimate());
}

/*****************************************************************************
 Inter-regional model
*****************************************************************************/

//' @title Fit inter-regional model using L-BFGS
//' @param theta_init unrestricted initialization of parameters for
//'  inter-regional model
//' @param X Data matrix of signals of 2 regions
//' @param Z fixed-effects design matrix of 2 regions
//' @param voxel_coords_1 Region 1 voxel coordinates
//' @param voxel_coords_2 Region 2 voxel coordinates
//' @param kernel_type_id Choice of spatial kernel
//' @param stage1_regional Regional parameters from stage 1
//' @return List of 3 components:
//'\item{theta}{estimated inter-regional parameters}
//'\item{asymptotic_var}{asymptotic variance of transformed correlation
//'     coefficient}
//' \item{rho_transformed}{Fisher transformation of correlation coefficient}
//' @noRd
// [[Rcpp::export]]
Rcpp::List opt_inter(const arma::vec& theta_init, const arma::mat& X,
                     const arma::mat& Z, const arma::mat& voxel_coords_1,
                     const arma::mat& voxel_coords_2,
                     const arma::mat& time_sqrd_mat,
                     const arma::vec& stage1_regional, int kernel_type_id) {
  // Necessary evil since we can't easily expose enums to R
  KernelType kernel_type = static_cast<KernelType>(kernel_type_id);

  // Read in parameters inits
  arma::mat theta_vec(6, 1);
  theta_vec.col(0) = theta_init;

  arma::mat sqrd_dist_region1 = squared_distance(voxel_coords_1);
  arma::mat sqrd_dist_region2 = squared_distance(voxel_coords_2);

  const arma::mat block_region_1 = arma::kron(
      get_cor_mat(kernel_type, sqrd_dist_region1, stage1_regional(0)),
      stage1_regional(2) *
          get_cor_mat(KernelType::Rbf, time_sqrd_mat, stage1_regional(1)));

  const arma::mat block_region_2 = arma::kron(
      get_cor_mat(kernel_type, sqrd_dist_region2, stage1_regional(3)),
      stage1_regional(5) *
          get_cor_mat(KernelType::Rbf, time_sqrd_mat, stage1_regional(4)));

  int num_voxel1 = sqrd_dist_region1.n_cols;
  int num_voxel2 = sqrd_dist_region2.n_cols;
  int num_timept = time_sqrd_mat.n_cols;
  // Construct the objective function.
  OptInterOld opt_schur_rho_f(X, Z, num_voxel1, num_voxel2, num_timept,
                           block_region_1, block_region_2, time_sqrd_mat);

  // Create the L_BFGS optimizer with default parameters.
  ens::L_BFGS optimizer(10);  // L-BFGS optimizer with 10 memory points
  // Maximum number of iterations
  optimizer.MaxIterations() = 50;
  optimizer.MaxLineSearchTrials() = 10;
  // Relative error
  optimizer.MinGradientNorm() = 1e-4;

  // Run the optimization
  optimizer.Optimize(opt_schur_rho_f, theta_vec);

  // Return rho
  theta_vec(0) = sigmoid_inv(theta_vec(0), -1, 1);
  theta_vec(1) = softplus(theta_vec(1));
  theta_vec(2) = softplus(theta_vec(2));
  theta_vec(3) = softplus(theta_vec(3));

  Rcpp::List asymp_var = asymptotic_variance(
      block_region_1, block_region_2, time_sqrd_mat, num_voxel1, num_voxel2,
      num_timept, theta_vec(2), theta_vec(1), theta_vec(3), theta_vec(0), Z);

  return Rcpp::List::create(Rcpp::Named("theta") = theta_vec,
                            Rcpp::Named("asymptotic_var") = asymp_var[0],
                            Rcpp::Named("rho_transformed") = asymp_var[1]);
}

// [[Rcpp::export]]
Rcpp::List opt_inter_new(const arma::vec& theta_init,
                     const arma::mat& dataRegion1,
                     const arma::mat& dataRegion2,
                     const arma::mat& voxel_coords_1,
                     const arma::mat& voxel_coords_2,
                     const arma::mat& time_sqrd_mat,
                     const arma::vec& stage1ParamsRegion1,
                     const arma::vec& stage1ParamsRegion2,
                     int kernel_type_id) {
  // Necessary evil since we can't easily expose enums to R
  KernelType kernel_type = static_cast<KernelType>(kernel_type_id);

  // Read in parameters inits
  arma::mat theta_vec(5, 1);
  theta_vec.col(0) = theta_init;

  arma::mat sqrd_dist_region1 = squared_distance(voxel_coords_1);
  arma::mat sqrd_dist_region2 = squared_distance(voxel_coords_2);

  // These kronecker products are expensive to compute, so we do them out here
  // instead of inside the optimization class
  // Stage 1 param list: phi_gamma, tau_gamma, k_gamma, nugget_gamma, var_noise
  const arma::mat block_region_1 = arma::kron(
      get_cor_mat(kernel_type, sqrd_dist_region1, stage1ParamsRegion1(0)),
      stage1ParamsRegion1(2) *
          get_cor_mat(KernelType::Rbf, time_sqrd_mat, stage1ParamsRegion1(1)) +
          stage1ParamsRegion1(3) * arma::eye(dataRegion1.n_rows, dataRegion1.n_rows));

  const arma::mat block_region_2 = arma::kron(
      get_cor_mat(kernel_type, sqrd_dist_region2, stage1ParamsRegion2(0)),
      stage1ParamsRegion2(2) *
          get_cor_mat(KernelType::Rbf, time_sqrd_mat, stage1ParamsRegion2(1)) +
          stage1ParamsRegion2(3) * arma::eye(dataRegion2.n_rows, dataRegion2.n_rows));

  // Construct the objective function.
  OptInter opt_schur_rho_f(dataRegion1, dataRegion2, stage1ParamsRegion1, stage1ParamsRegion2,
                           block_region_1, block_region_2, time_sqrd_mat);

  ens::SPSA optimizer;
  // Run the optimization
  optimizer.Optimize(opt_schur_rho_f, theta_vec);
  optimizer.StepSize() = 0.5;

  // Return rho
  theta_vec(0) = sigmoid_inv(theta_vec(0), -1, 1);
  theta_vec(1) = softplus(theta_vec(1));
  theta_vec(2) = softplus(theta_vec(2));
  theta_vec(3) = softplus(theta_vec(3));

  return Rcpp::List::create(Rcpp::Named("theta") = theta_vec);
}