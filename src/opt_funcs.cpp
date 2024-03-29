#include <RcppEnsmallen.h>

#include "OptInter.h"
#include "OptIntra.h"
#include "Rcpp/vector/instantiation.h"
#include "ensmallen_bits/callbacks/grad_clip_by_norm.hpp"
#include "ensmallen_bits/callbacks/store_best_coordinates.hpp"
#include "get_cor_mat.h"
#include "helper.h"
// [[Rcpp::depends(RcppEnsmallen)]]

/*****************************************************************************
 Intra-regional model
*****************************************************************************/

//' @title Fit intra-regional model using L-BFGS
//' @param theta_init unrestricted initialization of parameters for 1 region
//' @param X_region Vectorized (LM) - data matrix of signals of 1 region
//' @param voxel_coords L x 3 matrix of voxel coordinates
//' @param time_sqrd_mat M x M temporal squared distance matrix
//' @param kernel_type_id Choice of spatial kernel
//' @return List of 2 components:
//'   theta: Estimated intra-regional parameters
//'   var_noise: Estimated noise variance
//' @noRd
// [[Rcpp::export]]
Rcpp::List opt_intra(const arma::vec& theta_init,
                     const arma::mat& X_region,
                     const arma::mat& voxel_coords,
                     const arma::mat& time_sqrd_mat,
                     int kernel_type_id) {
  // Necessary evil since we can't easily expose enums to R
  KernelType kernel_type = static_cast<KernelType>(kernel_type_id);

  // Update basis coefficents
  arma::mat theta = theta_init;
  arma::mat dist_sqrd_mat = squared_distance(voxel_coords);

  // Construct the objective function.
  OptIntra opt_intra(X_region, dist_sqrd_mat, time_sqrd_mat, kernel_type);

  // Create the L_BFGS optimizer with default parameters.
  ens::L_BFGS optimizer(20);
  optimizer.MaxIterations() = 100;
  optimizer.MaxLineSearchTrials() = 10;
  optimizer.MinGradientNorm() = 1e-4;

  // Run the optimization
  optimizer.Optimize(opt_intra, theta);
  theta = softplus(theta);

  return Rcpp::List::create(
    Rcpp::Named("theta") = theta,
    Rcpp::Named("var_noise") = opt_intra.GetNoiseVarianceEstimate());
}

class StatusCallback
{
  public:
    StatusCallback() {};

  template<typename OptimizerType, typename FunctionType>
  void Gradient(OptimizerType& optimizer,
                FunctionType& function,
                const arma::mat& coordinates,
                const arma::mat& gradient)
  {
    // Rcpp::Rcout << "Grad norm: " << arma::norm(gradient) << std::endl;
    Rcpp::Rcout << "params: " << sigmoid_inv(coordinates(0), -1, 1) << ", "
                            << softplus(coordinates(1)) << ", "
                            << softplus(coordinates(2)) << ", "
                            << softplus(coordinates(3)) << ", "
                            << softplus(coordinates(4)) << std::endl;

  }
};


/*****************************************************************************
 Inter-regional model
*****************************************************************************/

//' @title Fit inter-regional model using L-BFGS
//' @param theta_init unrestricted initialization of parameters for
//'  inter-regional model
//' @param dataRegion1 Vectorized region 1 data matrix
//' @param dataRegion2 Vectorized region 2 data matrix
//' @param voxel_coords_1 Region 1 voxel coordinates
//' @param voxel_coords_2 Region 2 voxel coordinates
//' @param time_sqrd_mat M x M temporal squared distance matrix
//' @param stage1ParamsRegion1 Regional parameters from stage 1
//' @param stage1ParamsRegion2 Regional parameters from stage 1
//' @param kernel_type_id Choice of spatial kernel
//' @return List of 3 components:
//'   theta: Estimated inter-regional parameters
//'   var_noise: Estimated noise variance
//'   objective: optimal loss (negiatve log-likelihood) found.
//' @noRd
// [[Rcpp::export]]
Rcpp::List opt_inter(const arma::vec& theta_init,
                     const arma::mat& dataRegion1,
                     const arma::mat& dataRegion2,
                     const arma::mat& voxel_coords_1,
                     const arma::mat& voxel_coords_2,
                     const arma::mat& time_sqrd_mat,
                     const arma::vec& stage1ParamsRegion1,
                     const arma::vec& stage1ParamsRegion2,
                     int kernel_type_id) {
  using arma::mat;
  using arma::vec;

  // Necessary evil since we can't easily expose enums to R
  KernelType kernel_type = static_cast<KernelType>(kernel_type_id);

  // Read in parameters inits
  mat theta_vec(theta_init);

  mat sqrd_dist_region1 = squared_distance(voxel_coords_1);
  mat sqrd_dist_region2 = squared_distance(voxel_coords_2);

  // These kronecker products are expensive to compute, so we do them out here
  // instead of inside the optimization class
  // Stage 1 param list: phi_gamma, tau_gamma, k_gamma, nugget_gamma, var_noise
  const mat block_region_1 = arma::kron(
      get_cor_mat(kernel_type, sqrd_dist_region1, stage1ParamsRegion1(0)),
      stage1ParamsRegion1(2) *
          get_cor_mat(KernelType::Rbf, time_sqrd_mat, stage1ParamsRegion1(1)) +
          stage1ParamsRegion1(3) * arma::eye(dataRegion1.n_rows, dataRegion1.n_rows));

  const mat block_region_2 = arma::kron(
      get_cor_mat(kernel_type, sqrd_dist_region2, stage1ParamsRegion2(0)),
      stage1ParamsRegion2(2) *
          get_cor_mat(KernelType::Rbf, time_sqrd_mat, stage1ParamsRegion2(1)) +
          stage1ParamsRegion2(3) * arma::eye(dataRegion2.n_rows, dataRegion2.n_rows));

  OptInter objective(dataRegion1, dataRegion2,
                     stage1ParamsRegion1, stage1ParamsRegion2,
                     block_region_1, block_region_2, time_sqrd_mat);

  ens::L_BFGS optimizer(10);
  optimizer.MaxIterations() = 50;
  optimizer.MaxLineSearchTrials() = 10;
  optimizer.MinGradientNorm() = 1e-4;

  ens::StoreBestCoordinates<mat> cb;
  optimizer.Optimize(objective, theta_vec, ens::GradClipByNorm(100), StatusCallback(), cb);

  Rcpp::Rcout << "NegLL Final: " << std::setprecision(10) << cb.BestObjective() << std::endl;

  vec best(cb.BestCoordinates());
  best(0) = sigmoid_inv(best(0), -1, 1); // rho
  best(1) = softplus(best(1)); // kEta1
  best(2) = softplus(best(2)); // kEta2
  best(3) = softplus(best(3)); // tauEta
  best(4) = softplus(best(4)); // nugget_eta

  std::pair<double, double> noise_estimates =
    objective.GetNoiseVarianceEstimates();
  vec var_noise = {noise_estimates.first, noise_estimates.second};

  return Rcpp::List::create(
    Rcpp::Named("theta") = best,
    Rcpp::Named("var_noise") = var_noise,
    Rcpp::Named("objective") = cb.BestObjective());
}
