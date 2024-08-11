#include <RcppEnsmallen.h>
#include <stdexcept>

#include "OptInter.h"
#include "OptIntra.h"
#include "Rcpp/exceptions.h"
#include "Rcpp/iostream/Rstreambuf.h"
#include "Rcpp/vector/instantiation.h"
#include "armadillo"
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
Rcpp::List opt_intra(const arma::vec &theta_init, const arma::mat &X_region,
                     const arma::mat &voxel_coords,
                     const arma::mat &time_sqrd_mat, int kernel_type_id,
                     bool nugget_only, bool noiseless,
                     bool noiseless_profiled) {
  using namespace arma;
  // Necessary evil since we can't easily expose enums to R
  KernelType kernel_type = static_cast<KernelType>(kernel_type_id);

  // phi, tau, k, nugget
  // or phi, tau, nugget_over_k if profiled
  arma::mat theta_unrestrict = softminus(theta_init);
  arma::mat dist_sqrd_mat = squared_distance(voxel_coords);

  // Construct the objective function.
  std::unique_ptr<IOptIntra> opt_intra;
  if (noiseless_profiled) {
    opt_intra = std::make_unique<OptIntraNoiselessProfiled>(
        X_region, dist_sqrd_mat, time_sqrd_mat, kernel_type);
  } else if (noiseless)
    opt_intra = std::make_unique<OptIntraNoiseless>(X_region, dist_sqrd_mat,
                                                    time_sqrd_mat, kernel_type);
  else if (nugget_only)
    opt_intra = std::make_unique<OptIntraDiagTime>(X_region, dist_sqrd_mat,
                                                   time_sqrd_mat, kernel_type);
  else
    opt_intra = std::make_unique<OptIntra>(X_region, dist_sqrd_mat,
                                           time_sqrd_mat, kernel_type);

  // Create the L_BFGS optimizer with default parameters.
  ens::L_BFGS optimizer(20);
  optimizer.MaxIterations() = 100;
  // optimizer.MaxStep() = 10;

  // Run the optimization
  double optval;
  try {
    optval = optimizer.Optimize(*opt_intra, theta_unrestrict, ens::Report(1));
  } catch (std::runtime_error &re) {
    Rcpp::stop("Optimization failed " + std::string(re.what()));
  }
  vec theta = softplus(theta_unrestrict);
  if (noiseless_profiled) {
    vec theta_orig = arma::zeros(4);
    theta_orig(0) = theta(0);
    theta_orig(1) = theta(1);
    theta_orig(2) = opt_intra->GetKStar();
    theta_orig(3) = theta(2) * theta_orig(2);
    theta = theta_orig;
  }

  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("var_noise") =
                                opt_intra->GetNoiseVarianceEstimate(),
                            Rcpp::Named("eblue") = opt_intra->GetEBlue(),
                            Rcpp::Named("objval") = optval);
}

// [[Rcpp::export]]
Rcpp::List eval_stage1_nll(const arma::vec &theta, const arma::mat &X_region,
                           const arma::mat &voxel_coords,
                           const arma::mat &time_sqrd_mat, int kernel_type_id) {

  arma::mat theta_transformed = theta;
  theta_transformed = theta_transformed.transform([](double x) {
    if (x > 100)
      return x;
    return log(exp(x) - 1);
  });
  // Necessary evil since we can't easily expose enums to R
  KernelType kernel_type = static_cast<KernelType>(kernel_type_id);
  arma::mat dist_sqrd_mat = squared_distance(voxel_coords);

  // Construct the objective function.
  OptIntra opt_intra(X_region, dist_sqrd_mat, time_sqrd_mat, kernel_type,
                     false);

  arma::mat grad = arma::zeros(4, 1);
  double nll = opt_intra.EvaluateWithGradient(theta_transformed, grad);

  return Rcpp::List::create(Rcpp::Named("nll") = nll,
                            Rcpp::Named("grad") = grad);
}

class StatusCallback {

  bool diag_time_;

public:
  StatusCallback(bool diag_time) : diag_time_(diag_time){};

  template <typename OptimizerType, typename FunctionType>
  void Gradient(OptimizerType &optimizer, FunctionType &function,
                const arma::mat &coordinates, const arma::mat &gradient) {
    // Rcpp::Rcout << "Grad norm: " << arma::norm(gradient) << std::endl;
    Rcpp::Rcout << "params: " << sigmoid_inv(coordinates(0), -1, 1) << ", "
                << softplus(coordinates(1)) << ", " << softplus(coordinates(2));

    if (!diag_time_) {
      Rcpp::Rcout << ", " << softplus(coordinates(3)) << ", "
                  << softplus(coordinates(4));
    }

    Rcpp::Rcout << std::endl;
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
Rcpp::List opt_inter(const arma::vec &theta_init, const arma::mat &dataRegion1,
                     const arma::mat &dataRegion2,
                     const arma::mat &voxel_coords_1,
                     const arma::mat &voxel_coords_2,
                     const arma::mat &time_sqrd_mat,
                     const arma::vec &stage1ParamsRegion1,
                     const arma::vec &stage1ParamsRegion2, int kernel_type_id,
                     bool diag_time) {
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
      stage1ParamsRegion1(2) * get_cor_mat(KernelType::Rbf, time_sqrd_mat,
                                           stage1ParamsRegion1(1)) +
          stage1ParamsRegion1(3) *
              arma::eye(dataRegion1.n_rows, dataRegion1.n_rows));

  const mat block_region_2 = arma::kron(
      get_cor_mat(kernel_type, sqrd_dist_region2, stage1ParamsRegion2(0)),
      stage1ParamsRegion2(2) * get_cor_mat(KernelType::Rbf, time_sqrd_mat,
                                           stage1ParamsRegion2(1)) +
          stage1ParamsRegion2(3) *
              arma::eye(dataRegion2.n_rows, dataRegion2.n_rows));

  // Construct the objective function.
  std::unique_ptr<IOptInter> opt_inter;
  if (diag_time) {
    opt_inter = std::make_unique<OptInterDiagTime>(
        dataRegion1, dataRegion2, stage1ParamsRegion1, stage1ParamsRegion2,
        block_region_1, block_region_2, time_sqrd_mat);
  } else {
    opt_inter = std::make_unique<OptInter>(
        dataRegion1, dataRegion2, stage1ParamsRegion1, stage1ParamsRegion2,
        block_region_1, block_region_2, time_sqrd_mat);
  }

  ens::L_BFGS optimizer(10);
  optimizer.MaxIterations() = 50;
  optimizer.MaxLineSearchTrials() = 10;
  optimizer.MinGradientNorm() = 1e-4;

  ens::StoreBestCoordinates<mat> cb;
  optimizer.Optimize(*opt_inter, theta_vec, ens::GradClipByNorm(100),
                     StatusCallback(diag_time), ens::Report(1), cb);

  Rcpp::Rcout << "NegLL Final: " << std::setprecision(10) << cb.BestObjective()
              << std::endl;

  vec best(cb.BestCoordinates());
  vec result(5);
  result(0) = sigmoid_inv(best(0), -1, 1); // rho
  result(1) = softplus(best(1));           // kEta1
  result(2) = softplus(best(2));           // kEta2

  if (diag_time) {
    result(3) = 0; // tauEta
    result(4) = 1; // nuggetEta
  } else {
    result(3) = softplus(best(3)); // tauEta
    result(4) = softplus(best(4)); // nuggetEta
  }

  std::pair<double, double> noise_estimates =
      opt_inter->GetNoiseVarianceEstimates();
  vec var_noise = {noise_estimates.first, noise_estimates.second};

  return Rcpp::List::create(Rcpp::Named("theta") = result,
                            Rcpp::Named("var_noise") = var_noise,
                            Rcpp::Named("objective") = cb.BestObjective());
}
