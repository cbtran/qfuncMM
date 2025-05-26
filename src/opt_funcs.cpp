#include <RcppEnsmallen.h>
#include <stdexcept>

#include "OptInter.h"
#include "OptIntra.h"
#include "Rcpp/exceptions.h"
#include "Rcpp/vector/instantiation.h"
#include "armadillo"
#include "cov_setting.h"
#include "ensmallen_bits/callbacks/grad_clip_by_norm.hpp"
#include "ensmallen_bits/callbacks/store_best_coordinates.hpp"
#include "get_cor_mat.h"
#include "helper.h"
#include "matern.h"
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
//' @param setting Choice of covariance structure
//' @noRd
// [[Rcpp::export]]
Rcpp::List opt_intra(const arma::vec &theta_init, const arma::mat &X_region,
                     const arma::mat &voxel_coords,
                     const arma::mat &time_sqrd_mat, int kernel_type_id,
                     int cov_setting_id, bool verbose) {
  using namespace arma;
  // Necessary evil since we can't easily expose enums to R
  KernelType kernel_type = static_cast<KernelType>(kernel_type_id);
  CovSetting cov_setting = static_cast<CovSetting>(cov_setting_id);

  // phi, tau, k, nugget
  arma::mat theta_unrestrict = softminus(theta_init);
  arma::mat dist_sqrd_mat = squared_distance(voxel_coords);

  // Construct the objective function.
  std::unique_ptr<IOptIntra> opt_intra;
  switch (cov_setting) {
  case CovSetting::noiseless:
    opt_intra = std::make_unique<OptIntraNoiseless>(X_region, dist_sqrd_mat,
                                                    time_sqrd_mat, kernel_type);
    break;
  default:
    opt_intra = std::make_unique<OptIntra>(X_region, dist_sqrd_mat,
                                           time_sqrd_mat, kernel_type);
  }

  // Create the L_BFGS optimizer with default parameters.
  ens::L_BFGS optimizer(20);
  optimizer.MaxIterations() = 100;
  // optimizer.MaxStep() = 10;

  // Run the optimization
  double optval;
  try {
    if (verbose)
      optval = optimizer.Optimize(*opt_intra, theta_unrestrict, ens::Report(1));
    else
      optval = optimizer.Optimize(*opt_intra, theta_unrestrict);
  } catch (std::runtime_error &re) {
    Rcpp::stop("Optimization failed " + std::string(re.what()));
  }
  vec theta = softplus(theta_unrestrict);

  // Compute average of spatial covariance
  double psi = accu(matern_5_2(dist_sqrd_mat, theta(0))) / dist_sqrd_mat.n_elem;

  return Rcpp::List::create(
      Rcpp::Named("theta") = theta, Rcpp::Named("psi") = psi,
      Rcpp::Named("sigma2_ep") = opt_intra->GetNoiseVarianceEstimate(),
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
//'   sigma2_ep: Estimated noise variance
//'   objective: optimal loss (negiatve log-likelihood) found.
//' @noRd
// [[Rcpp::export]]
Rcpp::List opt_inter(const arma::vec &theta_init, const arma::mat &data_r1,
                     const arma::mat &data_r2, const arma::mat &coords_r1,
                     const arma::mat &coords_r2, const arma::mat &time_sqrd_mat,
                     const Rcpp::NumericVector &stage1_r1,
                     const Rcpp::NumericVector &stage1_r2, int cov_setting_id1,
                     int cov_setting_id2, int kernel_type_id, bool verbose) {
  using arma::mat;
  using arma::vec;

  // Necessary evil since we can't easily expose enums to R
  KernelType kernel_type = static_cast<KernelType>(kernel_type_id);
  CovSetting cov_setting1 = static_cast<CovSetting>(cov_setting_id1);
  CovSetting cov_setting2 = static_cast<CovSetting>(cov_setting_id2);

  // Read in parameters inits
  mat theta_vec(theta_init);

  mat sqrd_dist_region1 = squared_distance(coords_r1);
  mat sqrd_dist_region2 = squared_distance(coords_r2);

  // These kronecker products are expensive to compute, so we do them out here
  // instead of inside the optimization class
  const mat lambda_region1 = arma::kron(
      get_cor_mat(kernel_type, sqrd_dist_region1, stage1_r1["phi_gamma"]),
      stage1_r1["k_gamma"] * get_cor_mat(KernelType::Rbf, time_sqrd_mat,
                                         stage1_r1["tau_gamma"]) +
          stage1_r1["nugget_gamma"] *
              arma::eye(data_r1.n_rows, data_r1.n_rows));

  const mat lambda_region2 = arma::kron(
      get_cor_mat(kernel_type, sqrd_dist_region2, stage1_r2["phi_gamma"]),
      stage1_r2["k_gamma"] * get_cor_mat(KernelType::Rbf, time_sqrd_mat,
                                         stage1_r2["tau_gamma"]) +
          stage1_r2["nugget_gamma"] *
              arma::eye(data_r2.n_rows, data_r2.n_rows));

  OptInter opt_inter(data_r1, data_r2, stage1_r1, stage1_r2, lambda_region1,
                     lambda_region2, cov_setting1, cov_setting2, time_sqrd_mat);

  ens::L_BFGS optimizer(10);
  optimizer.MaxIterations() = 50;
  optimizer.MaxLineSearchTrials() = 10;
  optimizer.MinGradientNorm() = 1e-4;

  ens::StoreBestCoordinates<mat> cb;
  if (verbose) {
    optimizer.Optimize(opt_inter, theta_vec, ens::GradClipByNorm(100),
                       ens::Report(1), cb);
  } else {
    optimizer.Optimize(opt_inter, theta_vec, ens::GradClipByNorm(100), cb);
  }

  vec best(cb.BestCoordinates());
  vec result(5);
  result(0) = sigmoid_inv(best(0), -1, 1); // rho
  result(1) = softplus(best(1));           // kEta1
  result(2) = softplus(best(2));           // kEta2
  result(3) = softplus(best(3));           // tauEta
  result(4) = softplus(best(4));           // nuggetEta

  std::pair<double, double> sigma2_ep_hat =
      opt_inter.GetNoiseVarianceEstimates();
  vec sigma2_ep = {sigma2_ep_hat.first, sigma2_ep_hat.second};

  return Rcpp::List::create(Rcpp::Named("theta") = result,
                            Rcpp::Named("sigma2_ep") = sigma2_ep,
                            Rcpp::Named("objval") = cb.BestObjective());
}

//' @title Get the asymptotic variance of rho from the Fisher information matrix
//' @noRd
// [[Rcpp::export]]
Rcpp::NumericMatrix
get_fisher_info(const arma::vec &theta, const arma::mat &data_r1,
                const arma::mat &data_r2, const arma::mat &coords_r1,
                const arma::mat &coords_r2, const arma::mat &time_sqrd_mat,
                const Rcpp::NumericVector &stage1_r1,
                const Rcpp::NumericVector &stage1_r2, int cov_setting_id1,
                int cov_setting_id2, int kernel_type_id) {
  using arma::mat;
  using arma::vec;

  // Necessary evil since we can't easily expose enums to R
  KernelType kernel_type = static_cast<KernelType>(kernel_type_id);
  CovSetting cov_setting1 = static_cast<CovSetting>(cov_setting_id1);
  CovSetting cov_setting2 = static_cast<CovSetting>(cov_setting_id2);

  mat theta_vec(theta);
  mat sqrd_dist_region1 = squared_distance(coords_r1);
  mat sqrd_dist_region2 = squared_distance(coords_r2);
  mat C1 = get_cor_mat(kernel_type, sqrd_dist_region1, stage1_r1["phi_gamma"]);
  mat B1 =
      stage1_r1["k_gamma"] *
          get_cor_mat(KernelType::Rbf, time_sqrd_mat, stage1_r1["tau_gamma"]) +
      stage1_r1["nugget_gamma"] * arma::eye(data_r1.n_rows, data_r1.n_rows);

  mat C2 = get_cor_mat(kernel_type, sqrd_dist_region2, stage1_r2["phi_gamma"]);
  mat B2 =
      stage1_r2["k_gamma"] *
          get_cor_mat(KernelType::Rbf, time_sqrd_mat, stage1_r2["tau_gamma"]) +
      stage1_r2["nugget_gamma"] * arma::eye(data_r2.n_rows, data_r2.n_rows);

  const mat lambda_region1 = arma::kron(C1, B1);
  const mat lambda_region2 = arma::kron(C2, B2);

  OptInter opt_inter(data_r1, data_r2, stage1_r1, stage1_r2, lambda_region1,
                     lambda_region2, cov_setting1, cov_setting2, time_sqrd_mat);

  mat stage1_params(2, 4);
  stage1_params.row(0) = {stage1_r1["phi_gamma"], stage1_r1["tau_gamma"],
                          stage1_r1["k_gamma"], stage1_r1["nugget_gamma"]};
  stage1_params.row(1) = {stage1_r2["phi_gamma"], stage1_r2["tau_gamma"],
                          stage1_r2["k_gamma"], stage1_r2["nugget_gamma"]};
  Rcpp::NumericMatrix fisher_info = opt_inter.ComputeFisherInformation(
      stage1_params, theta_vec, sqrd_dist_region1, sqrd_dist_region2, &C1, &B1,
      &C2, &B2);

  return fisher_info;
}
