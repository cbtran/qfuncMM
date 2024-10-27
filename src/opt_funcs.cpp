#include <RcppEnsmallen.h>
#include <stdexcept>

#include "OptInter.h"
#include "OptIntra.h"
#include "Rcpp/exceptions.h"
#include "Rcpp/iostream/Rstreambuf.h"
#include "Rcpp/vector/instantiation.h"
#include "armadillo"
#include "cov_setting.h"
#include "ensmallen_bits/callbacks/grad_clip_by_norm.hpp"
#include "ensmallen_bits/callbacks/store_best_coordinates.hpp"
#include "get_cor_mat.h"
#include "helper.h"
#include "matern.h"
#include "rbf.h"
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
  // or phi, tau, nugget_over_k if profiled
  arma::mat theta_unrestrict = softminus(theta_init);
  arma::mat dist_sqrd_mat = squared_distance(voxel_coords);

  // Construct the objective function.
  std::unique_ptr<IOptIntra> opt_intra;
  switch (cov_setting) {
  case CovSetting::noiseless_profiled:
    opt_intra = std::make_unique<OptIntraNoiselessProfiled>(
        X_region, dist_sqrd_mat, time_sqrd_mat, kernel_type);
    break;
  case CovSetting::noiseless:
    opt_intra = std::make_unique<OptIntraNoiseless>(X_region, dist_sqrd_mat,
                                                    time_sqrd_mat, kernel_type);
    break;
  case CovSetting::diag_time:
    opt_intra = std::make_unique<OptIntraDiagTime>(X_region, dist_sqrd_mat,
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
  if (cov_setting == CovSetting::noiseless_profiled) {
    vec theta_orig = arma::zeros(4);
    theta_orig(0) = theta(0);
    theta_orig(1) = theta(1);
    theta_orig(2) = opt_intra->GetKStar();
    theta_orig(3) = theta(2) * theta_orig(2);
    theta = theta_orig;
  }

  // Compute average of spatial covariance
  double psi = accu(matern_5_2(dist_sqrd_mat, theta(0))) / dist_sqrd_mat.n_elem;

  return Rcpp::List::create(
      Rcpp::Named("theta") = theta, Rcpp::Named("psi") = psi,
      Rcpp::Named("var_noise") = opt_intra->GetNoiseVarianceEstimate(),
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
                     const arma::vec &stage1ParamsRegion2, int cov_setting_id1,
                     int cov_setting_id2, int kernel_type_id, bool verbose) {
  using arma::mat;
  using arma::vec;

  // Necessary evil since we can't easily expose enums to R
  KernelType kernel_type = static_cast<KernelType>(kernel_type_id);
  CovSetting cov_setting1 = static_cast<CovSetting>(cov_setting_id1);
  CovSetting cov_setting2 = static_cast<CovSetting>(cov_setting_id2);

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

  bool any_diag_time = cov_setting1 == CovSetting::diag_time ||
                       cov_setting2 == CovSetting::diag_time;
  // Construct the objective function.
  std::unique_ptr<IOptInter> opt_inter;
  if (any_diag_time) {
    opt_inter = std::make_unique<OptInterDiagTime>(
        dataRegion1, dataRegion2, stage1ParamsRegion1, stage1ParamsRegion2,
        block_region_1, block_region_2, cov_setting1, cov_setting2,
        time_sqrd_mat);
  } else {
    opt_inter = std::make_unique<OptInter>(
        dataRegion1, dataRegion2, stage1ParamsRegion1, stage1ParamsRegion2,
        block_region_1, block_region_2, cov_setting1, cov_setting2,
        time_sqrd_mat);
  }

  ens::L_BFGS optimizer(10);
  optimizer.MaxIterations() = 50;
  optimizer.MaxLineSearchTrials() = 10;
  optimizer.MinGradientNorm() = 1e-4;

  ens::StoreBestCoordinates<mat> cb;
  if (verbose) {
    optimizer.Optimize(*opt_inter, theta_vec, ens::GradClipByNorm(100),
                       ens::Report(1), cb);
  } else {
    optimizer.Optimize(*opt_inter, theta_vec, ens::GradClipByNorm(100), cb);
  }

  vec best(cb.BestCoordinates());
  vec result(5);
  result(0) = sigmoid_inv(best(0), -1, 1); // rho
  result(1) = softplus(best(1));           // kEta1
  result(2) = softplus(best(2));           // kEta2

  if (any_diag_time) {
    result(3) = 0; // tauEta
    result(4) = 1; // nuggetEta
  } else {
    result(3) = softplus(best(3)); // tauEta
    result(4) = softplus(best(4)); // nuggetEta
  }

  std::pair<double, double> sigma2_hat = opt_inter->GetNoiseVarianceEstimates();
  vec var_noise = {sigma2_hat.first, sigma2_hat.second};

  return Rcpp::List::create(Rcpp::Named("theta") = result,
                            Rcpp::Named("var_noise") = var_noise,
                            Rcpp::Named("objective") = cb.BestObjective());
}

// [[Rcpp::export]]
double stage2_inter_nll(const arma::vec &theta_init, const arma::mat &region1,
                        const arma::mat &region2, const arma::mat &r1_coords,
                        const arma::mat &r2_coords,
                        const arma::mat &time_sqrd_mat,
                        const arma::vec &r1_stage1, const arma::vec &r2_stage1,
                        const arma::mat &lambda1, const arma::mat &lambda2,
                        int cov_setting_id1, int cov_setting_id2,
                        int kernel_type_id) {
  using namespace arma;
  // theta parameter list:
  // rho, kEta1, kEta2, tauEta, nugget
  double rho = theta_init(0);
  double kEta1 = theta_init(1);
  double kEta2 = theta_init(2);
  double tauEta = theta_init(3);
  double nuggetEta = theta_init(4);

  int m = region1.n_rows;
  int l1 = region1.n_cols;
  int l2 = region2.n_cols;

  CovSetting cov_setting1 = static_cast<CovSetting>(cov_setting_id1);
  CovSetting cov_setting2 = static_cast<CovSetting>(cov_setting_id2);
  double sigma2_region1 = IsNoiseless(cov_setting1) ? 1 : r1_stage1(4);
  double sigma2_region2 = IsNoiseless(cov_setting2) ? 1 : r2_stage1(4);
  vec x_scaled = join_vert(vectorise(region1) / sqrt(sigma2_region1),
                           vectorise(region2) / sqrt(sigma2_region2));

  // A Matrix
  mat At = rbf(time_sqrd_mat, tauEta);
  At.diag() += nuggetEta;

  mat M_11 = lambda1;
  M_11 += repmat(kEta1 * At, l1, l1);
  mat M_22 = lambda2;
  M_22 += repmat(kEta2 * At, l2, l2);
  mat M_12 = repmat(rho * sqrt(kEta1 * kEta2) * At, l1, l2);
  if (!IsNoiseless(cov_setting1)) {
    M_11.diag() += 1;
  }
  if (!IsNoiseless(cov_setting2)) {
    M_22.diag() += 1;
  }

  mat V = join_vert(join_horiz(M_11, M_12), join_horiz(M_12.t(), M_22));
  mat U = join_vert(join_horiz(ones(m * l1, 1), zeros(m * l1, 1)),
                    join_horiz(zeros(m * l2, 1), ones(m * l2, 1)));

  mat VR = arma::chol(V);
  mat VRinv = arma::inv(arma::trimatu(VR));
  mat Htilde = VRinv.t() * U;
  mat UtVinvU = Htilde.t() * Htilde;
  mat UtVinvU_R = chol(UtVinvU);
  mat Rinv = inv(trimatu(UtVinvU_R));
  Htilde = VRinv * Htilde * Rinv;

  double nll1 = 2 * arma::sum(arma::log(arma::diagvec(VR)));
  double nll2 = 2 * std::real(arma::log_det(arma::trimatu(UtVinvU_R)));

  vec HX = VRinv * VRinv.t() * x_scaled - Htilde * Htilde.t() * x_scaled;

  double nll3 = as_scalar(x_scaled.t() * HX);
  double negLL = nll1 + nll2 + nll3;

  return negLL;
}