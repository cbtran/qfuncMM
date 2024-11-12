#include "cov_setting.h"
#include "rbf.h"
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;

// [[Rcpp::export]]
double stage2_inter_reml(const arma::vec &theta_init, const arma::mat &region1,
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

  mat U1 = join_horiz(ones(m * l1, 1), zeros(m * l1, 1));
  mat U2 = join_horiz(zeros(m * l2, 1), ones(m * l2, 1));

  CovSetting cov_setting1 = static_cast<CovSetting>(cov_setting_id1);
  CovSetting cov_setting2 = static_cast<CovSetting>(cov_setting_id2);
  double sigma2_region1 = IsNoiseless(cov_setting1) ? 1 : r1_stage1(4);
  double sigma2_region2 = IsNoiseless(cov_setting2) ? 1 : r2_stage1(4);
  vec x_region1 = vectorise(region1) / sqrt(sigma2_region1);
  vec x_region2 = vectorise(region2) / sqrt(sigma2_region2);

  // A Matrix
  mat At = rbf(time_sqrd_mat, tauEta);
  At.diag() += nuggetEta;

  mat M_11 = lambda1;
  mat At_scaled = kEta1 * At;

  for (size_t i = 0; i < l1; ++i) {
    for (size_t j = 0; j < l1; ++j) {
      M_11.submat(i * m, j * m, (i + 1) * m - 1, (j + 1) * m - 1) += At_scaled;
    }
  }

  mat M_22 = lambda2;
  At_scaled = kEta2 * At;
  for (size_t i = 0; i < l2; ++i) {
    for (size_t j = 0; j < l2; ++j) {
      M_22.submat(i * m, j * m, (i + 1) * m - 1, (j + 1) * m - 1) += At_scaled;
    }
  }

  At_scaled = rho * sqrt(kEta1 * kEta2) * At.t();
  mat M_12t = repmat(At_scaled, l2, l1);
  if (!IsNoiseless(cov_setting1)) {
    M_11.diag() += 1;
  }
  if (!IsNoiseless(cov_setting2)) {
    M_22.diag() += 1;
  }
  mat M11R = chol(M_11);
  mat M11Ri = inv(trimatu(M11R));
  mat D = M_12t * M11Ri;
  mat S = -D * D.t();
  S += M_22;
  mat SR = chol(S);
  mat SRi = inv(trimatu(SR));

  mat &VRinv11 = M11Ri;
  mat &VRinv22 = SRi;
  mat Htilde1t = U1.t() * VRinv11;
  mat Htilde2t = -U1.t() * M11Ri * D.t() * VRinv22 + U2.t() * VRinv22;
  mat UtVinvU = Htilde1t * Htilde1t.t() + Htilde2t * Htilde2t.t();
  mat UtVinvU_R = chol(UtVinvU);
  mat Rinv = inv(trimatu(UtVinvU_R));
  mat Htilde2Rinv = Htilde2t.t() * Rinv;
  mat HHtilde1 =
      VRinv11 * Htilde1t.t() * Rinv - M11Ri * D.t() * SRi * Htilde2Rinv;
  mat HHtilde2 = VRinv22 * Htilde2Rinv;

  mat VitX1t = x_region1.t() * VRinv11;
  mat VitX2t = -x_region1.t() * M11Ri * D.t() * SRi + x_region2.t() * VRinv22;
  VitX1t *= VRinv11.t();
  VitX1t -= VitX2t * SRi.t() * D * M11Ri.t();
  VitX2t *= VRinv22.t();

  vec HX1 = VitX1t.t();
  mat HXcross = HHtilde1.t() * x_region1 + HHtilde2.t() * x_region2;
  HX1 -= HHtilde1 * HXcross;
  vec HX2 = VitX2t.t();
  HX2 -= HHtilde2 * HXcross;

  double nll1 = 2 * (sum(log(diagvec(M11R))) + sum(log(diagvec(SR))));
  double nll2 = 2 * std::real(log_det(trimatu(UtVinvU_R)));
  double nll3 = as_scalar(x_region1.t() * HX1 + x_region2.t() * HX2);
  double negLL = nll1 + nll2 + nll3;

  return -negLL;
}