#include "OptInter.h"
#include "armadillo"
#include "cov_setting.h"
#include "get_cor_mat.h"
#include "helper.h"
#include "rbf.h"
#include <R/R_ext/Arith.h>
#include <math.h>

/*****************************************************************************
 Inter-regional model
*****************************************************************************/

// Compute both objective function and its gradient
double OptInter::EvaluateWithGradient(const arma::mat &theta_unrestrict,
                                      arma::mat &gradient) {
  using namespace arma;

  // theta parameter list:
  // rho, kEta1, kEta2, tauEta, nugget
  // Transform unrestricted parameters to original forms.
  double rho = sigmoid_inv(theta_unrestrict(0), -1, 1);
  double kEta1 = softplus(theta_unrestrict(1));
  double kEta2 = softplus(theta_unrestrict(2));
  double tauEta = softplus(theta_unrestrict(3));
  double nuggetEta = softplus(theta_unrestrict(4));

  // A Matrix
  mat At = rbf(time_sqrd_, tauEta);
  At.diag() += nuggetEta;

  mat M_11 = lambda_r1_;
  M_11 += repmat(kEta1 * At, l1_, l1_);
  mat M_22 = lambda_r2_;
  M_22 += repmat(kEta2 * At, l2_, l2_);
  mat M_12 = repmat(rho * sqrt(kEta1 * kEta2) * At, l1_, l2_);
  if (!IsNoiseless(cov_setting_r1_)) {
    M_11.diag() += 1;
  }
  if (!IsNoiseless(cov_setting_r2_)) {
    M_22.diag() += 1;
  }

  mat V = join_vert(join_horiz(M_11, M_12), join_horiz(M_12.t(), M_22));

  // These are the most computationally expensive steps
  // Try Cholesky decomposition with adaptive regularization if needed
  mat VR;
  bool success = chol(VR, V);
  if (!success) {
    // Matrix is not positive definite, add a small regularization
    double reg =
        1e-10 * trace(V) / V.n_rows; // Small fraction of average diagonal
    mat V_reg = V;
    V_reg.diag() += reg;

    // Try increasing regularization until success
    int max_attempts = 10;
    int attempt = 0;
    while (!success && attempt < max_attempts) {
      V_reg.diag() += reg;
      success = chol(VR, V_reg);
      reg *= 10.0; // Increase regularization for next attempt
      attempt++;
    }

    if (!success) {
      // If still fails, use a more robust fallback
      Rcpp::Rcout << "Warning: Cholesky decomposition failed, using larger "
                     "regularization.\n";
      V_reg = V;
      V_reg.diag() += 1e-6 * trace(V) / V.n_rows;
      success = chol(VR, V_reg);
    }
  }
  mat VRinv = inv(trimatu(VR));

  mat Htilde = VRinv.t() * design_;
  mat UtVinvU = Htilde.t() * Htilde;
  mat UtVinvU_R = chol(UtVinvU);
  mat Rinv = inv(trimatu(UtVinvU_R));
  Htilde = VRinv * Htilde * Rinv;

  // l1 is logdet(Vjj')
  double nll1 = 2 * sum(log(diagvec(VR)));
  // l2 is logdet(UtVinvjj'U)
  double nll2 = 2 * std::real(log_det(trimatu(UtVinvU_R)));

  vec HX = VRinv * VRinv.t() * data_ - Htilde * Htilde.t() * data_;

  double nll3 = as_scalar(data_.t() * HX);
  double negLL = nll1 + nll2 + nll3;

  // Compute gradients
  mat zeroL11 = zeros(l1_, l1_);
  mat zeroL22 = zeros(l2_, l2_);
  mat dAt_dtau_eta = rbf_deriv(time_sqrd_, tauEta);
  mat dAt_dnugget = eye(m_, m_);
  double rho_deriv = sigmoid_inv_derivative(rho, -1, 1);

  mat VRi_head = VRinv.head_rows(m_ * l1_);
  mat VRi_tail = VRinv.tail_rows(m_ * l2_);
  mat rho_block1 = repmat(At, 1, l2_) * VRi_tail;
  mat rho_bigblock1 = repmat(rho_block1, l1_, 1);
  // There's a potential optimizaiton in this step by considering the row-block
  // form of rho_bigblock1.

  // Gradient for rho
  gradient(0) = 2 * sqrt(kEta1 * kEta2) * trace(VRi_head.t() * rho_bigblock1);
  mat dVdrho_kron1(l1_, l2_, fill::value(sqrt(kEta1 * kEta2)));
  dVdrho_kron1 = join_vert(join_horiz(zeroL11, dVdrho_kron1),
                           join_horiz(dVdrho_kron1.t(), zeroL22));
  gradient(0) -= trace(Htilde.t() * kronecker_mmm(dVdrho_kron1, At, Htilde));
  gradient(0) -= as_scalar(HX.t() * kronecker_mvm(dVdrho_kron1, At, HX));
  gradient(0) *= rho_deriv;

  // Gradient for kEta1
  mat oneL11 = ones(l1_, l1_);
  mat oneL22 = ones(l2_, l2_);
  mat oneL12 = ones(l1_, l2_);
  mat dVdketa_kronblock(l1_, l2_, fill::value(sqrt(kEta2 / kEta1) * rho * 0.5));
  mat dVdketa1_kron = join_vert(join_horiz(oneL11, dVdketa_kronblock),
                                join_horiz(dVdketa_kronblock.t(), zeroL22));
  gradient(1) = trace(VRi_head.t() * kronecker_mmm(oneL11, At, VRi_head));
  double keta_trace = trace(VRi_head.t() * rho_bigblock1);
  gradient(1) += sqrt(kEta2 / kEta1) * rho * keta_trace;
  gradient(1) -= trace(Htilde.t() * kronecker_mmm(dVdketa1_kron, At, Htilde)) +
                 as_scalar(HX.t() * kronecker_mvm(dVdketa1_kron, At, HX));
  gradient(1) *= logistic(kEta1);

  // Gradient for kEta2
  dVdketa_kronblock.fill(sqrt(kEta1 / kEta2) * rho * 0.5);
  mat dVdketa2_kron = join_vert(join_horiz(zeroL11, dVdketa_kronblock),
                                join_horiz(dVdketa_kronblock.t(), oneL22));
  gradient(2) = trace(VRi_tail.t() * kronecker_mmm(oneL22, At, VRi_tail));
  gradient(2) += sqrt(kEta1 / kEta2) * rho * keta_trace;
  gradient(2) -= trace(Htilde.t() * kronecker_mmm(dVdketa2_kron, At, Htilde)) +
                 as_scalar(HX.t() * kronecker_mvm(dVdketa2_kron, At, HX));
  gradient(2) *= logistic(kEta2);

  // Gradient for tauEta
  gradient(3) = kEta1 * trace(VRi_head.t() *
                              kronecker_mmm(oneL11, dAt_dtau_eta, VRi_head));
  gradient(3) += kEta2 * trace(VRi_tail.t() *
                               kronecker_mmm(oneL22, dAt_dtau_eta, VRi_tail));

  rho_block1 = repmat(dAt_dtau_eta, 1, l2_) * VRi_tail;
  rho_bigblock1 = repmat(rho_block1, l1_, 1);
  gradient(3) +=
      2 * sqrt(kEta1 * kEta2) * rho * trace(VRi_head.t() * rho_bigblock1);
  mat dVdtauEta_kron = join_vert(
      join_horiz(kEta1 * oneL11, sqrt(kEta1 * kEta2) * rho * oneL12),
      join_horiz(sqrt(kEta1 * kEta2) * rho * oneL12.t(), kEta2 * oneL22));
  gradient(3) -=
      trace(Htilde.t() * kronecker_mmm(dVdtauEta_kron, dAt_dtau_eta, Htilde)) +
      as_scalar(HX.t() * kronecker_mvm(dVdtauEta_kron, dAt_dtau_eta, HX));
  gradient(3) *= logistic(tauEta);

  // Gradient for nugget
  gradient(4) = kEta1 * trace(VRi_head.t() * kronecker_mmm(oneL11, dAt_dnugget,
                                                           VRi_head, true));
  gradient(4) += kEta2 * trace(VRi_tail.t() * kronecker_mmm(oneL22, dAt_dnugget,
                                                            VRi_tail, true));

  rho_bigblock1 = repmat(repmat(dAt_dnugget, 1, l2_) * VRi_tail, l1_, 1);
  gradient(4) +=
      2 * sqrt(kEta1 * kEta2) * rho * trace(VRi_head.t() * rho_bigblock1);
  gradient(4) -=
      trace(Htilde.t() *
            kronecker_mmm(dVdtauEta_kron, dAt_dnugget, Htilde, true)) +
      as_scalar(HX.t() * kronecker_mvm(dVdtauEta_kron, dAt_dnugget, HX, true));
  gradient(4) *= logistic(nuggetEta);

  return negLL;
}

double OptInter::Evaluate(const arma::mat &theta_unrestrict) {
  using arma::join_horiz;
  using arma::join_vert;
  using arma::mat;
  using arma::vec;

  // theta parameter list:
  // rho, kEta1, kEta2, tauEta, nugget
  // Transform unrestricted parameters to original forms.
  double rho = sigmoid_inv(theta_unrestrict(0), -1, 1);
  double kEta1 = softplus(theta_unrestrict(1));
  double kEta2 = softplus(theta_unrestrict(2));
  double tauEta = softplus(theta_unrestrict(3));
  double nuggetEta = softplus(theta_unrestrict(4));

  // A Matrix
  mat At = rbf(time_sqrd_, tauEta);
  At.diag() += nuggetEta;

  mat M_12 = arma::repmat(rho * sqrt(kEta1) * sqrt(kEta2) * At, l1_, l2_);

  mat M_11 = lambda_r1_ + kEta1 * arma::repmat(At, l1_, l1_) +
             arma::eye(l1_ * m_, l1_ * m_);
  mat M_22 = lambda_r2_ + kEta2 * arma::repmat(At, l2_, l2_) +
             arma::eye(l2_ * m_, l2_ * m_);

  mat V = join_vert(join_horiz(M_11, M_12), join_horiz(M_12.t(), M_22));

  mat VR = arma::chol(V);
  mat VRinv = arma::inv(arma::trimatu(VR));

  mat Vinv = VRinv * VRinv.t();
  mat VinvU = arma::solve(arma::trimatu(VR), VRinv.t() * design_);
  mat UtVinvU = design_.t() * VinvU;
  mat UtVinvU_R = arma::chol(UtVinvU);
  mat Rinv = arma::inv(arma::trimatu(UtVinvU_R));
  mat H = Vinv - VinvU * Rinv * Rinv.t() * VinvU.t();

  // l1 is logdet(Vjj')
  double l1 = 2 * arma::sum(arma::log(arma::diagvec(VR)));
  // l2 is logdet(UtVinvjj'U)
  double l2 = 2 * std::real(arma::log_det(arma::trimatu(UtVinvU_R)));

  vec HX = H * data_;

  double l3 = arma::as_scalar(data_.t() * HX);
  double negLL = l1 + l2 + l3;

  return negLL;
}

// Compute Fisher Information Matrix for all variance components
Rcpp::NumericMatrix OptInter::ComputeFisherInformation(
    const arma::mat &theta_stage1, const arma::mat &theta_stage2,
    const arma::mat &dist_sqrd1, const arma::mat &dist_sqrd2, arma::mat *C1,
    arma::mat *B1, arma::mat *C2, arma::mat *B2) {
  using namespace arma;

  // stage1 parameter list:
  // phi, tau, k, nugget
  double phi_gamma1 = theta_stage1(0, 0);
  double tau_gamma1 = theta_stage1(0, 1);
  double k_gamma1 = theta_stage1(0, 2);
  double nugget_gamma1 = theta_stage1(0, 3);
  double phi_gamma2 = theta_stage1(1, 0);
  double tau_gamma2 = theta_stage1(1, 1);
  double k_gamma2 = theta_stage1(1, 2);
  double nugget_gamma2 = theta_stage1(1, 3);

  // stage2 parameter list:
  // rho, kEta1, kEta2, tauEta, nugget
  double rho = theta_stage2(0);
  double kEta1 = theta_stage2(1);
  double kEta2 = theta_stage2(2);
  double tauEta = theta_stage2(3);
  double nuggetEta = theta_stage2(4);

  // A Matrix
  mat At = rbf(time_sqrd_, tauEta);
  At.diag() += nuggetEta;

  mat M_11 = lambda_r1_;
  M_11 += repmat(kEta1 * At, l1_, l1_);
  mat M_22 = lambda_r2_;
  M_22 += repmat(kEta2 * At, l2_, l2_);
  mat M_12 = repmat(rho * sqrt(kEta1 * kEta2) * At, l1_, l2_);
  if (!IsNoiseless(cov_setting_r1_)) {
    M_11.diag() += 1;
  }
  if (!IsNoiseless(cov_setting_r2_)) {
    M_22.diag() += 1;
  }

  mat V = join_vert(join_horiz(M_11, M_12), join_horiz(M_12.t(), M_22));

  // Cholesky decomposition with regularization if needed
  mat VR;
  bool success = chol(VR, V);
  if (!success) {
    double reg = 1e-10 * trace(V) / V.n_rows;
    mat V_reg = V;
    V_reg.diag() += reg;

    int max_attempts = 10;
    int attempt = 0;
    while (!success && attempt < max_attempts) {
      V_reg.diag() += reg;
      success = chol(VR, V_reg);
      reg *= 10.0;
      attempt++;
    }

    if (!success) {
      V_reg = V;
      V_reg.diag() += 1e-6 * trace(V) / V.n_rows;
      success = chol(VR, V_reg);
    }
  }
  mat VRinv = inv(trimatu(VR));
  mat Vinv = VRinv * VRinv.t();
  // mat VinvU = arma::solve(arma::trimatu(VR), VRinv.t() * design_);
  // mat fixed_fx_block = design_.t() * VinvU;

  // Prepare derivative matrices (spatial structure only)
  mat zeroL11 = zeros(l1_, l1_);
  mat zeroL22 = zeros(l2_, l2_);
  mat zeroL12 = zeros(l1_, l2_);
  mat dAt_dtau_eta = rbf_deriv(time_sqrd_, tauEta);
  mat dAt_dnugget = eye(m_, m_);

  // Derivatives wrt stage 1 parameters
  mat dB_dk_gamma1 = rbf(time_sqrd_, tau_gamma1);
  mat dB_dk_gamma2 = rbf(time_sqrd_, tau_gamma2);
  mat dB_dtau_gamma1 = k_gamma1 * rbf_deriv(time_sqrd_, tau_gamma1);
  mat dB_dtau_gamma2 = k_gamma2 * rbf_deriv(time_sqrd_, tau_gamma2);
  mat dC_dphi_gamma1 =
      get_cor_mat_deriv(KernelType::Matern52, dist_sqrd1, phi_gamma1);
  mat dC_dphi_gamma2 =
      get_cor_mat_deriv(KernelType::Matern52, dist_sqrd2, phi_gamma2);

  // Stage 1 block diag derivative matrices
  std::vector<mat *> s1_Kmats_r1(4);
  std::vector<mat *> s1_Amats_r1(4);

  std::vector<mat *> s1_Kmats_r2(4);
  std::vector<mat *> s1_Amats_r2(4);

  // 5 for stage 2, 4 for each stage 1 region
  int num_param = 13;

  int r1_sigma2_ep_colid = -1, r2_sigma2_ep_colid = -1;
  double sigma2_ep_r1 = 1, sigma2_ep_r2 = 1;
  if (!IsNoiseless(cov_setting_r1_)) {
    sigma2_ep_r1 = sigma2_ep_.first;
    r1_sigma2_ep_colid = num_param;
    num_param++;
  }
  if (!IsNoiseless(cov_setting_r2_)) {
    sigma2_ep_r2 = sigma2_ep_.second;
    r2_sigma2_ep_colid = num_param;
    num_param++;
  }
  // mat oneL11 = ones(l1_, l1_);
  mat oneL11(l1_, l1_, fill::value(sigma2_ep_r1));
  mat oneL22(l2_, l2_, fill::value(sigma2_ep_r2));
  mat oneL12(l1_, l2_, fill::value(sqrt(sigma2_ep_r1 * sigma2_ep_r2)));

  // Scale spatial derivatives by noise variance
  dC_dphi_gamma1 *= sigma2_ep_r1;
  dC_dphi_gamma2 *= sigma2_ep_r2;
  *C1 *= sigma2_ep_r1;
  *C2 *= sigma2_ep_r2;

  // c("phi_gamma", "tau_gamma", "k_gamma", "nugget_gamma")
  s1_Kmats_r1[0] = &dC_dphi_gamma1;
  s1_Amats_r1[0] = B1;
  s1_Kmats_r1[1] = C1;
  s1_Amats_r1[1] = &dB_dtau_gamma1;
  s1_Kmats_r1[2] = C1;
  s1_Amats_r1[2] = &dB_dk_gamma1;
  s1_Kmats_r1[3] = C1;
  s1_Amats_r1[3] = &dAt_dnugget;

  s1_Kmats_r2[0] = &dC_dphi_gamma2;
  s1_Amats_r2[0] = B2;
  s1_Kmats_r2[1] = C2;
  s1_Amats_r2[1] = &dB_dtau_gamma2;
  s1_Kmats_r2[2] = C2;
  s1_Amats_r2[2] = &dB_dk_gamma2;
  s1_Kmats_r2[3] = C2;
  s1_Amats_r2[3] = &dAt_dnugget;

  // Store the 5 spatial derivative structures
  std::vector<mat> s2_Kmats(5);
  std::vector<mat *> s2_Amats(5);
  // 1. dV/drho = K_rho ⊗ At
  s2_Kmats[0] =
      join_vert(join_horiz(zeroL11, oneL12 * sqrt(kEta1 * kEta2)),
                join_horiz(oneL12.t() * sqrt(kEta1 * kEta2), zeroL22));
  s2_Amats[0] = &At;

  // 2. dV/dkEta1 = K_kEta1 ⊗ At
  mat keta1_block = oneL12 * (sqrt(kEta2 / kEta1) * rho * 0.5);
  s2_Kmats[1] = join_vert(join_horiz(oneL11, keta1_block),
                          join_horiz(keta1_block.t(), zeroL22));
  s2_Amats[1] = &At;

  // 3. dV/dkEta2 = K_kEta2 ⊗ At
  mat keta2_block = oneL12 * (sqrt(kEta1 / kEta2) * rho * 0.5);
  s2_Kmats[2] = join_vert(join_horiz(zeroL11, keta2_block),
                          join_horiz(keta2_block.t(), oneL22));
  s2_Amats[2] = &At;

  // 4. dV/dtauEta = K_tauEta ⊗ dAt_dtau_eta
  s2_Kmats[3] = join_vert(
      join_horiz(kEta1 * oneL11, sqrt(kEta1 * kEta2) * rho * oneL12),
      join_horiz(sqrt(kEta1 * kEta2) * rho * oneL12.t(), kEta2 * oneL22));
  s2_Amats[3] = &dAt_dtau_eta;

  // 5. dV/dnuggetEta = K_nugget ⊗ dAt_dnugget
  s2_Kmats[4] = s2_Kmats[3]; // Same spatial structure as tauEta
  s2_Amats[4] = &dAt_dnugget;

  // Fisher Information Matrix elements are given by.
  // I_ij = trace(V^-1 * dV_i * V^-1 * dV_j) / 2
  // This stores components of the form V^-1 * dV_i
  std::vector<mat> Vinv_components(num_param);

  if (!IsNoiseless(cov_setting_r1_)) {
    mat upper_right = M_12 * sqrt(sigma2_ep_r2 / sigma2_ep_r1) / 2;
    mat dVdsigma2ep1 =
        join_vert(join_horiz(M_11, upper_right),
                  join_horiz(upper_right.t(), zeros(m_ * l2_, m_ * l2_)));
    Vinv_components[r1_sigma2_ep_colid] = dVdsigma2ep1 * Vinv;
  }
  if (!IsNoiseless(cov_setting_r2_)) {
    mat upper_right = M_12 * sqrt(sigma2_ep_r1 / sigma2_ep_r2) / 2;
    mat dVdsigma2ep2 =
        join_vert(join_horiz(zeros(m_ * l1_, m_ * l1_), upper_right),
                  join_horiz(upper_right.t(), M_22));
    Vinv_components[r2_sigma2_ep_colid] = dVdsigma2ep2 * Vinv;
  }

  Vinv.cols(0, l1_ * m_) /= sqrt(sigma2_ep_r1);
  Vinv.cols(l1_ * m_, Vinv.n_cols - 1) /= sqrt(sigma2_ep_r2);
  Vinv.rows(0, l1_ * m_ - 1) /= sqrt(sigma2_ep_r1);
  Vinv.rows(l1_ * m_, Vinv.n_rows - 1) /= sqrt(sigma2_ep_r2);

  mat fisher_info_mx = zeros(num_param, num_param);

  const mat &Vinv11 = Vinv.submat(0, 0, l1_ * m_ - 1, l1_ * m_ - 1);
  const mat &Vinv22 =
      Vinv.submat(l1_ * m_, l1_ * m_, V.n_rows - 1, V.n_cols - 1);
  for (int i = 0; i < 4; i++) {
    // Compute Vinv * (K_i ⊗ A_i)^T for stage 1 region 1
    mat upper_left = kronecker_mmm(*s1_Kmats_r1[i], *s1_Amats_r1[i], Vinv11);
    Vinv_components[i] = upper_left;

    mat lower_right = kronecker_mmm(*s1_Kmats_r2[i], *s1_Amats_r2[i], Vinv22);
    Vinv_components[i + 4] = lower_right;
  }

  for (int i = 0; i < 5; i++) {
    // Compute (K_i ⊗ A_i) * Vinv
    Vinv_components[i + 8] = kronecker_mmm(s2_Kmats[i], *s2_Amats[i], Vinv);
  }

  for (int i = 0; i < num_param; i++) {
    for (int j = i; j < num_param; j++) {
      if (i < 4 && 4 <= j && j < 8) {
        // Cross-terms between stage 1 params of different regions are zero
        fisher_info_mx(i, j) = 0;
        fisher_info_mx(j, i) = 0;
        continue;
      }
      mat right;
      if (i < 4 && j >= 8) {
        // i < 8 are the stage 1 parameters, so only the upper-left or
        // lower-right blocks are nonzero. Extract the relevant submatrices for
        // stage 2 parameters based on this.
        right = Vinv_components[j].submat(0, 0, l1_ * m_ - 1, l1_ * m_ - 1);
      } else if (i < 8 && j >= 8) {
        right = Vinv_components[j].submat(l1_ * m_, l1_ * m_, Vinv.n_rows - 1,
                                          Vinv.n_cols - 1);
      } else {
        right = Vinv_components[j];
      }
      fisher_info_mx(i, j) = trace(Vinv_components[i] * right) / 2.0;
      if (i != j) {
        fisher_info_mx(j, i) = fisher_info_mx(i, j);
      }
    }
  }

  Rcpp::CharacterVector param_names(
      {"phi_gamma1", "tau_gamma1", "k_gamma1", "nugget_gamma1", "phi_gamma2",
       "tau_gamma2", "k_gamma2", "nugget_gamma2", "rho", "k_eta1", "k_eta2",
       "tau_eta", "nugget_eta"});
  if (r1_sigma2_ep_colid >= 0) {
    param_names.push_back("sigma2_ep1");
  }
  if (r2_sigma2_ep_colid >= 0) {
    param_names.push_back("sigma2_ep2");
  }
  Rcpp::NumericMatrix r_matrix = Rcpp::wrap(fisher_info_mx);
  r_matrix.attr("dimnames") = Rcpp::List::create(param_names, param_names);

  return r_matrix;
}