#include "OptInter.h"
#include "cov_setting.h"
#include "helper.h"
#include "rbf.h"
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
Rcpp::NumericMatrix
OptInter::ComputeFisherInformation(const arma::mat &theta_stage1,
                                   const arma::mat &theta_stage2) {
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
  mat oneL11 = ones(l1_, l1_);
  mat oneL22 = ones(l2_, l2_);
  mat oneL12 = ones(l1_, l2_);
  mat dAt_dtau_eta = rbf_deriv(time_sqrd_, tauEta);
  mat dAt_dnugget = eye(m_, m_);

  // Store the 5 spatial derivative structures
  std::vector<mat> K_matrices(5);
  std::vector<mat *> A_matrices(5);

  // 1. dV/drho = K_rho ⊗ At
  K_matrices[0] =
      join_vert(join_horiz(zeroL11, oneL12 * sqrt(kEta1 * kEta2)),
                join_horiz(oneL12.t() * sqrt(kEta1 * kEta2), zeroL22));
  A_matrices[0] = &At;

  // 2. dV/dkEta1 = K_kEta1 ⊗ At
  mat keta1_block = oneL12 * (sqrt(kEta2 / kEta1) * rho * 0.5);
  K_matrices[1] = join_vert(join_horiz(oneL11, keta1_block),
                            join_horiz(keta1_block.t(), zeroL22));
  A_matrices[1] = &At;

  // 3. dV/dkEta2 = K_kEta2 ⊗ At
  mat keta2_block = oneL12 * (sqrt(kEta1 / kEta2) * rho * 0.5);
  K_matrices[2] = join_vert(join_horiz(zeroL11, keta2_block),
                            join_horiz(keta2_block.t(), oneL22));
  A_matrices[2] = &At;

  // 4. dV/dtauEta = K_tauEta ⊗ dAt_dtau_eta
  K_matrices[3] = join_vert(
      join_horiz(kEta1 * oneL11, sqrt(kEta1 * kEta2) * rho * oneL12),
      join_horiz(sqrt(kEta1 * kEta2) * rho * oneL12.t(), kEta2 * oneL22));
  A_matrices[3] = &dAt_dtau_eta;

  // 5. dV/dnuggetEta = K_nugget ⊗ dAt_dnugget
  K_matrices[4] = K_matrices[3]; // Same spatial structure as tauEta
  A_matrices[4] = &dAt_dnugget;

  mat fisher_info_mx = zeros(5, 5);

  // Compute Fisher Information Matrix elements:
  // I_ij = trace(V^-1 * dV_i * V^-1 * dV_j) / 2
  std::vector<mat> Vinv_matrices(5);
  for (int i = 0; i < 5; i++) {
    // Compute Vinv * (K_i ⊗ A_i)^T
    Vinv_matrices[i] = kronecker_mmm(K_matrices[i], *A_matrices[i], Vinv);
  }

  for (int i = 0; i < 5; i++) {
    for (int j = i; j < 5; j++) {
      fisher_info_mx(i, j) = trace(Vinv_matrices[i] * Vinv_matrices[j]) / 2.0;
      if (i != j) {
        fisher_info_mx(j, i) = fisher_info_mx(i, j);
      }
    }
  }

  Rcpp::CharacterVector param_names(
      {"rho", "k_eta1", "k_eta2", "tau_eta", "nugget_eta"});
  Rcpp::NumericMatrix r_matrix = Rcpp::wrap(fisher_info_mx);
  r_matrix.attr("dimnames") = Rcpp::List::create(param_names, param_names);

  return r_matrix;
}