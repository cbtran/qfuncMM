#include "OptInter.h"
#include "cov_setting.h"
#include "helper.h"
#include "rbf.h"
#include <math.h>
//[[Rcpp:depends(RcppClock)]]
#include <RcppClock.h>

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
  // Rcpp::Rcout << "Params: " << rho << " " << kEta1 << " " << kEta2 << " " <<
  // tauEta << " " << nuggetEta << std::endl;

  int M_L1 = numTimePt_ * numVoxelRegion1_;
  int M_L2 = numTimePt_ * numVoxelRegion2_;
  Rcpp::Clock clock;
  clock.tick("reml");
  clock.tick("reml setup");

  // A Matrix
  mat At = rbf(timeSqrd_, tauEta);
  At.diag() += nuggetEta;

  mat M_11, M_22, M_12;

  clock.tick("V blocks");
  if (!IsNoiseless(cov_setting_region1_) &&
      !IsNoiseless(cov_setting_region2_)) {
    M_12 = repmat(rho * sqrt(kEta1) * sqrt(kEta2) * At, numVoxelRegion1_,
                  numVoxelRegion2_);
    M_11 = spaceTimeKernelRegion1_ +
           kEta1 * repmat(At, numVoxelRegion1_, numVoxelRegion1_) +
           eye(numVoxelRegion1_ * numTimePt_, numVoxelRegion1_ * numTimePt_);
    M_22 = spaceTimeKernelRegion2_ +
           kEta2 * repmat(At, numVoxelRegion2_, numVoxelRegion2_) +
           eye(numVoxelRegion2_ * numTimePt_, numVoxelRegion2_ * numTimePt_);
  } else if (IsNoiseless(cov_setting_region1_) &&
             IsNoiseless(cov_setting_region2_)) {
    M_12 = repmat(rho * sqrt(kEta1) * sqrt(kEta2) * At, numVoxelRegion1_,
                  numVoxelRegion2_);
    M_11 = spaceTimeKernelRegion1_ +
           repmat(kEta1 * At, numVoxelRegion1_, numVoxelRegion1_);
    M_22 = spaceTimeKernelRegion2_ +
           repmat(kEta2 * At, numVoxelRegion2_, numVoxelRegion2_);
  } else if (IsNoiseless(cov_setting_region1_)) {
    // Region 1 is noiseless and region 2 is noisy
    double sigma_region2 = sqrt(sigma2_.second);
    M_12 = repmat(rho * sqrt(kEta1) * sqrt(kEta2) * At * sigma_region2,
                  numVoxelRegion1_, numVoxelRegion2_);
    M_11 = spaceTimeKernelRegion1_ +
           kEta1 * repmat(At, numVoxelRegion1_, numVoxelRegion1_);
    M_22 = sigma_region2 * spaceTimeKernelRegion2_ +
           sigma2_.second * kEta2 *
               repmat(At, numVoxelRegion2_, numVoxelRegion2_) +
           sigma2_.second * eye(numVoxelRegion2_ * numTimePt_,
                                numVoxelRegion2_ * numTimePt_);
  } else {
    // Region 1 is noisy and region 2 is noiseless
    double sigma_region1 = sqrt(sigma2_.first);
    M_12 = repmat(rho * sqrt(kEta1) * sqrt(kEta2) * At * sigma_region1,
                  numVoxelRegion1_, numVoxelRegion2_);
    M_11 =
        sigma_region1 * spaceTimeKernelRegion1_ +
        sigma2_.first * kEta1 * repmat(At, numVoxelRegion1_, numVoxelRegion1_) +
        sigma2_.first *
            eye(numVoxelRegion1_ * numTimePt_, numVoxelRegion1_ * numTimePt_);
    M_22 = spaceTimeKernelRegion2_ +
           kEta2 * repmat(At, numVoxelRegion2_, numVoxelRegion2_);
  }
  clock.tock("V blocks");

  // clock.tick("V join");
  // mat V = join_vert(join_horiz(M_11, M_12), join_horiz(M_12.t(), M_22));
  // clock.tock("V join");
  clock.tock("reml setup");

  // These are the most computationally expensive steps
  // clock.tick("V chol");
  // mat VR = chol(V);
  // clock.tock("V chol");
  // clock.tick("V chol inv");
  // mat VRinv = inv(trimatu(VR));
  // clock.tock("V chol inv");

  // These are the most computationally expensive steps
  clock.tick("M11R chol inv");
  // mat M11R = chol(M_11, "lower");
  mat M11R = chol(M_11);
  // mat M11Ri = inv(trimatl(M11R));
  mat M11Ri = inv(trimatu(M11R));
  clock.tock("M11R chol inv");
  clock.tick("S");
  // mat S = M_12.t() * M11Ri.t();
  mat D = M_12.t() * M11Ri;
  mat S = D * D.t();
  S = M_22 - S;
  clock.tock("S");
  // clock.tick("V chol 2 S chol");
  // clock.tock("V chol 2 S chol");
  // clock.tick("V chol 2 join");
  // VR = join_vert(join_horiz(M11R, M11Ri.t() * M_12),
  //                join_horiz(zeros(M_L2, M_L1), SR));
  // clock.tock("V chol 2 join");

  clock.tick("S chol inv");
  mat SR = chol(S);
  mat SRi = inv(trimatu(SR));
  clock.tock("S chol inv");

  // clock.tick("V chol inv 2");
  // mat VRinv = join_vert(join_horiz(M11Ri, -M11Ri * D.t() * SRi),
  //                       join_horiz(zeros(M_L2, M_L1), SRi));
  // clock.tock("V chol inv 2");

  clock.tick("Htilde");
  // mat Htilde = VRinv.t() * design_;
  mat &VRinv11 = M11Ri;
  mat &VRinv22 = SRi;
  mat Htilde1 = VRinv11.t() * design_.head_rows(M_L1);
  mat Htilde2 = -SRi.t() * D * M11Ri.t() * design_.head_rows(M_L1) +
                VRinv22.t() * design_.tail_rows(M_L2);
  mat UtVinvU = Htilde1.t() * Htilde1 + Htilde2.t() * Htilde2;
  mat UtVinvU_R = chol(UtVinvU);
  mat Rinv = inv(trimatu(UtVinvU_R));
  mat Htilde2Rinv = Htilde2 * Rinv;
  // mat Htilde = VRinv * Htilde * Rinv;
  mat Htilde = VRinv11 * Htilde1 * Rinv - M11Ri * D.t() * SRi * Htilde2Rinv;
  Htilde = join_vert(Htilde, VRinv22 * Htilde2Rinv);
  clock.tock("Htilde");

  clock.tick("HX");
  // l1 is logdet(Vjj')
  double l1 = 2 * (sum(log(diagvec(M11R))) + sum(log(diagvec(SR))));
  // l2 is logdet(UtVinvjj'U)
  double l2 = 2 * std::real(log_det(trimatu(UtVinvU_R)));

  mat Vtilde1 = VRinv11.t() * dataRegionCombined_.head_rows(M_L1);
  mat Vtilde2 = -SRi.t() * D * M11Ri.t() * dataRegionCombined_.head_rows(M_L1) +
                VRinv22.t() * dataRegionCombined_.tail_rows(M_L2);
  mat VitX = join_vert(VRinv11 * Vtilde1 + -M11Ri * D.t() * SRi * Vtilde2,
                       VRinv22 * Vtilde2);

  vec HX = VitX - Htilde * Htilde.t() * dataRegionCombined_;

  double l3 = as_scalar(dataRegionCombined_.t() * HX);
  double negLL = l1 + l2 + l3;
  clock.tock("HX");
  clock.tock("reml");

  clock.tick("gradients");
  // Compute gradients
  clock.tick("grad setup");
  mat zeroL11 = zeros(numVoxelRegion1_, numVoxelRegion1_);
  mat zeroL22 = zeros(numVoxelRegion2_, numVoxelRegion2_);
  mat dAt_dtau_eta = rbf_deriv(timeSqrd_, tauEta);
  mat dAt_dnugget = eye(numTimePt_, numTimePt_);
  double rho_deriv = sigmoid_inv_derivative(rho, -1, 1);

  // mat VRi_head = VRinv.head_rows(M_L1);
  // mat VRi_tail = VRinv.tail_rows(M_L2);

  clock.tick("temp create VRinv12");
  // This is surprisingly expensive!!
  mat VRinv12 = -M11Ri * D.t() * SRi;
  clock.tock("temp create VRinv12");
  mat VRi_head = join_rows(VRinv11, VRinv12);
  mat VRi_tail = join_rows(zeros(M_L2, M_L1), VRinv22);

  mat rho_block1 = repmat(At, 1, numVoxelRegion2_) * VRi_tail;
  mat rho_bigblock1 = repmat(rho_block1, numVoxelRegion1_, 1);
  // There's a potential optimizaiton in this step by considering the row-block
  // form of rho_bigblock1.

  clock.tock("grad setup");
  clock.tick("grad rho");
  // Gradient for rho
  gradient(0) = 2 * sqrt(kEta1 * kEta2) * trace(VRi_head.t() * rho_bigblock1);
  mat dVdrho_kron1(numVoxelRegion1_, numVoxelRegion2_,
                   fill::value(sqrt(kEta1 * kEta2)));
  dVdrho_kron1 = join_vert(join_horiz(zeroL11, dVdrho_kron1),
                           join_horiz(dVdrho_kron1.t(), zeroL22));
  gradient(0) -= trace(Htilde.t() * kronecker_mmm(dVdrho_kron1, At, Htilde));
  gradient(0) -= as_scalar(HX.t() * kronecker_mvm(dVdrho_kron1, At, HX));
  gradient(0) *= rho_deriv;
  clock.tock("grad rho");

  // Gradient for kEta1
  clock.tick("grad keta1");
  mat oneL11 = ones(numVoxelRegion1_, numVoxelRegion1_);
  mat oneL22 = ones(numVoxelRegion2_, numVoxelRegion2_);
  mat oneL12 = ones(numVoxelRegion1_, numVoxelRegion2_);
  mat dVdketa_kronblock(numVoxelRegion1_, numVoxelRegion2_,
                        fill::value(sqrt(kEta2 / kEta1) * rho / 2));
  mat dVdketa1_kron = join_vert(join_horiz(oneL11, dVdketa_kronblock),
                                join_horiz(dVdketa_kronblock.t(), zeroL22));
  gradient(1) = trace(VRi_head.t() * kronecker_mmm(oneL11, At, VRi_head));
  double keta_trace = trace(VRi_head.t() * rho_bigblock1);
  gradient(1) += sqrt(kEta2 / kEta1) * rho * keta_trace;
  gradient(1) -= trace(Htilde.t() * kronecker_mmm(dVdketa1_kron, At, Htilde)) +
                 as_scalar(HX.t() * kronecker_mvm(dVdketa1_kron, At, HX));
  gradient(1) *= logistic(kEta1);
  clock.tock("grad keta1");

  // Gradient for kEta2
  clock.tick("grad keta2");
  dVdketa_kronblock.fill(sqrt(kEta1 / kEta2) * rho / 2);
  mat dVdketa2_kron = join_vert(join_horiz(zeroL11, dVdketa_kronblock),
                                join_horiz(dVdketa_kronblock.t(), oneL22));
  gradient(2) = trace(VRi_tail.t() * kronecker_mmm(oneL22, At, VRi_tail));
  gradient(2) += sqrt(kEta1 / kEta2) * rho * keta_trace;
  gradient(2) -= trace(Htilde.t() * kronecker_mmm(dVdketa2_kron, At, Htilde)) +
                 as_scalar(HX.t() * kronecker_mvm(dVdketa2_kron, At, HX));
  gradient(2) *= logistic(kEta2);
  clock.tock("grad keta2");

  // Gradient for tauEta
  clock.tick("grad tauEta");
  clock.tick("grad tauEta 1");
  clock.tick("grad tauEta 1");
  gradient(3) = kEta1 * trace(VRi_head.t() *
                              kronecker_mmm(oneL11, dAt_dtau_eta, VRi_head));
  clock.tock("grad tauEta 1");
  clock.tick("grad tauEta 2");
  gradient(3) += kEta2 * trace(VRi_tail.t() *
                               kronecker_mmm(oneL22, dAt_dtau_eta, VRi_tail));
  clock.tock("grad tauEta 2");

  clock.tick("grad tauEta 3");
  rho_block1 = repmat(dAt_dtau_eta, 1, numVoxelRegion2_) * VRi_tail;
  rho_bigblock1 = repmat(rho_block1, numVoxelRegion1_, 1);
  gradient(3) +=
      2 * sqrt(kEta1 * kEta2) * rho * trace(VRi_head.t() * rho_bigblock1);
  clock.tock("grad tauEta 3");
  mat dVdtauEta_kron = join_vert(
      join_horiz(kEta1 * oneL11, sqrt(kEta1 * kEta2) * rho * oneL12),
      join_horiz(sqrt(kEta1 * kEta2) * rho * oneL12.t(), kEta2 * oneL22));
  gradient(3) -=
      trace(Htilde.t() * kronecker_mmm(dVdtauEta_kron, dAt_dtau_eta, Htilde)) +
      as_scalar(HX.t() * kronecker_mvm(dVdtauEta_kron, dAt_dtau_eta, HX));
  gradient(3) *= logistic(tauEta);
  clock.tock("grad tauEta");

  // Gradient for nugget
  clock.tick("grad nuggetEta");
  gradient(4) = kEta1 * trace(VRi_head.t() * kronecker_mmm(oneL11, dAt_dnugget,
                                                           VRi_head, true));
  gradient(4) += kEta2 * trace(VRi_tail.t() * kronecker_mmm(oneL22, dAt_dnugget,
                                                            VRi_tail, true));

  rho_bigblock1 = repmat(repmat(dAt_dnugget, 1, numVoxelRegion2_) * VRi_tail,
                         numVoxelRegion1_, 1);
  gradient(4) +=
      2 * sqrt(kEta1 * kEta2) * rho * trace(VRi_head.t() * rho_bigblock1);
  gradient(4) -=
      trace(Htilde.t() *
            kronecker_mmm(dVdtauEta_kron, dAt_dnugget, Htilde, true)) +
      as_scalar(HX.t() * kronecker_mvm(dVdtauEta_kron, dAt_dnugget, HX, true));
  gradient(4) *= logistic(nuggetEta);
  clock.tock("grad nuggetEta");
  clock.tock("gradients");
  clock.stop("naptimes");

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
  mat At = rbf(timeSqrd_, tauEta);
  At.diag() += nuggetEta;

  mat M_12 = arma::repmat(rho * sqrt(kEta1) * sqrt(kEta2) * At,
                          numVoxelRegion1_, numVoxelRegion2_);

  mat M_11 =
      spaceTimeKernelRegion1_ +
      kEta1 * arma::repmat(At, numVoxelRegion1_, numVoxelRegion1_) +
      arma::eye(numVoxelRegion1_ * numTimePt_, numVoxelRegion1_ * numTimePt_);
  mat M_22 =
      spaceTimeKernelRegion2_ +
      kEta2 * arma::repmat(At, numVoxelRegion2_, numVoxelRegion2_) +
      arma::eye(numVoxelRegion2_ * numTimePt_, numVoxelRegion2_ * numTimePt_);

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

  vec HX = H * dataRegionCombined_;

  double l3 = arma::as_scalar(dataRegionCombined_.t() * HX);
  double negLL = l1 + l2 + l3;

  return negLL;
}
