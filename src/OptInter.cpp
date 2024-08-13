#include "OptInter.h"
#include "helper.h"
#include "rbf.h"
#include <math.h>

/*****************************************************************************
 Inter-regional model
*****************************************************************************/

OptInter::OptInter(const arma::mat &dataRegion1, const arma::mat &dataRegion2,
                   const arma::vec &stage1ParamsRegion1,
                   const arma::vec &stage1ParamsRegion2,
                   const arma::mat &spaceTimeKernelRegion1,
                   const arma::mat &spaceTimeKernelRegion2,
                   const arma::mat &timeSqrd, bool noiseless)
    : IOptInter(dataRegion1, dataRegion2, stage1ParamsRegion1,
                stage1ParamsRegion2, spaceTimeKernelRegion1,
                spaceTimeKernelRegion2, timeSqrd) {
  noiseless_ = noiseless;
}

// Compute both objective function and its gradient
double OptInter::EvaluateWithGradient(const arma::mat &theta_unrestrict,
                                      arma::mat &gradient) {
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
  // Rcpp::Rcout << "Params: " << rho << " " << kEta1 << " " << kEta2 << " " <<
  // tauEta << " " << nuggetEta << std::endl;

  int M_L1 = numTimePt_ * numVoxelRegion1_;
  int M_L2 = numTimePt_ * numVoxelRegion2_;

  // A Matrix
  mat At = rbf(timeSqrd_, tauEta);
  At.diag() += nuggetEta;

  mat M_12 = arma::repmat(rho * sqrt(kEta1) * sqrt(kEta2) * At,
                          numVoxelRegion1_, numVoxelRegion2_);

  mat M_11 = spaceTimeKernelRegion1_ +
             kEta1 * arma::repmat(At, numVoxelRegion1_, numVoxelRegion1_);
  mat M_22 = spaceTimeKernelRegion2_ +
             kEta2 * arma::repmat(At, numVoxelRegion2_, numVoxelRegion2_);

  if (!noiseless_) {
    M_11 +=
        arma::eye(numVoxelRegion1_ * numTimePt_, numVoxelRegion1_ * numTimePt_);
    M_22 +=
        arma::eye(numVoxelRegion2_ * numTimePt_, numVoxelRegion2_ * numTimePt_);
  }

  mat V = join_vert(join_horiz(M_11, M_12), join_horiz(M_12.t(), M_22));

  mat VR = arma::chol(V);
  mat VRinv = arma::inv(arma::trimatu(VR));

  mat Vinv = VRinv * VRinv.t();
  mat VinvU = Vinv * design_;
  mat UtVinvU = design_.t() * VinvU;
  mat UtVinvU_R = arma::chol(UtVinvU);
  mat Rinv = arma::inv(arma::trimatu(UtVinvU_R));
  mat H = VinvU * Rinv;
  H *= H.t();
  H = Vinv - H;

  // l1 is logdet(Vjj')
  double l1 = 2 * arma::sum(arma::log(arma::diagvec(VR)));
  // l2 is logdet(UtVinvjj'U)
  double l2 = 2 * std::real(arma::log_det(arma::trimatu(UtVinvU_R)));

  vec HX = H * dataRegionCombined_;

  double l3 = arma::as_scalar(dataRegionCombined_.t() * HX);
  double negLL = l1 + l2 + l3;

  // Compute gradients
  mat At_11 = arma::repmat(At, numVoxelRegion1_, numVoxelRegion1_);
  mat At_22 = arma::repmat(At, numVoxelRegion2_, numVoxelRegion2_);
  mat At_12 = arma::repmat(At, numVoxelRegion1_, numVoxelRegion2_);

  mat dkEta2_12 = At_12 * sqrt(kEta2) * rho / (2 * sqrt(kEta1));

  mat commondiag1 = At_12 * sqrt(kEta2) * rho / (2 * sqrt(kEta1));
  mat commondiag2 = At_12 * sqrt(kEta1) * rho / (2 * sqrt(kEta2));
  mat zeroML1 = arma::zeros(M_L1, M_L1);
  mat zeroML2 = arma::zeros(M_L2, M_L2);

  mat dVdkEta1 = join_vert(join_horiz(At_11, commondiag1),
                           join_horiz(commondiag2.t(), zeroML2));

  mat dVdkEta2 = join_vert(join_horiz(zeroML1, commondiag2),
                           join_horiz(commondiag1.t(), At_22));

  mat commonRhoDiag = At_12 * sqrt(kEta1 * kEta2);
  mat dVdrho = join_vert(join_horiz(zeroML1, commonRhoDiag),
                         join_horiz(commonRhoDiag.t(), zeroML2));

  mat dAt_dtau_eta = rbf_deriv(timeSqrd_, tauEta);
  mat cross = arma::repmat(dAt_dtau_eta, numVoxelRegion1_, numVoxelRegion2_) *
              sqrt(kEta1 * kEta2) * rho;
  mat dVdtauEta = join_vert(
      join_horiz(
          arma::repmat(dAt_dtau_eta, numVoxelRegion1_, numVoxelRegion1_) *
              kEta1,
          cross),
      join_horiz(cross.t(), arma::repmat(dAt_dtau_eta, numVoxelRegion2_,
                                         numVoxelRegion2_) *
                                kEta2));

  mat dAt_dnugget = arma::eye(numTimePt_, numTimePt_);
  cross = arma::repmat(dAt_dnugget, numVoxelRegion1_, numVoxelRegion2_) *
          sqrt(kEta1 * kEta2) * rho;
  mat dVdnugget = join_vert(
      join_horiz(arma::repmat(dAt_dnugget, numVoxelRegion1_, numVoxelRegion1_) *
                     kEta1,
                 cross),
      join_horiz(cross.t(),
                 arma::repmat(dAt_dnugget, numVoxelRegion2_, numVoxelRegion2_) *
                     kEta2));

  double rho_deriv = sigmoid_inv_derivative(rho, -1, 1);

  H -= HX * HX.t();
  gradient(0) = rho_deriv * trace(dVdrho * H);
  gradient(1) = logistic(kEta1) * trace(dVdkEta1 * H);
  gradient(2) = logistic(kEta2) * trace(dVdkEta2 * H);
  gradient(3) = logistic(tauEta) * trace(dVdtauEta * H);
  gradient(4) = logistic(nuggetEta) * trace(dVdnugget * H);

  // Rcpp::Rcout << "NegLL: " << std::setprecision(10) << negLL << std::endl;

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
