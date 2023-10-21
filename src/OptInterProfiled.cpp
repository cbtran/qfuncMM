#include "OptInter.h"
#include <math.h>
#include <cmath>
#include "rbf.h"
#include "helper.h"

/*****************************************************************************
 Inter-regional model: Option 2 ReML
*****************************************************************************/

OptInterProfiled::OptInterProfiled(const arma::mat& dataRegion1,
                           const arma::mat& dataRegion2,
                           const arma::vec& stage1ParamsRegion1,
                           const arma::vec& stage1ParamsRegion2,
                           const arma::mat& spaceTimeKernelRegion1,
                           const arma::mat& spaceTimeKernelRegion2,
                           const arma::mat& timeSqrd)
    : OptInter(dataRegion1, dataRegion2, stage1ParamsRegion1,
               stage1ParamsRegion2, spaceTimeKernelRegion1,
               spaceTimeKernelRegion2, timeSqrd) {}

double OptInterProfiled::ComputeGradient(const arma::mat& H, const arma::mat& dHdq,
                                     const arma::mat& dVdq)
{
  using arma::as_scalar;
  double profVarNoise1 = noiseVarianceEstimates_.first;
  double profVarNoise2 = noiseVarianceEstimates_.second;
  int ML1 = numTimePt_ * numVoxelRegion1_;
  int ML2 = numTimePt_ * numVoxelRegion2_;
  double a1 = as_scalar(dataRegion1_.t() * dHdq.submat(0, 0, ML1 - 1, ML1 - 1) *
                        dataRegion1_) /
              profVarNoise1;
  double a2 = as_scalar(dataRegion2_.t() *
                        dHdq.submat(ML1, ML1, ML1 + ML2 - 1, ML1 + ML2 - 1) *
                        dataRegion2_) /
              profVarNoise2;
  double b =
      as_scalar(dataRegion1_.t() * dHdq.submat(0, ML1, ML1 - 1, ML1 + ML2 - 1) *
                dataRegion2_) /
      sqrt(profVarNoise1 * profVarNoise2);
  double c1 = as_scalar(dataRegion1_.t() * H.submat(0, 0, ML1 - 1, ML1 - 1) *
                        dataRegion1_) /
              pow(profVarNoise1, 2);
  double c2 = as_scalar(dataRegion2_.t() *
                        H.submat(ML1, ML1, ML1 + ML2 - 1, ML1 + ML2 - 1) *
                        dataRegion2_) /
              pow(profVarNoise2, 2);
  double cross =
      as_scalar(dataRegion1_.t() * H.submat(0, ML1, ML1 - 1, ML1 + ML2 - 1) *
                dataRegion2_);
  double d = cross / (2 * pow(profVarNoise1, 1.5) * sqrt(profVarNoise2));
  double e = cross / (2 * pow(profVarNoise2, 1.5) * sqrt(profVarNoise1));

  double denom = d * c2 + c1 * (e + c2);
  double dprofvar1dq = (-e * a2 + b * c2 + a1 * (e + c2)) / denom;
  double dprofvar2dq = (-d * a1 + b * c1 + a2 * (d + c1)) / denom;

//   std::cout << "grad eq1: " << (a1 + b) - (c1 + d) * dprofvar1dq - e * dprofvar2dq << std::endl;
//   std::cout << "grad eq2: " << (a2 + b) - (c2 + e) * dprofvar2dq - d * dprofvar1dq << std::endl;

  double result = arma::trace(H * dVdq) +
                  (ML1 - 1) * dprofvar1dq / (2 * profVarNoise1) +
                  (ML2 - 1) * dprofvar2dq / (2 * profVarNoise2);
  return result;
}

double OptInterProfiled::EvaluateWithGradient(const arma::mat& theta_unrestrict,
                                          arma::mat& gradient) {
  using arma::join_horiz;
  using arma::join_vert;
  using arma::mat;
  using arma::vec;

  // theta parameter list:
  // rho, kEta1, kEta2, tauEta, nugget, noiseVar1, noiseVar2
  // Transform unrestricted parameters to original forms.
  double rho = sigmoid_inv(theta_unrestrict(0), -1, 1);
  double kEta1 = 0.5;
  double kEta2 = 0.5;
  double tauEta = 0.25;
  double nugget = 0.1;
  // double kEta1 = softplus(theta_unrestrict(1));
  // double kEta2 = softplus(theta_unrestrict(2));
//   double tauEta = softplus(theta_unrestrict(3));
//   double nugget = softplus(theta_unrestrict(4));
  std::cout << "params: " << rho << " " << kEta1 << " " << kEta2 << " "
            << tauEta << " " << nugget << std::endl;

  // log-likelihood components
  int ML1 = numTimePt_ * numVoxelRegion1_;
  int ML2 = numTimePt_ * numVoxelRegion2_;

  // Construct the Sigma_alpha matrix.

  // Spatial covariance matrix
  mat At = rbf(timeSqrd_, tauEta) + nugget * arma::eye(numTimePt_, numTimePt_);

  mat M_12 = arma::repmat(rho * sqrt(kEta1) * sqrt(kEta2) * At,
                          numVoxelRegion1_, numVoxelRegion2_);

  mat M_11 = spaceTimeKernelRegion1_ +
             kEta1 * arma::repmat(At, numVoxelRegion1_, numVoxelRegion1_) +
             arma::eye(ML1, ML1);
  mat M_22 = spaceTimeKernelRegion2_ +
             kEta2 * arma::repmat(At, numVoxelRegion2_, numVoxelRegion2_) +
             arma::eye(ML2, ML2);

  mat V = join_vert(join_horiz(M_11, M_12), join_horiz(M_12.t(), M_22));

  mat Vinv = arma::inv_sympd(V);
  mat VinvU = Vinv * design_;
  mat UtVinvU = design_.t() * VinvU;
  mat H = Vinv - VinvU * arma::inv_sympd(UtVinvU) * VinvU.t();
  mat H11 = H.submat(0, 0, ML1 - 1, ML1 - 1);
  mat H12 = H.submat(0, ML1, ML1 - 1, ML1 + ML2 - 1);
  mat H22 = H.submat(ML1, ML1, ML1 + ML2 - 1, ML1 + ML2 - 1);

  // TODO: we are repeating this computation in the gradient function.
  double pA = arma::as_scalar(dataRegion1_.t() * H11 * dataRegion1_);
  double pB = arma::as_scalar(dataRegion2_.t() * H22 * dataRegion2_);
  double pC = arma::as_scalar(dataRegion1_.t() * H12 * dataRegion2_);
  double pC2 = pow(pC, 2);
  double alpha = ML1 - 1;
  double beta = ML2 - 1;
  double discrim = pC * sqrt(4 * alpha * beta * pA * pB + pC2 * pow(alpha - beta, 2));
  double profVarNoise1 =
      (2 * alpha * pA * pB + discrim - alpha * pC2 + beta * pC2) /
      (2 * pow(alpha, 2) * pB);
  double profVarNoise2 =
      (2 * beta * pA * pB + discrim - beta * pC2 + alpha * pC2) /
      (2 * pow(beta, 2) * pA);
  noiseVarianceEstimates_.first = profVarNoise1;
  noiseVarianceEstimates_.second = profVarNoise2;

  // l1 is logdet(Vjj')
  double l1 = arma::log_det_sympd(V);
  // l2 is logdet(UtVinvjj'U)
  double l2 = arma::log_det_sympd(UtVinvU);
  double l3 = alpha * log(profVarNoise1) + beta * log(profVarNoise2);
  double negLL = l1 + l2 + l3;
//   std::cout << "done with NLL\n";

  // Get gradient for each component of the REML function.

  mat At_11 = arma::repmat(At, numVoxelRegion1_, numVoxelRegion1_);
  mat At_22 = arma::repmat(At, numVoxelRegion2_, numVoxelRegion2_);
  mat At_12 = arma::repmat(At, numVoxelRegion1_, numVoxelRegion2_);

  mat dVdkEta1 = join_vert(
      join_horiz(At_11, At_12 * sqrt(kEta2) * rho / (2 * sqrt(kEta1))),
      join_horiz(At_12.t() * sqrt(kEta1) * rho / (2 * sqrt(kEta2)),
                 arma::zeros(ML2, ML2)));
  mat dHdkEta1 = -H * dVdkEta1 * H;

  mat dVdkEta2 = join_vert(
      join_horiz(arma::zeros(ML1, ML1),
                 At_12 * sqrt(kEta1) * rho / (2 * sqrt(kEta2))),
      join_horiz(At_12.t() * sqrt(kEta2) * rho / (2 * sqrt(kEta1)), At_22));
  mat dHdkEta2 = -H * dVdkEta2 * H;

  mat dVdrho = join_vert(
      join_horiz(arma::zeros(ML1, ML1), At_12 * sqrt(kEta1 * kEta2)),
      join_horiz(At_12.t() * sqrt(kEta1 * kEta2), arma::zeros(ML2, ML2)));
  mat dHdrho = -H * dVdrho * H;

  mat dAt_dtau_eta = rbf_deriv(timeSqrd_, tauEta);
  mat dVdtauEta = join_vert(
      join_horiz(
          arma::repmat(dAt_dtau_eta, numVoxelRegion1_, numVoxelRegion1_) *
              kEta1,
          arma::repmat(dAt_dtau_eta, numVoxelRegion1_, numVoxelRegion2_) *
              sqrt(kEta1 * kEta2) * rho),
      join_horiz(
          arma::repmat(dAt_dtau_eta, numVoxelRegion2_, numVoxelRegion1_) *
              sqrt(kEta1 * kEta2) * rho,
          arma::repmat(dAt_dtau_eta, numVoxelRegion2_, numVoxelRegion2_) *
              kEta2));
  mat dHdtauEta = -H * dVdtauEta * H;

  mat dAt_dnugget = arma::eye(numTimePt_, numTimePt_);
  mat dVdnugget = join_vert(
      join_horiz(
          arma::repmat(dAt_dnugget, numVoxelRegion1_, numVoxelRegion1_) * kEta1,
          arma::repmat(dAt_dnugget, numVoxelRegion1_, numVoxelRegion2_) *
              sqrt(kEta1 * kEta2) * rho),
      join_horiz(arma::repmat(dAt_dnugget, numVoxelRegion2_, numVoxelRegion1_) *
                     sqrt(kEta1 * kEta2) * rho,
                 arma::repmat(dAt_dnugget, numVoxelRegion2_, numVoxelRegion2_) *
                     kEta2));
  mat dHdnugget = -H * dVdnugget * H;

  double rho_deriv = sigmoid_inv_derivative(rho, -1, 1);

  gradient(0) = rho_deriv * ComputeGradient(H, dHdrho, dVdrho);
  // gradient(1) = logistic(kEta1) * ComputeGradient(H, dHdkEta1, dVdkEta1);
  // gradient(2) = logistic(kEta2) * ComputeGradient(H, dHdkEta2, dVdkEta2);
//   gradient(3) = logistic(tauEta) * ComputeGradient(H, dHdtauEta, dVdtauEta);
//   gradient(4) = logistic(nugget) * ComputeGradient(H, dHdnugget, dVdnugget);

  std::cout << "NegLL, GradNorm: " << negLL << ", " << arma::norm(gradient) << " " << std::endl;

  return negLL;
}

double OptInterProfiled::Evaluate(const arma::mat& theta)
{
  using arma::join_horiz;
  using arma::join_vert;
  using arma::mat;
  using arma::vec;

  // theta parameter list:
  // rho, kEta1, kEta2, tauEta, nugget
  // Transform unrestricted parameters to original forms.
  double rho = theta(0);
  double kEta1 = 0.5;
  double kEta2 = 0.5;
  double tauEta = 0.25;
  double nugget = 0.1;

  // log-likelihood components
  int ML1 = numTimePt_ * numVoxelRegion1_;
  int ML2 = numTimePt_ * numVoxelRegion2_;

  // Construct the Sigma_alpha matrix.

  // Spatial covariance matrix
  mat At = rbf(timeSqrd_, tauEta) + nugget * arma::eye(numTimePt_, numTimePt_);

  mat M_12 = arma::repmat(rho * sqrt(kEta1) * sqrt(kEta2) * At,
                          numVoxelRegion1_, numVoxelRegion2_);

  mat M_11 = spaceTimeKernelRegion1_ +
             kEta1 * arma::repmat(At, numVoxelRegion1_, numVoxelRegion1_) +
             arma::eye(ML1, ML1);
  mat M_22 = spaceTimeKernelRegion2_ +
             kEta2 * arma::repmat(At, numVoxelRegion2_, numVoxelRegion2_) +
             arma::eye(ML2, ML2);

  mat V = join_vert(join_horiz(M_11, M_12), join_horiz(M_12.t(), M_22));

  mat Vinv = arma::inv_sympd(V);
  mat VinvU = Vinv * design_;
  mat UtVinvU = design_.t() * VinvU;
  mat H = Vinv - VinvU * arma::inv_sympd(UtVinvU) * VinvU.t();
  mat H11 = H.submat(0, 0, ML1 - 1, ML1 - 1);
  mat H12 = H.submat(0, ML1, ML1 - 1, ML1 + ML2 - 1);
  mat H22 = H.submat(ML1, ML1, ML1 + ML2 - 1, ML1 + ML2 - 1);

  // TODO: we are repeating this computation in the gradient function.
  double pA = arma::as_scalar(dataRegion1_.t() * H11 * dataRegion1_);
  double pB = arma::as_scalar(dataRegion2_.t() * H22 * dataRegion2_);
  double pC = arma::as_scalar(dataRegion1_.t() * H12 * dataRegion2_);
  double pC2 = pow(pC, 2);
  double alpha = ML1 - 1;
  double beta = ML2 - 1;
  double discrim = pC * sqrt(4 * alpha * beta * pA * pB + pC2 * pow(alpha - beta, 2));
  double profVarNoise1 =
      (2 * alpha * pA * pB + discrim - alpha * pC2 + beta * pC2) /
      (2 * pow(alpha, 2) * pB);
  double profVarNoise2 =
      (2 * beta * pA * pB + discrim - beta * pC2 + alpha * pC2) /
      (2 * pow(beta, 2) * pA);
  noiseVarianceEstimates_.first = profVarNoise1;
  noiseVarianceEstimates_.second = profVarNoise2;
  std::cout << "profVarNoise: " << profVarNoise1 << " " << profVarNoise2
            << std::endl;
  std::cout << "profVarNoise TRUE: " << noiseVarianceEstimates_.first
            << " " << noiseVarianceEstimates_.second
            << std::endl;

  // l1 is logdet(Vjj')
  double l1 = arma::log_det_sympd(V);
  // l2 is logdet(UtVinvjj'U)
  double l2 = arma::log_det_sympd(UtVinvU);
  double l3 = alpha * log(profVarNoise1) + beta * log(profVarNoise2);
//   vec scaleStd = arma::join_vert(arma::ones(ML1) / sqrt(profVarNoise1),
//                                 arma::ones(ML2) / sqrt(profVarNoise2));
//   mat scaleStdDiag = arma::diagmat(scaleStd);
//   mat Hscaled = H*scaleStdDiag * dataRegionCombined_ * dataRegionCombined_.t() * scaleStdDiag;
//   double l3 = arma::trace(Hscaled);
//   std::cout << "l1, l2, l3: " << l1 << " " << l2 << " " << l3 << std::endl;
  double negLL = (l1 + l2 + l3) / 2;

  return negLL;
}

double OptInterProfiled::EvaluateProfiled(double rho, const arma::mat& fixed_params)
{
  using arma::join_horiz;
  using arma::join_vert;
  using arma::mat;
  using arma::vec;

  // theta parameter list:
  // rho, kEta1, kEta2, tauEta, nugget
  // Transform unrestricted parameters to original forms.
  double kEta1 = fixed_params(0);
  double kEta2 = fixed_params(1);
  double tauEta = fixed_params(2);
  double nugget = fixed_params(3);

  // log-likelihood components
  int ML1 = numTimePt_ * numVoxelRegion1_;
  int ML2 = numTimePt_ * numVoxelRegion2_;

  // Construct the Sigma_alpha matrix.

  // Spatial covariance matrix
  mat At = rbf(timeSqrd_, tauEta) + nugget * arma::eye(numTimePt_, numTimePt_);

  mat M_12 = arma::repmat(rho * sqrt(kEta1) * sqrt(kEta2) * At,
                          numVoxelRegion1_, numVoxelRegion2_);

  mat M_11 = spaceTimeKernelRegion1_ +
             kEta1 * arma::repmat(At, numVoxelRegion1_, numVoxelRegion1_) +
             arma::eye(ML1, ML1);
  mat M_22 = spaceTimeKernelRegion2_ +
             kEta2 * arma::repmat(At, numVoxelRegion2_, numVoxelRegion2_) +
             arma::eye(ML2, ML2);

  mat V = join_vert(join_horiz(M_11, M_12), join_horiz(M_12.t(), M_22));

  mat Vinv = arma::inv_sympd(V);
  mat VinvU = Vinv * design_;
  mat UtVinvU = design_.t() * VinvU;
  mat H = Vinv - VinvU * arma::inv_sympd(UtVinvU) * VinvU.t();
  mat H11 = H.submat(0, 0, ML1 - 1, ML1 - 1);
  mat H12 = H.submat(0, ML1, ML1 - 1, ML1 + ML2 - 1);
  mat H22 = H.submat(ML1, ML1, ML1 + ML2 - 1, ML1 + ML2 - 1);

  // TODO: we are repeating this computation in the gradient function.
  double pA = arma::as_scalar(dataRegion1_.t() * H11 * dataRegion1_);
  double pB = arma::as_scalar(dataRegion2_.t() * H22 * dataRegion2_);
  double pC = arma::as_scalar(dataRegion1_.t() * H12 * dataRegion2_);
  double pC2 = pow(pC, 2);
  double alpha = ML1 - 1;
  double beta = ML2 - 1;
  double discrim = pC * sqrt(4 * alpha * beta * pA * pB + pC2 * pow(alpha - beta, 2));
  double profVarNoise1 =
      (2 * alpha * pA * pB + discrim - alpha * pC2 + beta * pC2) /
      (2 * pow(alpha, 2) * pB);
  double profVarNoise2 =
      (2 * beta * pA * pB + discrim - beta * pC2 + alpha * pC2) /
      (2 * pow(beta, 2) * pA);
  noiseVarianceEstimates_.first = profVarNoise1;
  noiseVarianceEstimates_.second = profVarNoise2;

  std::cout << "discrim: " << discrim << std::endl;
  std::cout << "pA, pB, pC: " << pA << " " << pB << " " << pC << std::endl;
  std::cout << "equation1: " << alpha - (pA / profVarNoise1 + pC / sqrt(profVarNoise1 * profVarNoise2)) << std::endl;
  std::cout << "equation2: " << beta - (pB / profVarNoise2 + pC / sqrt(profVarNoise1 * profVarNoise2)) << std::endl;

  // l1 is logdet(Vjj')
  double l1 = arma::log_det_sympd(V);
  // l2 is logdet(UtVinvjj'U)
  double l2 = arma::log_det_sympd(UtVinvU);
  double l3 = alpha * log(profVarNoise1) + beta * log(profVarNoise2);
//   vec scaleStd = arma::join_vert(arma::ones(ML1) / sqrt(profVarNoise1),
//                                 arma::ones(ML2) / sqrt(profVarNoise2));
//   mat scaleStdDiag = arma::diagmat(scaleStd);
//   mat Hscaled = H*scaleStdDiag * dataRegionCombined_ * dataRegionCombined_.t() * scaleStdDiag;
//   double l3 = arma::trace(Hscaled);
//   std::cout << "l1, l2, l3: " << l1 << " " << l2 << " " << l3 << std::endl;
  double negLL = l1 + l2 + l3;

  return negLL;
}