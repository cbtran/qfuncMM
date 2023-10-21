#include "OptInter.h"
#include <math.h>
#include "Rcpp/exceptions.h"
#include "rbf.h"
#include "helper.h"
#include "OptException.h"

/*****************************************************************************
 Inter-regional model
*****************************************************************************/

OptInter::OptInter(const arma::mat& dataRegion1, const arma::mat& dataRegion2,
                   const arma::vec& stage1ParamsRegion1,
                   const arma::vec& stage1ParamsRegion2,
                   const arma::mat& spaceTimeKernelRegion1,
                   const arma::mat& spaceTimeKernelRegion2,
                   const arma::mat& timeSqrd)
    : dataRegion1_(arma::vectorise(dataRegion1)),
      dataRegion2_(arma::vectorise(dataRegion2)),
      dataRegionCombined_(arma::join_vert(dataRegion1_, dataRegion2_)),
      numVoxelRegion1_(dataRegion1.n_cols),
      numVoxelRegion2_(dataRegion2.n_cols),
      numTimePt_(dataRegion1.n_rows),
      spaceTimeKernelRegion1_(spaceTimeKernelRegion1),
      spaceTimeKernelRegion2_(spaceTimeKernelRegion2),
      timeSqrd_(timeSqrd),
      noiseVarianceEstimates_(std::make_pair(stage1ParamsRegion1(4), stage1ParamsRegion2(4)))
{
  using namespace arma;
  design_ = join_vert(join_horiz(ones(numTimePt_ * numVoxelRegion1_, 1),
                                 zeros(numTimePt_ * numVoxelRegion1_, 1)),
                      join_horiz(zeros(numTimePt_ * numVoxelRegion2_, 1),
                                 ones(numTimePt_ * numVoxelRegion2_, 1)));
}

  // Compute both objective function and its gradient
double OptInter::EvaluateWithGradient(
    const arma::mat &theta_unrestrict, arma::mat &gradient)
{
  if (abs(theta_unrestrict(0)) > 1e5) {
    throw OptException("Possible poor initialization detected.");
  }

  using arma::mat;
  using arma::vec;
  using arma::join_horiz;
  using arma::join_vert;

  // theta parameter list:
  // rho, kEta1, kEta2, tauEta, nugget
  // Transform unrestricted parameters to original forms.
  double rho = sigmoid_inv(theta_unrestrict(0), -1, 1);
  double kEta1 = softplus(theta_unrestrict(1));
  double kEta2 = softplus(theta_unrestrict(2));
  double tauEta = softplus(theta_unrestrict(3));
  double nuggetEta = softplus(theta_unrestrict(4));
  std::cout << "params: " << rho << " " << kEta1 << " " << kEta2 << " " << tauEta << " " << nuggetEta << std::endl;
  std::cout << "params unrestrict: " << theta_unrestrict(0) << ", " << theta_unrestrict(1) << ", " << theta_unrestrict(2) << ", " << theta_unrestrict(3) << ", " << theta_unrestrict(4) << std::endl;

  // log-likelihood components
  int M_L1 = numTimePt_*numVoxelRegion1_;
  int M_L2 = numTimePt_*numVoxelRegion2_;

  // A Matrix
  mat At = rbf(timeSqrd_, tauEta) + diagmat(vec(numTimePt_, arma::fill::value(nuggetEta)));

  mat M_12 = arma::repmat(
      rho*sqrt(kEta1)*sqrt(kEta2)*At, numVoxelRegion1_, numVoxelRegion2_);

  mat M_11 = spaceTimeKernelRegion1_ +
      kEta1 * arma::repmat(At, numVoxelRegion1_, numVoxelRegion1_) +
      arma::eye(numVoxelRegion1_ * numTimePt_, numVoxelRegion1_ * numTimePt_);
  mat M_22 = spaceTimeKernelRegion2_ +
      kEta2 * arma::repmat(At, numVoxelRegion2_, numVoxelRegion2_) +
      arma::eye(numVoxelRegion2_ * numTimePt_, numVoxelRegion2_ * numTimePt_);

  mat V = join_vert(
    join_horiz(M_11, M_12),
    join_horiz(M_12.t(), M_22)
  );

  mat Vinv = arma::inv_sympd(V);
  mat VinvU = Vinv * design_;
  mat UtVinvU = design_.t() * VinvU;
  mat H = Vinv - VinvU * arma::inv_sympd(UtVinvU) * VinvU.t();

  // l1 is logdet(Vjj')
  // l1 = arma::sum(arma::log(M_22_chol.diag())) + arma::sum(arma::log(C_11_chol.diag()));
  double l1 = arma::log_det_sympd(V);
  // l2 is logdet(UtVinvjj'U)
  double l2 = arma::log_det_sympd(UtVinvU);

  vec scaleStd = arma::join_vert(arma::ones(M_L1) / sqrt(noiseVarianceEstimates_.first),
                                arma::ones(M_L2) / sqrt(noiseVarianceEstimates_.second));
  mat scaleStdDiag = arma::diagmat(scaleStd);
  mat Hscaled = H*scaleStdDiag * dataRegionCombined_ * dataRegionCombined_.t() * scaleStdDiag;

  double l3 = arma::trace(Hscaled);

  double negLL = l1 + l2 + l3;


  // Compute gradients
  mat At_11 = arma::repmat(At, numVoxelRegion1_, numVoxelRegion1_);
  mat At_22 = arma::repmat(At, numVoxelRegion2_, numVoxelRegion2_);
  mat At_12 = arma::repmat(At, numVoxelRegion1_, numVoxelRegion2_);

  mat dkEta2_12 = At_12 * sqrt(kEta2) * rho / (2 * sqrt(kEta1));

  mat dVdkEta1 = join_vert(
    join_horiz(At_11, At_12 * sqrt(kEta2) * rho / (2 * sqrt(kEta1))),
    join_horiz(At_12.t() * sqrt(kEta1) * rho / (2 * sqrt(kEta2)),
      arma::zeros(M_L2, M_L2))
  );

  mat dVdkEta2 = join_vert(
    join_horiz(arma::zeros(M_L1, M_L1), At_12 * sqrt(kEta1) * rho / (2 * sqrt(kEta2))),
    join_horiz(At_12.t() * sqrt(kEta2) * rho / (2 * sqrt(kEta1)), At_22)
  );

  mat dVdrho = join_vert(
    join_horiz(arma::zeros(M_L1, M_L1), At_12 * sqrt(kEta1 * kEta2)),
    join_horiz(At_12.t() * sqrt(kEta1 * kEta2), arma::zeros(M_L2, M_L2))
  );

  mat dAt_dtau_eta = rbf_deriv(timeSqrd_, tauEta);
  mat cross = arma::repmat(dAt_dtau_eta, numVoxelRegion1_, numVoxelRegion2_) * sqrt(kEta1 * kEta2) * rho;
  mat dVdtauEta = join_vert(
    join_horiz(arma::repmat(dAt_dtau_eta, numVoxelRegion1_, numVoxelRegion1_) * kEta1,
              cross),
    join_horiz(cross.t(),
              arma::repmat(dAt_dtau_eta, numVoxelRegion2_, numVoxelRegion2_) * kEta2)
  );

  mat dAt_dnugget = arma::eye(numTimePt_, numTimePt_);
  cross = arma::repmat(dAt_dnugget, numVoxelRegion1_, numVoxelRegion2_) * sqrt(kEta1 * kEta2) * rho;
  mat dVdnugget = join_vert(
    join_horiz(arma::repmat(dAt_dnugget, numVoxelRegion1_, numVoxelRegion1_) * kEta1,
              cross),
    join_horiz(cross.t(),
              arma::repmat(dAt_dnugget, numVoxelRegion2_, numVoxelRegion2_) * kEta2)
  );

  double rho_deriv = sigmoid_inv_derivative(rho, -1, 1);

  Hscaled *= -1;
  Hscaled.diag() += 1;
  gradient(0) =  rho_deriv * trace(H * dVdrho * Hscaled);
  gradient(1) =  logistic(kEta1) * trace(H * dVdkEta1 * Hscaled);
  gradient(2) =  logistic(kEta2) * trace(H * dVdkEta2 * Hscaled);
  gradient(3) =  logistic(tauEta) * trace(H * dVdtauEta * Hscaled);
  gradient(4) =  logistic(nuggetEta) * trace(H * dVdnugget * Hscaled);

  std::cout << "NegLL, Grad: " << std::setprecision(5) << negLL << ", "
    << gradient(0) << ", " << gradient(1) << ", " << gradient(2) << ", " << gradient(3) << ", " << gradient(4) << ", "
            << arma::norm(gradient) <<  std::endl;

  return negLL;
}

double OptInter::Evaluate(const arma::mat &theta)
{
  using arma::mat;
  using arma::vec;
  using arma::join_horiz;
  using arma::join_vert;

  // theta parameter list:
  // rho, kEta1, kEta2, tauEta, nugget
  // Transform unrestricted parameters to original forms.
  double rho = theta(0);
  double kEta1 = theta(1);
  double kEta2 = theta(2);
  double tauEta = theta(3);
  double nuggetEta = theta(4);
  // double nuggetEta = softplus(theta_unrestrict(4));
  std::cout << "params: " << rho << " " << kEta1 << " " << kEta2 << " " << tauEta << " " << nuggetEta << std::endl;
  // std::cout << "params unrestrict: " << theta_unrestrict(0) << ", " << theta_unrestrict(1) << ", " << theta_unrestrict(2) << ", " << theta_unrestrict(3) << ", " << theta_unrestrict(4) << std::endl;

  if (rho + 1 < 0.0001) {
    throw Rcpp::exception("uh oh");
  }

  // log-likelihood components
  int M_L1 = numTimePt_*numVoxelRegion1_;
  int M_L2 = numTimePt_*numVoxelRegion2_;

  // Construct the Sigma_alpha matrix.

  // A Matrix

  mat At = rbf(timeSqrd_, tauEta) + diagmat(vec(numTimePt_, arma::fill::value(nuggetEta)));

  mat M_12 = arma::repmat(
      rho*sqrt(kEta1)*sqrt(kEta2)*At, numVoxelRegion1_, numVoxelRegion2_);

  mat M_11 = spaceTimeKernelRegion1_ +
      kEta1 * arma::repmat(At, numVoxelRegion1_, numVoxelRegion1_) +
      arma::eye(numVoxelRegion1_ * numTimePt_, numVoxelRegion1_ * numTimePt_);
  mat M_22 = spaceTimeKernelRegion2_ +
      kEta2 * arma::repmat(At, numVoxelRegion2_, numVoxelRegion2_) +
      arma::eye(numVoxelRegion2_ * numTimePt_, numVoxelRegion2_ * numTimePt_);

  mat V = join_vert(
    join_horiz(M_11, M_12),
    join_horiz(M_12.t(), M_22)
  );

  mat Vinv = arma::inv_sympd(V);
  mat VinvU = Vinv * design_;
  mat UtVinvU = design_.t() * VinvU;
  mat H = Vinv - VinvU * arma::inv_sympd(UtVinvU) * VinvU.t();

  // l1 is logdet(Vjj')
  // l1 = arma::sum(arma::log(M_22_chol.diag())) + arma::sum(arma::log(C_11_chol.diag()));
  double l1 = arma::log_det_sympd(V);
  // l2 is logdet(UtVinvjj'U)
  double l2 = arma::log_det_sympd(UtVinvU);

  vec scaleStd = arma::join_vert(arma::ones(M_L1) / sqrt(noiseVarianceEstimates_.first),
                                arma::ones(M_L2) / sqrt(noiseVarianceEstimates_.second));
  mat scaleStdDiag = arma::diagmat(scaleStd);
  mat Hscaled = H*scaleStdDiag * dataRegionCombined_ * dataRegionCombined_.t() * scaleStdDiag;

  double l3 = arma::trace(Hscaled);

  double negLL = l1 + l2 + l3;
  return negLL;
}

std::pair<double, double> OptInter::GetNoiseVarianceEstimates()
{
  return noiseVarianceEstimates_;
}