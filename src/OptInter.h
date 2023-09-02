#ifndef OPTINTER_H
#define OPTINTER_H

#include <RcppArmadillo.h>

class OptInter
{
  const arma::vec dataRegion1_;
  const arma::vec dataRegion2_;
  const arma::vec dataRegionCombined_;
  arma::mat design_;
  int numVoxelRegion1_;
  int numVoxelRegion2_;
  int numTimePt_;
  const arma::vec& stage1ParamsRegion1_;
  const arma::vec& stage1ParamsRegion2_;
  const arma::mat& spaceTimeKernelRegion1_;
  const arma::mat& spaceTimeKernelRegion2_;
  const arma::mat& timeSqrd_;

public:
  OptInter(const arma::mat& dataRegion1,
           const arma::mat& dataRegion2,
           const arma::vec& stage1ParamsRegion1,
           const arma::vec& stage1ParamsRegion2,
           const arma::mat& spatialKernelRegion1,
           const arma::mat& spatialKernelRegion2,
           const arma::mat& timeSqrd);

  // Compute both objective function and its gradient
  double EvaluateWithGradient(
      const arma::mat &theta_unrestrict, arma::mat &gradient);

  // Compute both objective function and its gradient
  double Evaluate(const arma::mat &theta);
};

#endif