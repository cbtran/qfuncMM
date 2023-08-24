#ifndef OPTINTEROLD_H
#define OPTINTEROLD_H

#include <RcppArmadillo.h>

class OptInterOld
{
  const arma::mat& dataRegionPair_; // The data matrix from two regions.
  const arma::mat& design_;
  int numVoxelRegion1_;
  int numVoxelRegion2_;
  int numTimePt_;
  const arma::mat& spatialKernelRegion1_;
  const arma::mat& spatialKernelRegion2_;
  const arma::mat& timeSqrd_;

public:
  OptInterOld(const arma::mat& dataRegionPair,
           const arma::mat& design,
           int numVoxelRegion1,
           int numVoxelRegion2,
           int numTimePt,
           const arma::mat& spatialKernelRegion1,
           const arma::mat& spatialKernelRegion2,
           const arma::mat& timeSqrd);

  // Compute both objective function and its gradient
  double EvaluateWithGradient(
    const arma::mat &theta_unrestrict, arma::mat &gradient);
};

#endif