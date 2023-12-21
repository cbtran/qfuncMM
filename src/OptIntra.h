#ifndef OPTINTRA_H
#define OPTINTRA_H
#include <RcppArmadillo.h>

#include "get_cor_mat.h"

class OptIntra {
  const arma::mat& data_;      // The data matrix.
  const arma::mat& distSqrd_;  // Square spatial distance matrix
  const arma::mat& timeSqrd_;  // Square temporal distance matrix
  int numVoxel_;
  int numTimePt_;
  KernelType kernelType_;  // Choice of spatial kernel
  double noiseVarianceEstimate_;

 public:
  OptIntra(const arma::mat& data, const arma::mat& distSqrd,
           const arma::mat& timeSqrd, KernelType kernelType);

  // Compute objective function update gradient
  double EvaluateWithGradient(const arma::mat& theta, arma::mat& gradient);
  double GetNoiseVarianceEstimate();
};

#endif