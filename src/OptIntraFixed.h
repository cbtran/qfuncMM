#ifndef OPTINTRAFIXED_H
#define OPTINTRAFIXED_H
#include <RcppArmadillo.h>

#include "get_cor_mat.h"

class OptIntraFixed {
  const arma::mat& data_;      // The data matrix.
  const arma::mat& distSqrd_;  // Square spatial distance matrix
  const arma::mat& timeSqrd_;  // Square temporal distance matrix
  int numVoxel_;
  int numTimePt_;
  KernelType kernelType_;  // Choice of spatial kernel
  double noiseVarianceEstimate_;
  const arma::vec& thetaFixed_;

 public:
  OptIntraFixed(const arma::mat& data, const arma::mat& distSqrd,
           const arma::mat& timeSqrd, int numVoxel, int numTimePt,
           KernelType kernelType, const arma::vec& thetaFixed);

  // Compute objective function update gradient
  double EvaluateWithGradient(const arma::mat& theta, arma::mat& gradient);
  double GetNoiseVarianceEstimate();
};

#endif