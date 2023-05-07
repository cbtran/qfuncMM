#ifndef OPTINTRA_H
#define OPTINTRA_H
#include <RcppArmadillo.h>

class OptIntra
{
  const arma::mat& data_; // The data matrix.
  const arma::mat& design_; // The design matrix.
  const arma::mat& distSqrd_; // Square spatial distance matrix
  const arma::mat& timeSqrd_; // Square temporal distance matrix
  int numVoxel_;
  int numTimePt_;
  arma::mat fixedEffect_; // Fixed-effect vector
  std::string kernelType_; // Choice of spatial kernel
  
public:
  OptIntra(
    const arma::mat& data,
    const arma::mat& design, 
    const arma::mat& distSqrd, 
    const arma::mat& timeSqrd,
    int numVoxel, int numTimePt, 
    const arma::mat& fixedEffect, 
    std::string kernelType);

  // Compute both objective function and its gradient
  double EvaluateWithGradient(
    const arma::mat &theta_unrestrict, arma::mat &gradient);
};

#endif