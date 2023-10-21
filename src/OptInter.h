#ifndef OPTINTER_H
#define OPTINTER_H

#include <RcppArmadillo.h>

class OptInter
{
protected:
  const arma::vec dataRegion1_;
  const arma::vec dataRegion2_;
  const arma::vec dataRegionCombined_;
  arma::mat design_;
  int numVoxelRegion1_;
  int numVoxelRegion2_;
  int numTimePt_;
  const arma::mat& spaceTimeKernelRegion1_;
  const arma::mat& spaceTimeKernelRegion2_;
  const arma::mat& timeSqrd_;
  std::pair<double, double> noiseVarianceEstimates_;

public:
  OptInter(const arma::mat& dataRegion1,
           const arma::mat& dataRegion2,
           const arma::vec& stage1ParamsRegion1,
           const arma::vec& stage1ParamsRegion2,
           const arma::mat& spatialKernelRegion1,
           const arma::mat& spatialKernelRegion2,
           const arma::mat& timeSqrd);

  // Compute both objective function and its gradient
  virtual double EvaluateWithGradient(
      const arma::mat &theta_unrestrict, arma::mat &gradient);

  // double EvaluateWithGradientFast(
  //     const arma::mat &theta_unrestrict, arma::mat &gradient);

  // Compute both objective function and its gradient
  virtual double Evaluate(const arma::mat &theta);

  std::pair<double, double> GetNoiseVarianceEstimates();
};

class OptInterProfiled : public OptInter
{
private:
  double ComputeGradient(
    const arma::mat &H, const arma::mat &dHdq, const arma::mat &dVdq);

public:
  OptInterProfiled(const arma::mat& dataRegion1,
               const arma::mat& dataRegion2,
               const arma::vec& stage1ParamsRegion1,
               const arma::vec& stage1ParamsRegion2,
               const arma::mat& spatialKernelRegion1,
               const arma::mat& spatialKernelRegion2,
               const arma::mat& timeSqrd);

  double EvaluateWithGradient(
      const arma::mat &theta_unrestrict, arma::mat &gradient);

  double Evaluate(const arma::mat &theta);

  double EvaluateProfiled(double rho, const arma::mat& fixed_params);
};

#endif