#ifndef OPTINTER_H
#define OPTINTER_H

#include "cov_setting.h"
#include <RcppArmadillo.h>

class IOptInter {
protected:
  arma::vec dataRegionCombined_;
  arma::mat design_;
  int numVoxelRegion1_, numVoxelRegion2_;
  int numTimePt_;
  const arma::mat &spaceTimeKernelRegion1_, &spaceTimeKernelRegion2_;
  std::pair<double, double> sigma2_;
  CovSetting cov_setting_region1_, cov_setting_region2_;
  const arma::mat &timeSqrd_;

public:
  IOptInter(const arma::mat &dataRegion1, const arma::mat &dataRegion2,
            const arma::vec &stage1ParamsRegion1,
            const arma::vec &stage1ParamsRegion2,
            const arma::mat &spatialKernelRegion1,
            const arma::mat &spatialKernelRegion2,
            CovSetting cov_setting_region1, CovSetting cov_setting_region2,
            const arma::mat &timeSqrd)
      : numVoxelRegion1_(dataRegion1.n_cols),
        numVoxelRegion2_(dataRegion2.n_cols), numTimePt_(dataRegion1.n_rows),
        spaceTimeKernelRegion1_(spatialKernelRegion1),
        spaceTimeKernelRegion2_(spatialKernelRegion2),
        cov_setting_region1_(cov_setting_region1),
        cov_setting_region2_(cov_setting_region2), timeSqrd_(timeSqrd) {
    using namespace arma;
    design_ = join_vert(join_horiz(ones(numTimePt_ * numVoxelRegion1_, 1),
                                   zeros(numTimePt_ * numVoxelRegion1_, 1)),
                        join_horiz(zeros(numTimePt_ * numVoxelRegion2_, 1),
                                   ones(numTimePt_ * numVoxelRegion2_, 1)));
  }

  virtual double EvaluateWithGradient(const arma::mat &theta_unrestrict,
                                      arma::mat &gradient) = 0;

  virtual double Evaluate(const arma::mat &theta) = 0;

  std::pair<double, double> GetNoiseVarianceEstimates() { return sigma2_; }
};

class OptInter : public IOptInter {
public:
  OptInter(const arma::mat &dataRegion1, const arma::mat &dataRegion2,
           const arma::vec &stage1ParamsRegion1,
           const arma::vec &stage1ParamsRegion2,
           const arma::mat &spatialKernelRegion1,
           const arma::mat &spatialKernelRegion2,
           CovSetting cov_setting_region1, CovSetting cov_setting_region2,
           const arma::mat &timeSqrd)
      : IOptInter(dataRegion1, dataRegion2, stage1ParamsRegion1,
                  stage1ParamsRegion2, spatialKernelRegion1,
                  spatialKernelRegion2, cov_setting_region1,
                  cov_setting_region2, timeSqrd) {
    double sigma2_region1 = cov_setting_region1 == CovSetting::noiseless
                                ? 1
                                : stage1ParamsRegion1(4);
    double sigma2_region2 = cov_setting_region2 == CovSetting::noiseless
                                ? 1
                                : stage1ParamsRegion2(4);
    dataRegionCombined_ =
        join_vert(vectorise(dataRegion1) / sqrt(sigma2_region1),
                  vectorise(dataRegion2) / sqrt(sigma2_region2));
    sigma2_region1 =
        IsNoiseless(cov_setting_region1) ? NA_REAL : stage1ParamsRegion1(4);
    sigma2_region2 =
        IsNoiseless(cov_setting_region2) ? NA_REAL : stage1ParamsRegion2(4);
    sigma2_ = std::make_pair(sigma2_region1, sigma2_region2);
  }

  double EvaluateWithGradient(const arma::mat &theta_unrestrict,
                              arma::mat &gradient) override;

  double Evaluate(const arma::mat &theta) override;
};

class OptInterDiagTime : public IOptInter {
public:
  OptInterDiagTime(const arma::mat &dataRegion1, const arma::mat &dataRegion2,
                   const arma::vec &stage1ParamsRegion1,
                   const arma::vec &stage1ParamsRegion2,
                   const arma::mat &spatialKernelRegion1,
                   const arma::mat &spatialKernelRegion2,
                   CovSetting cov_setting_region1,
                   CovSetting cov_setting_region2, const arma::mat &timeSqrd)
      : IOptInter(dataRegion1, dataRegion2, stage1ParamsRegion1,
                  stage1ParamsRegion2, spatialKernelRegion1,
                  spatialKernelRegion2, cov_setting_region1,
                  cov_setting_region2, timeSqrd) {

    sigma2_ = std::make_pair(stage1ParamsRegion1(4), stage1ParamsRegion2(4));
    dataRegionCombined_ =
        join_vert(vectorise(dataRegion1) / sqrt(sigma2_.first),
                  vectorise(dataRegion2) / sqrt(sigma2_.second));
  }

  double EvaluateWithGradient(const arma::mat &theta_unrestrict,
                              arma::mat &gradient) override;

  double Evaluate(const arma::mat &theta) override;
};

#endif