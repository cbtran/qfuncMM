#ifndef OPTINTER_H
#define OPTINTER_H

#include <RcppArmadillo.h>

class IOptInter {
protected:
  arma::vec dataRegionCombined_;
  arma::mat design_;
  int numVoxelRegion1_;
  int numVoxelRegion2_;
  int numTimePt_;
  const arma::mat &spaceTimeKernelRegion1_;
  const arma::mat &spaceTimeKernelRegion2_;
  const arma::mat &timeSqrd_;
  std::pair<double, double> noiseVarianceEstimates_;

public:
  IOptInter(const arma::mat &dataRegion1, const arma::mat &dataRegion2,
            const arma::vec &stage1ParamsRegion1,
            const arma::vec &stage1ParamsRegion2,
            const arma::mat &spatialKernelRegion1,
            const arma::mat &spatialKernelRegion2, const arma::mat &timeSqrd)
      : numVoxelRegion1_(dataRegion1.n_cols),
        numVoxelRegion2_(dataRegion2.n_cols), numTimePt_(dataRegion1.n_rows),
        spaceTimeKernelRegion1_(spatialKernelRegion1),
        spaceTimeKernelRegion2_(spatialKernelRegion2), timeSqrd_(timeSqrd) {
    using namespace arma;
    design_ = join_vert(join_horiz(ones(numTimePt_ * numVoxelRegion1_, 1),
                                   zeros(numTimePt_ * numVoxelRegion1_, 1)),
                        join_horiz(zeros(numTimePt_ * numVoxelRegion2_, 1),
                                   ones(numTimePt_ * numVoxelRegion2_, 1)));
  }

  virtual double EvaluateWithGradient(const arma::mat &theta_unrestrict,
                                      arma::mat &gradient) = 0;

  virtual double Evaluate(const arma::mat &theta) = 0;

  std::pair<double, double> GetNoiseVarianceEstimates() {
    return noiseVarianceEstimates_;
  }
};

class OptInter : public IOptInter {
  bool noiseless_;

public:
  OptInter(const arma::mat &dataRegion1, const arma::mat &dataRegion2,
           const arma::vec &stage1ParamsRegion1,
           const arma::vec &stage1ParamsRegion2,
           const arma::mat &spatialKernelRegion1,
           const arma::mat &spatialKernelRegion2, const arma::mat &timeSqrd,
           bool noiseless)
      : IOptInter(dataRegion1, dataRegion2, stage1ParamsRegion1,
                  stage1ParamsRegion2, spatialKernelRegion1,
                  spatialKernelRegion2, timeSqrd),
        noiseless_(noiseless) {
    if (noiseless) {
      noiseVarianceEstimates_ = std::make_pair(NA_REAL, NA_REAL);
      dataRegionCombined_ =
          join_vert(vectorise(dataRegion1), vectorise(dataRegion2));
    } else {
      noiseVarianceEstimates_ =
          std::make_pair(stage1ParamsRegion1(4), stage1ParamsRegion2(4));
      dataRegionCombined_ = join_vert(
          vectorise(dataRegion1) / sqrt(noiseVarianceEstimates_.first),
          vectorise(dataRegion2) / sqrt(noiseVarianceEstimates_.second));
    }
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
                   const arma::mat &timeSqrd)
      : IOptInter(dataRegion1, dataRegion2, stage1ParamsRegion1,
                  stage1ParamsRegion2, spatialKernelRegion1,
                  spatialKernelRegion2, timeSqrd) {

    noiseVarianceEstimates_ =
        std::make_pair(stage1ParamsRegion1(4), stage1ParamsRegion2(4));
    dataRegionCombined_ = join_vert(
        vectorise(dataRegion1) / sqrt(noiseVarianceEstimates_.first),
        vectorise(dataRegion2) / sqrt(noiseVarianceEstimates_.second));
  }

  double EvaluateWithGradient(const arma::mat &theta_unrestrict,
                              arma::mat &gradient) override;

  double Evaluate(const arma::mat &theta) override;
};

#endif