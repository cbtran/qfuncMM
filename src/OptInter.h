#ifndef OPTINTER_H
#define OPTINTER_H

#include "cov_setting.h"
#include <RcppArmadillo.h>

class IOptInter {
protected:
  arma::vec data_;
  arma::mat design_;
  int l1_, l2_, m_; // Number of voxels in region 1, region 2, and time points
  const arma::mat &lambda_r1_, &lambda_r2_; // Space-time kernels from stage 1
  std::pair<double, double> sigma2_ep_;
  CovSetting cov_setting_r1_, cov_setting_r2_;
  const arma::mat &time_sqrd_;

public:
  IOptInter(const arma::mat &data_r1, const arma::mat &data_r2,
            const arma::mat &lambda_r1, const arma::mat &lambda_r2,
            CovSetting cov_setting_r1, CovSetting cov_setting_r2,
            const arma::mat &time_sqrd)
      : l1_(data_r1.n_cols), l2_(data_r2.n_cols), m_(data_r1.n_rows),
        lambda_r1_(lambda_r1), lambda_r2_(lambda_r2),
        cov_setting_r1_(cov_setting_r1), cov_setting_r2_(cov_setting_r2),
        time_sqrd_(time_sqrd) {
    using namespace arma;
    design_ = join_vert(join_horiz(ones(m_ * l1_, 1), zeros(m_ * l1_, 1)),
                        join_horiz(zeros(m_ * l2_, 1), ones(m_ * l2_, 1)));
  }

  virtual double EvaluateWithGradient(const arma::mat &theta_unrestrict,
                                      arma::mat &gradient) = 0;

  virtual double Evaluate(const arma::mat &theta) = 0;

  std::pair<double, double> GetNoiseVarianceEstimates() { return sigma2_ep_; }
};

class OptInter : public IOptInter {
public:
  OptInter(const arma::mat &data_r1, const arma::mat &data_r2,
           const Rcpp::NumericVector &stage1_r1,
           const Rcpp::NumericVector &stage1_r2, const arma::mat &lambda_r1,
           const arma::mat &lambda_r2, CovSetting cov_setting_r1,
           CovSetting cov_setting_r2, const arma::mat &time_sqrd)
      : IOptInter(data_r1, data_r2, lambda_r1, lambda_r2, cov_setting_r1,
                  cov_setting_r2, time_sqrd) {
    double sigma2_ep_region1 = stage1_r1["sigma2_ep"];
    double sigma2_ep_region2 = stage1_r2["sigma2_ep"];
    sigma2_ep_region1 = IsNoiseless(cov_setting_r1) ? 1.0 : sigma2_ep_region1;
    sigma2_ep_region2 = IsNoiseless(cov_setting_r2) ? 1.0 : sigma2_ep_region2;
    data_ = join_vert(vectorise(data_r1) / sqrt(sigma2_ep_region1),
                      vectorise(data_r2) / sqrt(sigma2_ep_region2));

    sigma2_ep_region1 = stage1_r1["sigma2_ep"];
    sigma2_ep_region2 = stage1_r2["sigma2_ep"];
    sigma2_ep_region1 =
        IsNoiseless(cov_setting_r1) ? NA_REAL : sigma2_ep_region1;
    sigma2_ep_region2 =
        IsNoiseless(cov_setting_r2) ? NA_REAL : sigma2_ep_region2;
    sigma2_ep_ = std::make_pair(sigma2_ep_region1, sigma2_ep_region2);
  }

  double EvaluateWithGradient(const arma::mat &theta_unrestrict,
                              arma::mat &gradient) override;

  double Evaluate(const arma::mat &theta) override;

  Rcpp::NumericMatrix ComputeFisherInformation(
      const arma::mat &theta_stage1, const arma::mat &theta_stage2,
      const arma::mat &dist_sqrd1, const arma::mat &dist_sqrd2, arma::mat *C1,
      arma::mat *B1, arma::mat *C2, arma::mat *B2, bool reml);

  double ComputeAsympVarRhoApprox(const arma::mat &theta_stage2,
                                  const arma::mat &dist_sqrd1,
                                  const arma::mat &dist_sqrd2, bool reml);

  double ComputeAsympVarRhoApproxVecchia(const arma::mat &theta_stage2,
                                         const arma::mat &dist_sqrd1,
                                         const arma::mat &dist_sqrd2);
};

#endif