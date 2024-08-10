#ifndef OPTINTRA_H
#define OPTINTRA_H
#include <RcppArmadillo.h>

#include "get_cor_mat.h"

class IOptIntra {
protected:
  const arma::mat &data_;     // The data matrix.
  const arma::mat &distSqrd_; // Square spatial distance matrix
  const arma::mat &timeSqrd_; // Square temporal distance matrix
  int numVoxel_;
  int numTimePt_;
  KernelType kernelType_; // Choice of spatial kernel
  double noiseVarianceEstimate_;
  arma::vec eblue_;
  bool verbose_;

public:
  IOptIntra(const arma::mat &data, const arma::mat &distSqrd,
            const arma::mat &timeSqrd, KernelType kernelType,
            bool verbose = true)
      : data_(data), distSqrd_(distSqrd), timeSqrd_(timeSqrd),
        kernelType_(kernelType), verbose_(verbose) {
    numVoxel_ = distSqrd.n_rows;
    numTimePt_ = timeSqrd.n_rows;
  };

  virtual double EvaluateWithGradient(const arma::mat &theta,
                                      arma::mat &gradient) = 0;

  double GetNoiseVarianceEstimate() {
    if (noiseVarianceEstimate_ < 0) {
      Rcpp::stop(
          "Noise variance estimate not computed yet. You must optimize first.");
    }
    return noiseVarianceEstimate_;
  }
  arma::vec GetEBlue() { return eblue_; }
  virtual double GetKStar() { throw std::runtime_error("Not implemented"); }
};

class OptIntra : public IOptIntra {
public:
  OptIntra(const arma::mat &data, const arma::mat &distSqrd,
           const arma::mat &timeSqrd, KernelType kernelType,
           bool verbose = true);

  double EvaluateWithGradient(const arma::mat &theta,
                              arma::mat &gradient) override;
};

class OptIntraDiagTime : public IOptIntra {
public:
  OptIntraDiagTime(const arma::mat &data, const arma::mat &distSqrd,
                   const arma::mat &timeSqrd, KernelType kernelType);

  double EvaluateWithGradient(const arma::mat &theta,
                              arma::mat &gradient) override;
};

class OptIntraNoiseless : public IOptIntra {
public:
  OptIntraNoiseless(const arma::mat &data, const arma::mat &distSqrd,
                    const arma::mat &timeSqrd, KernelType kernelType,
                    bool verbose = true);

  double EvaluateWithGradient(const arma::mat &theta,
                              arma::mat &gradient) override;
};

class OptIntraNoiselessProfiled : public IOptIntra {
  double kstar_;
public:
  OptIntraNoiselessProfiled(const arma::mat &data, const arma::mat &distSqrd,
                    const arma::mat &timeSqrd, KernelType kernelType,
                    bool verbose = true);

  double EvaluateWithGradient(const arma::mat &theta,
                              arma::mat &gradient) override;

  double GetKStar() override;
};


#endif