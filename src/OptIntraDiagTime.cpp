#include "OptIntra.h"
#include "helper.h"

/*****************************************************************************
 Intra-regional model
*****************************************************************************/

OptIntraDiagTime::OptIntraDiagTime(const arma::mat &data,
                                   const arma::mat &distSqrd,
                                   const arma::mat &timeSqrd,
                                   KernelType kernelType)
    : IOptIntra(data, distSqrd, timeSqrd, kernelType) {}

double OptIntraDiagTime::EvaluateWithGradient(const arma::mat &theta,
                                              arma::mat &gradient) {
  using namespace arma;

  double scaleSpatial = softplus(theta(0));
  //   double scaleTemporal = 1;
  //   double varTemporal = 0;
  double varTemporalNugget = softplus(theta(1));
  Rcpp::Rcout << "======\nTheta: " << scaleSpatial << " " << varTemporalNugget
              << std::endl;
  mat timeIdentity = arma::eye(numTimePt_, numTimePt_);
  mat U = arma::repmat(timeIdentity, numVoxel_, 1);

  mat covarSpatial = get_cor_mat(kernelType_, distSqrd_, scaleSpatial);
  mat covarTemporal = varTemporalNugget * timeIdentity;

  vec temporalEigval, spatialEigval;
  mat temporalEigvec, spatialEigvec;
  arma::eig_sym(temporalEigval, temporalEigvec, covarTemporal);
  arma::eig_sym(spatialEigval, spatialEigvec, covarSpatial);
  vec eigen = arma::kron(spatialEigval, temporalEigval) + 1;
  vec eigenInv = 1 / eigen;

  mat vInvU = U;
  vInvU.each_col([&spatialEigvec, &temporalEigvec, &eigenInv](arma::vec &uCol) {
    uCol = kronecker_mvm(
        spatialEigvec, temporalEigvec,
        eigenInv % kronecker_mvm(spatialEigvec.t(), temporalEigvec.t(), uCol));
  });
  mat UtVinvU = U.t() * vInvU;
  mat Gt = arma::inv_sympd(UtVinvU) * vInvU.t();
  eblue_ = Gt * data_;
  vec dataCentered = data_ - U * eblue_;
  vec vInvCentered =
      kronecker_mvm(spatialEigvec, temporalEigvec,
                    eigenInv % kronecker_mvm(spatialEigvec.t(),
                                             temporalEigvec.t(), dataCentered));
  double qdr = as_scalar(dataCentered.t() * vInvCentered);
  double logreml3 = (numVoxel_ - 1) * numTimePt_ * log(qdr);

  double logreml1 = arma::accu(arma::log(eigen));
  double logreml2 = arma::log_det_sympd(UtVinvU);

  double logremlval = 0.5 * (logreml1 + logreml2 + logreml3);

  // Gradients
  mat dSpatialDscale = get_cor_mat_deriv(kernelType_, distSqrd_, scaleSpatial);

  // The calls to diagvec are delayed
  // so this does not compute the entire matrix product to just get the
  // diagonal.
  mat temporalCovarEig = temporalEigvec.t() * covarTemporal * temporalEigvec;
  mat spatialCovarEig = spatialEigvec.t() * covarSpatial * spatialEigvec;
  mat spatialScaleEig = spatialEigvec.t() * dSpatialDscale * spatialEigvec;

  mat eigvalOuterInv = arma::reshape(eigenInv, numTimePt_, numVoxel_);
  double tracedVdVarTemporalNugget =
      arma::dot(arma::diagvec(spatialCovarEig), arma::sum(eigvalOuterInv, 0));
  double tracedVdScaleSpatial =
      arma::dot(arma::diagvec(spatialScaleEig),
                eigvalOuterInv.t() * arma::diagvec(temporalCovarEig));

  // Second part
  cube spatialColumnsVec(numTimePt_, numVoxel_, vInvU.n_cols);
  for (int i = 0; i < (int)vInvU.n_cols; i++) {
    spatialColumnsVec.slice(i) =
        arma::reshape(vInvU.col(i), numTimePt_, numVoxel_) * covarSpatial.t();
  }

  double trace2VarTemporalNugget =
      arma::trace(Gt * mat(spatialColumnsVec.memptr(),
                           spatialColumnsVec.n_rows * spatialColumnsVec.n_cols,
                           spatialColumnsVec.n_slices, false));

  vInvU.each_col([&dSpatialDscale, &covarTemporal](arma::vec &uCol) {
    uCol = kronecker_mvm(dSpatialDscale, covarTemporal, uCol);
  });
  double trace2ScaleSpatial = arma::trace(Gt * vInvU);

  // Third part
  mat dataTemporalVarNugget1 =
      vInvCentered.t() *
      kronecker_mvm(covarSpatial, timeIdentity, vInvCentered);
  mat dataTemporalVarNugget2 =
      2 * vInvCentered.t() * U *
      (-Gt * kronecker_mvm(covarSpatial, timeIdentity, vInvCentered));
  double dataTemporalVarNuggetNum =
      dataTemporalVarNugget1(0, 0) + dataTemporalVarNugget2(0, 0);

  double dataSpatialScale1 =
      as_scalar(vInvCentered.t() *
                kronecker_mvm(dSpatialDscale, covarTemporal, vInvCentered));
  double dataSpatialScale2 = as_scalar(
      2 * vInvCentered.t() * U *
      (-Gt * kronecker_mvm(dSpatialDscale, covarTemporal, vInvCentered)));
  double dataSpatialScaleNum = dataSpatialScale1 + dataSpatialScale2;

  noiseVarianceEstimate_ = qdr / ((numVoxel_ - 1) * numTimePt_);
  gradient(0) = 0.5 *
                (tracedVdScaleSpatial - trace2ScaleSpatial -
                 dataSpatialScaleNum / noiseVarianceEstimate_) *
                logistic(scaleSpatial);
  gradient(1) = 0.5 *
                (tracedVdVarTemporalNugget - trace2VarTemporalNugget -
                 dataTemporalVarNuggetNum / noiseVarianceEstimate_) *
                logistic(varTemporalNugget);

  Rcpp::Rcout << "Gradient: ";
  Rcpp::Rcout << gradient(0) << " " << gradient(1) << " "
              << arma::norm(gradient) << std::endl;
  Rcpp::Rcout << "logreml: " << logremlval << std::endl;

  return logremlval;
}
