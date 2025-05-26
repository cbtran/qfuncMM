#include "OptIntra.h"
#include "helper.h"
#include "rbf.h"

/*****************************************************************************
 Intra-regional model
*****************************************************************************/

OptIntra::OptIntra(const arma::mat &data, const arma::mat &distSqrd,
                   const arma::mat &timeSqrd, KernelType kernelType,
                   bool verbose)
    : IOptIntra(data, distSqrd, timeSqrd, kernelType, verbose) {}

double OptIntra::EvaluateWithGradient(const arma::mat &theta,
                                      arma::mat &gradient) {
  using namespace arma;

  double scaleSpatial = softplus(theta(0));
  double scaleTemporal = softplus(theta(1));
  double varTemporal = softplus(theta(2));
  double varTemporalNugget = softplus(theta(3));
  //   if (verbose_) {
  //     Rcpp::Rcout << "======\nTheta: " << scaleSpatial << " " <<
  //     scaleTemporal
  //                 << " " << varTemporal << " " << varTemporalNugget <<
  //                 std::endl;
  //   }
  mat timeIdentity = arma::eye(numTimePt_, numTimePt_);
  mat U = arma::repmat(timeIdentity, numVoxel_, 1);

  mat timeRbf = rbf(timeSqrd_, scaleTemporal);
  mat covarSpatial = get_cor_mat(kernelType_, distSqrd_, scaleSpatial);
  mat covarTemporal = varTemporal * timeRbf + varTemporalNugget * timeIdentity;

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
  double logreml3mat = (numVoxel_ - 1) * numTimePt_ * log(qdr);
  double logreml3 = logreml3mat;

  double logreml1 = arma::accu(arma::log(eigen));
  double logreml2;
  if (!arma::log_det_sympd(logreml2, UtVinvU)) {
    logreml2 = arma::log_det(UtVinvU).real();
  }

  double logremlval = 0.5 * (logreml1 + logreml2 + logreml3);
  if (std::isnan(logremlval)) {
    throw std::runtime_error("logremlval is nan");
  }

  // Gradients
  const mat &dB_dk_gamma = timeRbf;
  mat dB_dtau_gamma = varTemporal * rbf_deriv(timeSqrd_, scaleTemporal);
  mat dC_dphi_gamma = get_cor_mat_deriv(kernelType_, distSqrd_, scaleSpatial);

  // The calls to diagvec are delayed
  // so this does not compute the entire matrix product to just get the
  // diagonal.
  mat temporalVarEig = temporalEigvec.t() * dB_dk_gamma * temporalEigvec;
  mat temporalScaleEig = temporalEigvec.t() * dB_dtau_gamma * temporalEigvec;
  mat temporalCovarEig = temporalEigvec.t() * covarTemporal * temporalEigvec;
  mat spatialCovarEig = spatialEigvec.t() * covarSpatial * spatialEigvec;
  mat spatialScaleEig = spatialEigvec.t() * dC_dphi_gamma * spatialEigvec;

  mat eigvalOuterInv = arma::reshape(eigenInv, numTimePt_, numVoxel_);
  double tracedVdVarTemporal =
      arma::dot(arma::diagvec(spatialCovarEig),
                eigvalOuterInv.t() * arma::diagvec(temporalVarEig));
  double tracedVdVarTemporalNugget =
      arma::dot(arma::diagvec(spatialCovarEig), arma::sum(eigvalOuterInv, 0));
  double tracedVdScaleTemporal =
      arma::dot(arma::diagvec(spatialCovarEig),
                eigvalOuterInv.t() * arma::diagvec(temporalScaleEig));
  double tracedVdScaleSpatial =
      arma::dot(arma::diagvec(spatialScaleEig),
                eigvalOuterInv.t() * arma::diagvec(temporalCovarEig));

  // Second part
  cube spatialColumnsVec(numTimePt_, numVoxel_, vInvU.n_cols);
  for (int i = 0; i < (int)vInvU.n_cols; i++) {
    spatialColumnsVec.slice(i) =
        arma::reshape(vInvU.col(i), numTimePt_, numVoxel_) * covarSpatial.t();
  }

  cube spatialDvar = dB_dk_gamma * spatialColumnsVec.each_slice();
  double trace2VarTemporal = arma::trace(
      Gt * mat(spatialDvar.memptr(), spatialDvar.n_rows * spatialDvar.n_cols,
               spatialDvar.n_slices, false));

  cube spatialDscale = dB_dtau_gamma * spatialColumnsVec.each_slice();
  double trace2ScaleTemporal =
      arma::trace(Gt * mat(spatialDscale.memptr(),
                           spatialDscale.n_rows * spatialDscale.n_cols,
                           spatialDscale.n_slices, false));

  double trace2VarTemporalNugget =
      arma::trace(Gt * mat(spatialColumnsVec.memptr(),
                           spatialColumnsVec.n_rows * spatialColumnsVec.n_cols,
                           spatialColumnsVec.n_slices, false));

  vInvU.each_col([&dC_dphi_gamma, &covarTemporal](arma::vec &uCol) {
    uCol = kronecker_mvm(dC_dphi_gamma, covarTemporal, uCol);
  });
  double trace2ScaleSpatial = arma::trace(Gt * vInvU);

  // Third part
  mat dataTemporalVar1 =
      vInvCentered.t() * kronecker_mvm(covarSpatial, dB_dk_gamma, vInvCentered);
  mat dataTemporalVar2 =
      2 * vInvCentered.t() * U *
      (-Gt * kronecker_mvm(covarSpatial, dB_dk_gamma, vInvCentered));
  double dataTemporalVarNum = dataTemporalVar1(0, 0) + dataTemporalVar2(0, 0);

  mat dataTemporalVarNugget1 =
      vInvCentered.t() *
      kronecker_mvm(covarSpatial, timeIdentity, vInvCentered);
  mat dataTemporalVarNugget2 =
      2 * vInvCentered.t() * U *
      (-Gt * kronecker_mvm(covarSpatial, timeIdentity, vInvCentered));
  double dataTemporalVarNuggetNum =
      dataTemporalVarNugget1(0, 0) + dataTemporalVarNugget2(0, 0);

  mat dataTemporalScale1 =
      vInvCentered.t() *
      kronecker_mvm(covarSpatial, dB_dtau_gamma, vInvCentered);
  mat dataTemporalScale2 =
      2 * vInvCentered.t() * U *
      (-Gt * kronecker_mvm(covarSpatial, dB_dtau_gamma, vInvCentered));
  double dataTemporalScaleNum =
      dataTemporalScale1(0, 0) + dataTemporalScale2(0, 0);

  double dataSpatialScale1 =
      as_scalar(vInvCentered.t() *
                kronecker_mvm(dC_dphi_gamma, covarTemporal, vInvCentered));
  double dataSpatialScale2 = as_scalar(
      2 * vInvCentered.t() * U *
      (-Gt * kronecker_mvm(dC_dphi_gamma, covarTemporal, vInvCentered)));
  double dataSpatialScaleNum = dataSpatialScale1 + dataSpatialScale2;

  noiseVarianceEstimate_ = qdr / ((numVoxel_ - 1) * numTimePt_);
  gradient(0) = 0.5 *
                (tracedVdScaleSpatial - trace2ScaleSpatial -
                 dataSpatialScaleNum / noiseVarianceEstimate_) *
                logistic(scaleSpatial);
  gradient(1) = 0.5 *
                (tracedVdScaleTemporal - trace2ScaleTemporal -
                 dataTemporalScaleNum / noiseVarianceEstimate_) *
                logistic(scaleTemporal);
  gradient(2) = 0.5 *
                (tracedVdVarTemporal - trace2VarTemporal -
                 dataTemporalVarNum / noiseVarianceEstimate_) *
                logistic(varTemporal);
  gradient(3) = 0.5 *
                (tracedVdVarTemporalNugget - trace2VarTemporalNugget -
                 dataTemporalVarNuggetNum / noiseVarianceEstimate_) *
                logistic(varTemporalNugget);

  //   if (verbose_) {
  //     Rcpp::Rcout << "Gradient: ";
  //     Rcpp::Rcout << gradient(0) << " " << gradient(1) << " " << gradient(2)
  //                 << " " << gradient(3) << " " << arma::norm(gradient)
  //                 << std::endl;
  //     Rcpp::Rcout << "logreml: " << logremlval << std::endl;
  //   }

  return logremlval;
}
