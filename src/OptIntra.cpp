#include "OptIntra.h"
#include "helper.h"
#include "rbf.h"

/*****************************************************************************
 Intra-regional model
*****************************************************************************/

OptIntra::OptIntra(const arma::mat &data, const arma::mat &distSqrd,
                   const arma::mat &timeSqrd, KernelType kernelType)
    : data_(data),
      distSqrd_(distSqrd),
      timeSqrd_(timeSqrd),
      kernelType_(kernelType),
      noiseVarianceEstimate_(-1)
{
    numVoxel_ = distSqrd.n_rows;
    numTimePt_ = timeSqrd.n_rows;
}

double OptIntra::EvaluateWithGradient(const arma::mat &theta,
                                      arma::mat &gradient) {
  using namespace arma;

  double scaleSpatial = softplus(theta(0));
  double scaleTemporal = softplus(theta(1));
  double varTemporal = softplus(theta(2));
  double varTemporalNugget = softplus(theta(3));
//   std::cout << "=== Theta ===\n"
//             << scaleSpatial << " " << scaleTemporal << " " << varTemporal << " "
//             << varTemporalNugget << std::endl;
  mat timeIdentity = arma::eye(numTimePt_, numTimePt_);
  mat timeSpaceIdentity =
      arma::eye(numTimePt_ * numVoxel_, numTimePt_ * numVoxel_);
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
  vec etaTildeStar = Gt * data_;

  vec dataCentered = data_ - U * etaTildeStar;
  vec vInvCentered =
      kronecker_mvm(spatialEigvec, temporalEigvec,
                    eigenInv % kronecker_mvm(spatialEigvec.t(),
                                             temporalEigvec.t(), dataCentered));
  vec qdr = dataCentered.t() * vInvCentered;
  mat logreml3mat = (numVoxel_ - 1) * numTimePt_ * log(qdr);
  double logreml3 = logreml3mat(0, 0);

  double logreml1 = arma::accu(arma::log(eigen));
  double logreml2 = arma::log_det_sympd(UtVinvU);

  double logremlval = 0.5 * (logreml1 + logreml2 + logreml3);

  // Gradients
  const mat &dTemporalDvar = timeRbf;
  mat dTemporalDscale = varTemporal * rbf_deriv(timeSqrd_, scaleTemporal);
  mat dSpatialDscale = get_cor_mat_deriv(kernelType_, distSqrd_, scaleSpatial);

  // The calls to diagvec are delayed
  // so this does not compute the entire matrix product to just get the
  // diagonal.
  mat temporalVarEig = temporalEigvec.t() * dTemporalDvar * temporalEigvec;
  mat temporalScaleEig = temporalEigvec.t() * dTemporalDscale * temporalEigvec;
  mat temporalCovarEig = temporalEigvec.t() * covarTemporal * temporalEigvec;
  mat spatialCovarEig = spatialEigvec.t() * covarSpatial * spatialEigvec;
  mat spatialScaleEig = spatialEigvec.t() * dSpatialDscale * spatialEigvec;

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
  for (int i = 0; i < (int) vInvU.n_cols; i++)
  {
    spatialColumnsVec.slice(i) =
      arma::reshape(vInvU.col(i), numTimePt_, numVoxel_) * covarSpatial.t();
  }

  cube spatialDvar = dTemporalDvar * spatialColumnsVec.each_slice();
  double trace2VarTemporal =
    arma::trace(Gt *
      mat(spatialDvar.memptr(), spatialDvar.n_rows * spatialDvar.n_cols, spatialDvar.n_slices, false));

  cube spatialDscale = dTemporalDscale * spatialColumnsVec.each_slice();
  double trace2ScaleTemporal =
    arma::trace(Gt *
      mat(spatialDscale.memptr(), spatialDscale.n_rows * spatialDscale.n_cols, spatialDscale.n_slices, false));

  double trace2VarTemporalNugget =
    arma::trace(Gt *
      mat(spatialColumnsVec.memptr(), spatialColumnsVec.n_rows * spatialColumnsVec.n_cols, spatialColumnsVec.n_slices, false));

  vInvU.each_col([&dSpatialDscale, &covarTemporal](arma::vec &uCol) {
    uCol = kronecker_mvm(dSpatialDscale, covarTemporal, uCol);
  });
  double trace2ScaleSpatial = arma::trace(Gt * vInvU);

  // Third part
  mat dataTemporalVar1 =
      vInvCentered.t() *
      kronecker_mvm(covarSpatial, dTemporalDvar, vInvCentered);
  mat dataTemporalVar2 =
      2 * vInvCentered.t() * U *
      (-Gt * kronecker_mvm(covarSpatial, dTemporalDvar, vInvCentered));
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
      kronecker_mvm(covarSpatial, dTemporalDscale, vInvCentered);
  mat dataTemporalScale2 =
      2 * vInvCentered.t() * U *
      (-Gt * kronecker_mvm(covarSpatial, dTemporalDscale, vInvCentered));
  double dataTemporalScaleNum =
      dataTemporalScale1(0, 0) + dataTemporalScale2(0, 0);

  mat dataSpatialScale1 =
      vInvCentered.t() *
      kronecker_mvm(dSpatialDscale, covarTemporal, vInvCentered);
  mat dataSpatialScale2 =
      2 * vInvCentered.t() * U *
      (-Gt * kronecker_mvm(dSpatialDscale, covarTemporal, vInvCentered));
  double dataSpatialScaleNum =
      dataSpatialScale1(0, 0) + dataSpatialScale2(0, 0);

  mat quadraticMat = dataCentered.t() * vInvCentered;
  noiseVarianceEstimate_ = quadraticMat(0, 0) / ((numVoxel_ - 1) * numTimePt_);
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

//   std::cout << "=== Gradient ===\n";
//   std::cout << gradient(0) << " " << gradient(1) << " " << gradient(2) << " "
//             << gradient(3) << " " << arma::norm(gradient) << std::endl;
//   std::cout << "logreml: " << logremlval << std::endl;
  // std::cout << scaleSpatial << " " << scaleTemporal << " " << varTemporal <<
  // " " << varTemporalNugget << std::endl;

  return logremlval;
}

double OptIntra::GetNoiseVarianceEstimate() {
  if (noiseVarianceEstimate_ < 0) {
    Rcpp::stop(
        "Noise variance estimate not computed yet. You must optimize first.");
  }
  return noiseVarianceEstimate_;
}
