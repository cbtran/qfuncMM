#include "OptIntra.h"

#include <math.h>

#include "helper.h"
#include "rbf.h"

/*****************************************************************************
 Intra-regional model
*****************************************************************************/

OptIntra::OptIntra(const arma::mat &data, const arma::mat &distSqrd,
                   const arma::mat &timeSqrd, int numVoxel, int numTimePt,
                   KernelType kernelType)
    : data_(data),
      distSqrd_(distSqrd),
      timeSqrd_(timeSqrd),
      numVoxel_(numVoxel),
      numTimePt_(numTimePt),
      kernelType_(kernelType) {}

double OptIntra::EvaluateWithGradient(const arma::mat &theta,
                                      arma::mat &gradient) {
  using arma::mat;
  using arma::vec;

  double scaleSpatial = softplus(theta(0));
  double scaleTemporal = softplus(theta(1));
  double varTemporal = softplus(theta(2));
  double varTemporalNugget = softplus(theta(3));
  mat timeIdentity = arma::eye(numTimePt_, numTimePt_);
  mat timeSpaceIdentity =
      arma::eye(numTimePt_ * numVoxel_, numTimePt_ * numVoxel_);
  mat dataReshape = arma::reshape(data_, numTimePt_, numVoxel_);
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
  mat G = vInvU * arma::inv_sympd(UtVinvU);
  vec etaTildeStar = G.t() * data_;

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

  // mat dVdVarTemporal = arma::kron(covarSpatial, dTemporalDvar);
  // mat dVdVarTemporalNugget = arma::kron(covarSpatial, timeIdentity);
  // mat dVdScaleTemporal = arma::kron(covarSpatial, dTemporalDscale);
  // mat dVdScaleSpatial = arma::kron(dSpatialDscale, covarTemporal);

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
  double trace2VarTemporal =
      arma::trace(G.t() * kronecker_mmm(covarSpatial, dTemporalDvar, vInvU));
  double trace2VarTemporalNugget =
      arma::trace(G.t() * kronecker_mmm(covarSpatial, timeIdentity, vInvU));
  double trace2ScaleTemporal =
      arma::trace(G.t() * kronecker_mmm(covarSpatial, dTemporalDscale, vInvU));
  double trace2ScaleSpatial =
      arma::trace(G.t() * kronecker_mmm(dSpatialDscale, covarTemporal, vInvU));

  // Third part
  mat dataTemporalVar1 =
      vInvCentered.t() *
      kronecker_mvm(covarSpatial, dTemporalDvar, vInvCentered);
  mat dataTemporalVar2 =
      2 * vInvCentered.t() * U *
      (-G.t() * kronecker_mvm(covarSpatial, dTemporalDvar, vInvCentered));
  double dataTemporalVarNum = dataTemporalVar1(0, 0) + dataTemporalVar2(0, 0);

  mat dataTemporalVarNugget1 =
      vInvCentered.t() *
      kronecker_mvm(covarSpatial, timeIdentity, vInvCentered);
  mat dataTemporalVarNugget2 =
      2 * vInvCentered.t() * U *
      (-G.t() * kronecker_mvm(covarSpatial, timeIdentity, vInvCentered));
  double dataTemporalVarNuggetNum =
      dataTemporalVarNugget1(0, 0) + dataTemporalVarNugget2(0, 0);

  mat dataTemporalScale1 =
      vInvCentered.t() *
      kronecker_mvm(covarSpatial, dTemporalDscale, vInvCentered);
  mat dataTemporalScale2 =
      2 * vInvCentered.t() * U *
      (-G.t() * kronecker_mvm(covarSpatial, dTemporalDscale, vInvCentered));
  double dataTemporalScaleNum =
      dataTemporalScale1(0, 0) + dataTemporalScale2(0, 0);

  mat dataSpatialScale1 =
      vInvCentered.t() *
      kronecker_mvm(dSpatialDscale, covarTemporal, vInvCentered);
  mat dataSpatialScale2 =
      2 * vInvCentered.t() * U *
      (-G.t() * kronecker_mvm(dSpatialDscale, covarTemporal, vInvCentered));
  double dataSpatialScaleNum =
      dataSpatialScale1(0, 0) + dataSpatialScale2(0, 0);

  mat quadraticMat = dataCentered.t() * vInvCentered;
  double quadratic = quadraticMat(0, 0);
  gradient(0) =
      0.5 * (tracedVdScaleSpatial - trace2ScaleSpatial -
             dataSpatialScaleNum * (numVoxel_ - 1) * numTimePt_ / quadratic);
  gradient(1) =
      0.5 * (tracedVdScaleTemporal - trace2ScaleTemporal -
             dataTemporalScaleNum * (numVoxel_ - 1) * numTimePt_ / quadratic);
  gradient(2) =
      0.5 * (tracedVdVarTemporal - trace2VarTemporal -
             dataTemporalVarNum * (numVoxel_ - 1) * numTimePt_ / quadratic);
  gradient(3) = 0.5 * (tracedVdVarTemporalNugget - trace2VarTemporalNugget -
                       dataTemporalVarNuggetNum * (numVoxel_ - 1) * numTimePt_ /
                           quadratic);

  //   std::cout << "=== New ===\n";
  //   std::cout << gradient(0) << " " << gradient(1) << " " << gradient(2)
  //             << " " << arma::norm(gradient) << std::endl;
  //   std::cout << "logreml: " << logremlval << std::endl;
  // std::cout << scaleSpatial << " " << scaleTemporal << " " << varTemporal <<
  // " " << varTemporalNugget << std::endl;

  return logremlval;
}
