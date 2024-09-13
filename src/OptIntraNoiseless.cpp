#include "OptIntra.h"
#include "armadillo"
#include "helper.h"
#include "rbf.h"

/*****************************************************************************
 Intra-regional model
*****************************************************************************/

OptIntraNoiseless::OptIntraNoiseless(const arma::mat &data,
                                     const arma::mat &distSqrd,
                                     const arma::mat &timeSqrd,
                                     KernelType kernelType, bool verbose)
    : IOptIntra(data, distSqrd, timeSqrd, kernelType, verbose) {
  noiseVarianceEstimate_ = NA_REAL;
}

double OptIntraNoiseless::EvaluateWithGradient(const arma::mat &theta,
                                               arma::mat &gradient) {
  using namespace arma;

  double phi = softplus(theta(0));
  double tau = softplus(theta(1));
  double k = softplus(theta(2));
  double nugget = softplus(theta(3));
  //   if (verbose_) {
  //     Rcpp::Rcout << "==Theta: " << phi << " " << tau << " " << k << " " <<
  //     nugget
  //                 << std::endl;
  //   }
  mat timeIdentity = arma::eye(numTimePt_, numTimePt_);
  mat U = arma::repmat(timeIdentity, numVoxel_, 1);

  mat timeRbf = rbf(timeSqrd_, tau);
  mat covarSpatial = get_cor_mat(kernelType_, distSqrd_, phi);
  mat covarTemporal = k * timeRbf + nugget * timeIdentity;

  vec temporalEigval, spatialEigval;
  mat temporalEigvec, spatialEigvec;
  arma::eig_sym(temporalEigval, temporalEigvec, covarTemporal);
  arma::eig_sym(spatialEigval, spatialEigvec, covarSpatial);
  vec eigen = arma::kron(spatialEigval, temporalEigval);
  vec eigenInv = 1 / eigen;

  mat vInvU = U;
  vInvU.each_col([&spatialEigvec, &temporalEigvec, &eigenInv](arma::vec &uCol) {
    uCol = kronecker_mvm(
        spatialEigvec, temporalEigvec,
        eigenInv % kronecker_mvm(spatialEigvec.t(), temporalEigvec.t(), uCol));
  });
  mat UtVinvU = U.t() * vInvU;
  //   if (!UtVinvU.is_symmetric() || !UtVinvU.is_sympd()) {
  //     mat V = kron(covarSpatial, covarTemporal);
  //     std::cout << "V sympd: " << V.is_sympd() << std::endl;
  //     std::cout << "V rcond: " << rcond(V) << std::endl;
  //     mat Vinv = inv_sympd(V);
  //     mat utvinvu = U.t() * Vinv * U;
  //     std::cout << "utvinvu sympd: " << utvinvu.is_sympd() << std::endl;
  //     std::cout << "utvinvu rcond: " << rcond(utvinvu) << std::endl;
  //     throw std::runtime_error("UtVinvU is not symmetric");
  //   }
  mat Gt = solve(UtVinvU, vInvU.t());
  eblue_ = Gt * data_;
  vec dataCentered = data_ - U * eblue_;
  vec vInvCentered =
      kronecker_mvm(spatialEigvec, temporalEigvec,
                    eigenInv % kronecker_mvm(spatialEigvec.t(),
                                             temporalEigvec.t(), dataCentered));
  double logreml1 = arma::accu(arma::log(eigen));
  double logreml2;
  if (!arma::log_det_sympd(logreml2, UtVinvU)) {
    logreml2 = arma::log_det(UtVinvU).real();
  }
  double qdr = as_scalar(dataCentered.t() * vInvCentered);

  double fixed =
      (numVoxel_ - 1) * numTimePt_ * log((numVoxel_ - 1) * numTimePt_) -
      (numVoxel_ - 1) * numTimePt_;

  double logremlval = 0.5 * (logreml1 + logreml2 + qdr + fixed);
  if (std::isnan(logremlval)) {
    throw std::runtime_error("logremlval is nan");
  }

  // Gradients
  const mat &dBdk = timeRbf;
  mat dBdtau = k * rbf_deriv(timeSqrd_, tau);
  mat dCdphi = get_cor_mat_deriv(kernelType_, distSqrd_, phi);

  // The calls to diagvec are delayed
  // so this does not compute the entire matrix product to just get the
  // diagonal.
  mat temporalVarEig = temporalEigvec.t() * dBdk * temporalEigvec;
  mat temporalScaleEig = temporalEigvec.t() * dBdtau * temporalEigvec;
  mat temporalCovarEig = temporalEigvec.t() * covarTemporal * temporalEigvec;
  mat spatialCovarEig = spatialEigvec.t() * covarSpatial * spatialEigvec;
  mat spatialScaleEig = spatialEigvec.t() * dCdphi * spatialEigvec;

  mat eigvalOuterInv = arma::reshape(eigenInv, numTimePt_, numVoxel_);
  double trace_dVdk =
      arma::dot(arma::diagvec(spatialCovarEig),
                eigvalOuterInv.t() * arma::diagvec(temporalVarEig));
  double trace_dVdnugget =
      arma::dot(arma::diagvec(spatialCovarEig), arma::sum(eigvalOuterInv, 0));
  double trace_dVdtau =
      arma::dot(arma::diagvec(spatialCovarEig),
                eigvalOuterInv.t() * arma::diagvec(temporalScaleEig));
  double trace_dVdphi =
      arma::dot(arma::diagvec(spatialScaleEig),
                eigvalOuterInv.t() * arma::diagvec(temporalCovarEig));

  // Second part
  cube spatialColumnsVec(numTimePt_, numVoxel_, vInvU.n_cols);
  for (int i = 0; i < (int)vInvU.n_cols; i++) {
    spatialColumnsVec.slice(i) =
        arma::reshape(vInvU.col(i), numTimePt_, numVoxel_) * covarSpatial.t();
  }

  cube spatialDvar = dBdk * spatialColumnsVec.each_slice();
  double trace2VarTemporal = arma::trace(
      Gt * mat(spatialDvar.memptr(), spatialDvar.n_rows * spatialDvar.n_cols,
               spatialDvar.n_slices, false));

  cube spatialDscale = dBdtau * spatialColumnsVec.each_slice();
  double trace2ScaleTemporal =
      arma::trace(Gt * mat(spatialDscale.memptr(),
                           spatialDscale.n_rows * spatialDscale.n_cols,
                           spatialDscale.n_slices, false));

  double trace2VarTemporalNugget =
      arma::trace(Gt * mat(spatialColumnsVec.memptr(),
                           spatialColumnsVec.n_rows * spatialColumnsVec.n_cols,
                           spatialColumnsVec.n_slices, false));

  vInvU.each_col([&dCdphi, &covarTemporal](arma::vec &uCol) {
    uCol = kronecker_mvm(dCdphi, covarTemporal, uCol);
  });
  double trace2ScaleSpatial = arma::trace(Gt * vInvU);

  // Third part
  mat dataTemporalVar1 =
      vInvCentered.t() * kronecker_mvm(covarSpatial, dBdk, vInvCentered);
  mat dataTemporalVar2 =
      2 * vInvCentered.t() * U *
      (-Gt * kronecker_mvm(covarSpatial, dBdk, vInvCentered));
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
      vInvCentered.t() * kronecker_mvm(covarSpatial, dBdtau, vInvCentered);
  mat dataTemporalScale2 =
      2 * vInvCentered.t() * U *
      (-Gt * kronecker_mvm(covarSpatial, dBdtau, vInvCentered));
  double dataTemporalScaleNum =
      dataTemporalScale1(0, 0) + dataTemporalScale2(0, 0);

  double dataSpatialScale1 = as_scalar(
      vInvCentered.t() * kronecker_mvm(dCdphi, covarTemporal, vInvCentered));
  double dataSpatialScale2 =
      as_scalar(2 * vInvCentered.t() * U *
                (-Gt * kronecker_mvm(dCdphi, covarTemporal, vInvCentered)));
  double dataSpatialScaleNum = dataSpatialScale1 + dataSpatialScale2;

  gradient(0) = 0.5 *
                (trace_dVdphi - trace2ScaleSpatial - dataSpatialScaleNum) *
                logistic(phi);
  gradient(1) = 0.5 *
                (trace_dVdtau - trace2ScaleTemporal - dataTemporalScaleNum) *
                logistic(tau);
  gradient(2) =
      0.5 * (trace_dVdk - trace2VarTemporal - dataTemporalVarNum) * logistic(k);
  gradient(3) =
      0.5 *
      (trace_dVdnugget - trace2VarTemporalNugget - dataTemporalVarNuggetNum) *
      logistic(nugget);
  return logremlval;
}
