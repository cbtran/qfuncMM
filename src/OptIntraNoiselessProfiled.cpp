#include "OptIntra.h"
#include "armadillo"
#include "helper.h"
#include "rbf.h"

/*****************************************************************************
 Intra-regional model
*****************************************************************************/

OptIntraNoiselessProfiled::OptIntraNoiselessProfiled(const arma::mat &data,
                                     const arma::mat &distSqrd,
                                     const arma::mat &timeSqrd,
                                     KernelType kernelType, bool verbose)
    : IOptIntra(data, distSqrd, timeSqrd, kernelType, verbose) {
  noiseVarianceEstimate_ = 1;
}

double OptIntraNoiselessProfiled::EvaluateWithGradient(const arma::mat &theta,
                                               arma::mat &gradient) {
  using namespace arma;

  double phi = softplus(theta(0));
  double tau = softplus(theta(1));
  double nugget_over_k = softplus(theta(2));
  // double k = softplus(theta(2));
  // double nugget = softplus(theta(3));
  //   if (verbose_) {
  //     Rcpp::Rcout << "==Theta: " << phi << " " << tau << " " << k << " " <<
  //     nugget
  //                 << std::endl;
  //   }
  mat timeIdentity = arma::eye(numTimePt_, numTimePt_);
  mat U = arma::repmat(timeIdentity, numVoxel_, 1);

  mat timeRbf = rbf(timeSqrd_, tau);
  mat C = get_cor_mat(kernelType_, distSqrd_, phi);
  mat B = timeRbf + nugget_over_k * timeIdentity;

  vec temporalEigval, spatialEigval;
  mat temporalEigvec, spatialEigvec;
  arma::eig_sym(temporalEigval, temporalEigvec, B);
  arma::eig_sym(spatialEigval, spatialEigvec, C);
  vec eigen = arma::kron(spatialEigval, temporalEigval);
  vec eigenInv = 1 / eigen;

  mat vInvU = U;
  vInvU.each_col([&spatialEigvec, &temporalEigvec, &eigenInv](arma::vec &uCol) {
    uCol = kronecker_mvm(
        spatialEigvec, temporalEigvec,
        eigenInv % kronecker_mvm(spatialEigvec.t(), temporalEigvec.t(), uCol));
  });
  mat UtVinvU = U.t() * vInvU;
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
  double logreml3 = (numVoxel_ - 1) * numTimePt_ * log(qdr);
  double logremlval = 0.5 * (logreml1 + logreml2 + logreml3);
  if (std::isnan(logremlval)) {
    throw std::runtime_error("logremlval is nan");
  }


  // Gradients
  kstar_ = qdr / ((numVoxel_ - 1) * numTimePt_);
  // const mat &dBdk = timeRbf;
  mat dBdtau = rbf_deriv(timeSqrd_, tau);
  mat dCdphi = get_cor_mat_deriv(kernelType_, distSqrd_, phi);

  // The calls to diagvec are delayed
  // so this does not compute the entire matrix product to just get the
  // diagonal.
  // mat temporalVarEig = temporalEigvec.t() * dBdk * temporalEigvec;
  mat temporalScaleEig = temporalEigvec.t() * dBdtau * temporalEigvec;
  mat temporalCovarEig = temporalEigvec.t() * B * temporalEigvec;
  mat spatialCovarEig = spatialEigvec.t() * C * spatialEigvec;
  mat spatialScaleEig = spatialEigvec.t() * dCdphi * spatialEigvec;

  mat eigvalOuterInv = arma::reshape(eigenInv, numTimePt_, numVoxel_);
  // double trace_dVdk =
  //     arma::dot(arma::diagvec(spatialCovarEig),
  //               eigvalOuterInv.t() * arma::diagvec(temporalVarEig));
  double trace_dVdnugget_over_k =
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
        arma::reshape(vInvU.col(i), numTimePt_, numVoxel_) * C.t();
  }

  // cube spatialDvar = dBdk * spatialColumnsVec.each_slice();
  // double trace2VarTemporal = arma::trace(
  //     Gt * mat(spatialDvar.memptr(), spatialDvar.n_rows * spatialDvar.n_cols,
  //              spatialDvar.n_slices, false));

  cube spatialDscale = dBdtau * spatialColumnsVec.each_slice();
  double trace2tau =
      arma::trace(Gt * mat(spatialDscale.memptr(),
                           spatialDscale.n_rows * spatialDscale.n_cols,
                           spatialDscale.n_slices, false));

  double trace2nugget_over_k =
      arma::trace(Gt * mat(spatialColumnsVec.memptr(),
                           spatialColumnsVec.n_rows * spatialColumnsVec.n_cols,
                           spatialColumnsVec.n_slices, false));

  vInvU.each_col([&dCdphi, &B](arma::vec &uCol) {
    uCol = kronecker_mvm(dCdphi, B, uCol);
  });
  double trace2phi = arma::trace(Gt * vInvU);

  // Third part
  // mat dataTemporalVar1 =
  //     vInvCentered.t() * kronecker_mvm(C, dBdk, vInvCentered);
  // mat dataTemporalVar2 =
  //     2 * vInvCentered.t() * U *
  //     (-Gt * kronecker_mvm(C, dBdk, vInvCentered));
  // double dataTemporalVarNum = dataTemporalVar1(0, 0) + dataTemporalVar2(0, 0);

  double data_nugget_over_k1 =
    as_scalar(
        vInvCentered.t() *
        kronecker_mvm(C, timeIdentity, vInvCentered)
    );
  double data_nugget_over_k2 =
    as_scalar(
      2 * vInvCentered.t() * U *
      (-Gt * kronecker_mvm(C, timeIdentity, vInvCentered))
    );
  double qdr_nugget_over_k = data_nugget_over_k1 + data_nugget_over_k2;

  double data_tau1 =
      as_scalar(vInvCentered.t() * kronecker_mvm(C, dBdtau, vInvCentered));
  double data_tau2 =
      as_scalar(2 * vInvCentered.t() * U *
      (-Gt * kronecker_mvm(C, dBdtau, vInvCentered)));
  double qdr_tau = data_tau1 + data_tau2;

  double data_phi1 = as_scalar(
      vInvCentered.t() * kronecker_mvm(dCdphi, B, vInvCentered));
  double data_phi2 =
      as_scalar(2 * vInvCentered.t() * U *
                (-Gt * kronecker_mvm(dCdphi, B, vInvCentered)));
  double qdr_phi = data_phi1 + data_phi2;

  gradient(0) = 0.5 * (trace_dVdphi - trace2phi - qdr_phi) * logistic(phi) / kstar_;
  gradient(1) = 0.5 * (trace_dVdtau - trace2tau - qdr_tau) * logistic(tau) / kstar_;
  // gradient(2) =
  //     0.5 * (trace_dVdk - trace2VarTemporal - dataTemporalVarNum) * logistic(k);
  gradient(2) = 0.5 *
      (trace_dVdnugget_over_k - trace2nugget_over_k - qdr_nugget_over_k) *
      logistic(nugget_over_k) / kstar_;
  return logremlval;
}

double OptIntraNoiselessProfiled::GetKStar() {
  return kstar_;
};
