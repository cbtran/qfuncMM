#ifndef RBF_H
#define RBF_H
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

inline arma::mat rbf(arma::mat xSqrd_mat, double tau)
{
  return (arma::exp(-pow(tau, 2)/2 * xSqrd_mat));
}

inline arma::mat rbf_deriv(arma::mat xSqrd_mat, double tau)
{
  return arma::exp(-pow(tau, 2)/2 * xSqrd_mat) % (-tau * xSqrd_mat);
}

#endif