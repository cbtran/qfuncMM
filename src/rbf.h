#ifndef RBF_H
#define RBF_H
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

inline double rbf(double xSqrd, double tau)
{
    return exp(-pow(tau, 2)/2 * xSqrd);
}

inline arma::vec rbf(arma::vec xSqrd_vec, double tau)
{
  return (arma::exp(-pow(tau, 2)/2 * xSqrd_vec));
}

inline arma::mat rbf(arma::mat xSqrd_mat, double tau)
{
  return (arma::exp(-pow(tau, 2)/2 * xSqrd_mat));
}


// Derivatives
inline arma::vec rbf_deriv(arma::vec xSqrd_vec, double tau)
{
  return arma::exp(-pow(tau, 2)/2 * xSqrd_vec) % (-tau * xSqrd_vec);
}

inline arma::mat rbf_deriv(arma::mat xSqrd_mat, double tau)
{
  return arma::exp(-pow(tau, 2)/2 * xSqrd_mat) % (-tau * xSqrd_mat);
}

#endif