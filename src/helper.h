#ifndef HELPER_H
#define HELPER_H
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' @title Kronecker matrix-vector product (Steeb & Hardy, 2011)
//' @description \deqn{(A \otimes B)v = \text{vectorise}(BVA^\top)}
//' @param A Left matrix
//' @param B Right matrix
//' @param v vector
//' @return Kronecker matrix-vector product
//' @noRd
arma::vec kronecker_mvm(const arma::mat &A, const arma::mat &B,
                        const arma::vec &v, bool eye_B = false);

// Kronecker matrix-matrix product using kronecker structure
arma::mat kronecker_mmm(const arma::mat &A, const arma::mat &B,
                        const arma::mat &C, bool eye_B = false);

arma::vec kronecker_ovm(const arma::mat &B, const arma::vec &v,
                        arma::uword n_row_A, double scalar = 1);

// Kronecker matrix product (A \otimes B) * C where A is a matrix of ones times
// a scalar. We can save computation time by computing row sums of C.
arma::mat kronecker_omm(const arma::mat &B, const arma::mat &C,
                        arma::uword n_row, double scalar = 1);

// Get squared distance matrix given the voxel coordinates for a region
arma::mat squared_distance(arma::mat coords);

// Forward-backward to solve linear system \eqn{R^{-1} * B = Z} with Cholesky
// decomposition
// @param R_chol Cholesky decomposition of Matrix R
// @param B Matrix to solve against
// @return solution Z of \eqn{z = R^{-1}b}
arma::mat R_inv_B(const arma::mat &R_chol, const arma::mat &B);

// Compute softplus
// @param x scalar
// @return compute sofplus function log(1 + exp(x))
double softplus(double x);

// Compute softplus
// @param x scalar or matrix
// @return compute sofplus function log(1 + exp(x)) element-wise
arma::mat softplus(arma::mat xMat);

// Compute softminus
// @param x scalar
// @return compute sofminus function log(exp(x) - 1)
double softminus(double x);

// Compute softminus
// @param x scalar or matrix
// @return compute sofminus function log(exp(x) - 1) element-wise
arma::mat softminus(arma::mat xMat);

// Compute logistic:
// @param scalar x
// @return compute logistic function 1 / (1+exp(-x))
double logistic(double x);

// Compute sigmoid inverse
// @param y: unconstrained scalar
// @param lower: scalar lower bound
// @param upper: scalar upper bound
// @return constrained scalar x with bound [lower, upper]
double sigmoid_inv(double y, double lower, double upper);

// Compute derivative dx/dy with unconstrained y and constrained and x with
// bound [lower, upper]
// @param unconstrained scalar y
// @param lower: scalar lower bound
// @param upper: scalar upper bound
// @return derivative dx/dy with constraint x with bound [lower, upper]
double sigmoid_inv_derivative(double y, double lower, double upper);
#endif
