#ifndef HELPER_H
#define HELPER_H
#include <RcppArmadillo.h>
#include "get_cor_mat.h"
// [[Rcpp::depends(RcppArmadillo)]]

//' @title Kronecker matrix-vector product (Steeb & Hardy, 2011)
//' @description \deqn{(A \otimes B)v = \text{vectorise}(BVA^\top)}
//' @param A Left matrix
//' @param B Right matrix
//' @param v vector
//' @return Kronecker matrix-vector product
//' @references Willi-Hans Steeb and Yorick Hardy. Matrix calculus and Kronecker product: a practical approach to linear and multilinear algebra. World Scientific Publishing Company, 2011.
//' @export
// [[Rcpp::export()]]
arma::vec kronecker_mvm(
  const arma::mat& A, const arma::mat& B, const arma::vec& v);

// Kronecker matrix-matrix product using kronecker structure
arma::mat kronecker_mmm(
  const arma::mat& A, const arma::mat& B, const arma::mat& C);

//' Get distance squared matrix for simulated data
//' @param L Number of voxels 
//' @param sideLength side length of that region
//' @param voxelID ID of sampled voxels
//' @return Matrix of squared distance between voxels for that region
//' @export
// [[Rcpp::export()]]
arma::mat get_dist_sqrd_mat(int L, int sideLength, Rcpp::NumericVector voxelID);


//' Get distance squared matrix for real data
//' @param coord_mat Matrix with each column is coordinate of each voxel
//' @return Matrix of squared distance between voxels for that region
//' @export
// [[Rcpp::export()]]
arma::mat get_dist_sqrd_mat_from_coord(arma::mat coord_mat);

// Forward-backward to solve linear system \eqn{R^{-1} * b = z} with Cholesky decomposition
// @param R_chol Cholesky decomposition of Matrix R
// @param b vector to solve against
// @return solution z of \eqn{z = R^{-1}b}
arma::vec R_inv_b(const arma::mat& R_chol, const arma::vec& b);

// Forward-backward to solve linear system \eqn{R^{-1} * B = Z} with Cholesky decomposition
// @param R_chol Cholesky decomposition of Matrix R
// @param B Matrix to solve against
// @return solution Z of \eqn{z = R^{-1}b}
arma::mat R_inv_B(const arma::mat& R_chol, const arma::mat& B);

// Compute softplus
// @param x scalar
// @return compute sofplus function log(1 + exp(x))
double softplus(double x);

// Compute softplus
// @param x scalar or matrix
// @return compute sofplus function log(1 + exp(x)) element-wise
arma::mat softplus(arma::mat xMat);

// Compute logistic:
// @param scalar x
// @return compute logistic function 1 / (1+exp(-x))
double logistic(double x);

// Compute sigmoid:
// @param x constrained scalar with bound [lower, upper]
// @param lower scalar lower bound
// @param upper scalar upper bound
// @return unconstrained scalar y
double sigmoid(double x, 
               double lower, 
               double upper);

// Compute sigmoid inverse
// @param y: unconstrained scalar
// @param lower: scalar lower bound
// @param upper: scalar upper bound
// @return constrained scalar x with bound [lower, upper]
double sigmoid_inv(double y, 
                   double lower, 
                   double upper);

// Compute derivative dx/dy with unconstrained y and constrained and x with bound [lower, upper]
// @param unconstrained scalar y
// @param lower: scalar lower bound
// @param upper: scalar upper bound
// @return derivative dx/dy with constraint x with bound [lower, upper]
double sigmoid_inv_derivative(double y, 
                              double lower, 
                              double upper);

// Compute asymptotic variance (inverse of Fisher information) of REML estimator 
Rcpp::List asymptotic_variance(const arma::mat& SingleRegionMatrix_1,
                               const arma::mat& SingleRegionMatrix_2,
                               const arma::mat& timeSqrd_mat,
                               int L_1, int L_2, int M,
                               double kEta, double tauEta, double nugget,
                               double rho,
                               const arma::mat& Z,
                               std::string kernel_type);
#endif
