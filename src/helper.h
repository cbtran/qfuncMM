#ifndef _HELPER_H
#define _HELPER_H
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <math.h>
#include "region.h"
#include "get_cor_mat.h"

//' @title Kronecker matrix-vector product (Steeb & Hardy, 2011)
//' @description \deqn{(A \otimes B)v = \text{vectorise}(BVA^\top)}
//' @param A Left matrix
//' @param B Right matrix
//' @param v vector
//' @return Kronecker matrix-vector product
//' @references Willi-Hans Steeb and Yorick Hardy. Matrix calculus and Kronecker product: a practical approach to linear and multilinear algebra. World Scientific Publishing Company, 2011.
//' @export
// [[Rcpp::export()]]
arma::vec kronecker_mvm (const arma::mat& A, 
                         const arma::mat& B,
                         const arma::vec& v) {
  arma::mat V_mat(v);
  V_mat.reshape(B.n_cols, A.n_cols);
  return arma::vectorise(B * V_mat * A.t());
}

// Kronecker matrix-matrix product using kronecker structure
arma::mat kronecker_mmm (const arma::mat& A,
                         const arma::mat& B,
                         const arma::mat& C) {
  arma::mat V(C);
  V = V.each_col( [&A, &B](arma::vec& c){ 
    c = kronecker_mvm(A, B, c); 
  });
  
  return V;
}

//' Get distance squared matrix for simulated data
//' @param L Number of voxels 
//' @param sideLength side length of that region
//' @param voxelID ID of sampled voxels
//' @return Matrix of squared distance between voxels for that region
//' @export
// [[Rcpp::export()]]
arma::mat get_dist_sqrd_mat(int L,
                           int sideLength,
                           Rcpp::NumericVector voxelID) {
  // Create a REGION object.
  REGION R(sideLength);
  
  arma::mat dist_sqrdMat(L, L, arma::fill::zeros);
  for (int i = 0; i < L; i++) {
    for (int j = 0; j <= i; j++) {
      int v1 = voxelID[i];
      int v2 = voxelID[j];
      dist_sqrdMat(i,j) = R.getDistPair(v1,v2);
    }
  }
  dist_sqrdMat = arma::symmatl(dist_sqrdMat);
  return (dist_sqrdMat);
}


//' Get distance squared matrix for real data
//' @param coord_mat Matrix with each column is coordinate of each voxel
//' @return Matrix of squared distance between voxels for that region
//' @export
// [[Rcpp::export()]]
arma::mat get_dist_sqrd_mat_from_coord(arma::mat coord_mat) {
  int L = coord_mat.n_cols; // Number of voxels
  arma::mat dist_sqrdMat(L, L, arma::fill::zeros);
  for (int i = 0; i < L; i++) {
    for (int j = 0; j <= i; j++) {
      arma::vec v1 = coord_mat.col(i);
      arma::vec v2 = coord_mat.col(j);
      arma::vec diff = v1 - v2;
      dist_sqrdMat(i,j) = pow(arma::norm(diff, 2), 2);
    }
  }
  dist_sqrdMat = arma::symmatl(dist_sqrdMat);
  
  return (dist_sqrdMat);
}

// Forward-backward to solve linear system \eqn{R^{-1} * b = z} with Cholesky decomposition
// @param R_chol Cholesky decomposition of Matrix R
// @param b vector to solve against
// @return solution z of \eqn{z = R^{-1}b}
arma::vec R_inv_b(const arma::mat& R_chol,
                  const arma::vec& b) {
  arma::vec y = solve(trimatl(R_chol), b);
  arma::vec z = solve(trimatu(R_chol.t()), y);
  
  return (z);
  
} 

// Forward-backward to solve linear system \eqn{R^{-1} * B = Z} with Cholesky decomposition
// @param R_chol Cholesky decomposition of Matrix R
// @param B Matrix to solve against
// @return solution Z of \eqn{z = R^{-1}b}
arma::mat R_inv_B(const arma::mat& R_chol,
                  const arma::mat& B) {
  arma::mat y = solve(trimatl(R_chol), B);
  arma::mat z = solve(trimatu(R_chol.t()), y);
  
  return (z);
  
} 


// Compute softplus
// @param x scalar
// @return compute sofplus function log(1 + exp(x))
double softplus(double x) {
  return log(1. + exp(x));
}

// Compute softplus
// @param x scalar or matrix
// @return compute sofplus function log(1 + exp(x)) element-wise
arma::mat softplus(arma::mat xMat) {
  return arma::log(1. + arma::exp(xMat));
}

// Compute invers of softplus
double softplus_inv(double x) {
  return log(exp(x) - 1.);
}



// Compute logistic:
// @param scalar x
// @return compute logistic function 1 / (1+exp(-x))
double logistic(double x) {
  return 1. / (1. + exp(-x));
}


// Compute sigmoid:
// @param x constrained scalar with bound [lower, upper]
// @param lower scalar lower bound
// @param upper scalar upper bound
// @return unconstrained scalar y
double sigmoid(double x, 
               double lower, 
               double upper) {
  double u = (x-lower) / (upper-lower);
  return log(u/(1.-u));
}

// Compute sigmoid inverse
// @param y: unconstrained scalar
// @param lower: scalar lower bound
// @param upper: scalar upper bound
// @return constrained scalar x with bound [lower, upper]
double sigmoid_inv(double y, 
                   double lower, 
                   double upper) {
  double u = logistic(y);
  double x = lower + (upper - lower) * u;
  return x;
}

// Compute derivative dx/dy with unconstrained y and constrained and x with bound [lower, upper]
// @param unconstrained scalar y
// @param lower: scalar lower bound
// @param upper: scalar upper bound
// @return derivative dx/dy with constraint x with bound [lower, upper]
double sigmoid_inv_derivative(double y, 
                              double lower, 
                              double upper) {
  double u = logistic(y);
  double x_derivative = (upper - lower) * u * (1-u);
  return x_derivative;
}




// Compute asymptotic variance (inverse of Fisher information) of REML estimator 
Rcpp::List asymptotic_variance(const arma::mat& SingleRegionMatrix_1,
                               const arma::mat& SingleRegionMatrix_2,
                               const arma::mat& timeSqrd_mat,
                               int L_1, int L_2, int M,
                               double kEta, double tauEta, double nugget,
                               double rho,
                               const arma::mat& Z,
                               std::string kernel_type) {
  int N = (L_1 + L_2) * M;
  arma::mat rhoL(L_1, L_2);
  rhoL.fill(rho);
  arma::mat I = arma::eye(N,N);
  arma::mat J_L1(L_1, L_1, arma::fill::ones);
  arma::mat J_L2(L_2, L_2, arma::fill::ones);
  
  
  arma::mat At = kEta*rbf(timeSqrd_mat, tauEta) + nugget*arma::eye(M,M);
  
  arma::mat rhoL_At = arma::kron(rhoL, At);
  
  arma::mat V = arma::join_vert(arma::join_horiz(SingleRegionMatrix_1 + kron(J_L1, At), rhoL_At),
                                arma::join_horiz(rhoL_At.t(), SingleRegionMatrix_2 + kron(J_L2, At))) + I;
  
  arma::mat VInv = arma::inv_sympd(V);
  arma::mat VInv_Z = VInv * Z;
  arma::mat HInv = arma::inv_sympd(Z.t() * VInv_Z);
  
  arma::mat rhoLDeriv(L_1, L_2);
  double rho_unrestricted = atanh(rho);
  
  double rho_deriv = 1 - pow(rho, 2);
  rhoLDeriv.fill(rho_deriv);
  
  arma::mat Zero_L1_M(L_1*M, L_1*M, arma::fill::zeros);
  arma::mat Zero_L2_M(L_2*M, L_2*M, arma::fill::zeros);
  arma::mat rhoLSqrd_At = arma::kron(rhoLDeriv, At);
  
  arma::mat dV_rho = arma::join_vert(arma::join_horiz(Zero_L1_M, rhoLSqrd_At),
                                     arma::join_horiz(rhoLSqrd_At.t(), Zero_L2_M));
  
  double tr = trace(0.5 * arma::powmat((VInv - VInv_Z * HInv * VInv_Z.t()) * dV_rho, 2));
  return Rcpp::List::create(Rcpp::Named("asymptotic_var") = 1/tr,
                            Rcpp::Named("rho_unrestricted") = rho_unrestricted);
} 

#endif
