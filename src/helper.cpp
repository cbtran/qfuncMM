#include "helper.h"
#include "matern.h"
#include "rbf.h"
#include <math.h>

arma::vec kronecker_mvm (const arma::mat& A, 
                         const arma::mat& B,
                         const arma::vec& v) {
  arma::mat V_mat(v);
  V_mat.reshape(B.n_cols, A.n_cols);
  return arma::vectorise(B * V_mat * A.t());
}

arma::mat kronecker_mmm (const arma::mat& A,
                         const arma::mat& B,
                         const arma::mat& C) {
  arma::mat V(C.n_rows, C.n_cols);
  int n_cols = C.n_cols;
  for (int i = 0; i < n_cols; i++)
  {
    V.col(i) = kronecker_mvm(A, B, C.col(i));
  }
  
  return V;
}

arma::mat get_dist_sqrd_mat(int L, int sideLength, Rcpp::NumericVector voxelID)
{
  arma::mat distSqrd(L, L, arma::fill::zeros);

  int sideLengthSqrd = pow(sideLength, 2);

  for (int i = 0; i < L; i++) {
    for (int j = 0; j <= i; j++) {
      // Convert to vector of coordinates assuming column-major order
      int v1 = voxelID[i];
      int v1z = v1 % sideLength;
      int v1y = v1 / sideLength;
      int v1x = v1 / sideLengthSqrd;

      int v2 = voxelID[j];
      int v2z = v2 % sideLength;
      int v2y = v2 / sideLength;
      int v2x = v2 / sideLengthSqrd;

      distSqrd(i, j) = pow(v1x - v2x, 2) + pow(v1y - v2y, 2) + pow(v1z - v2z, 2);
    }
  }
  distSqrd = arma::symmatl(distSqrd);
  return (distSqrd);
}


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

arma::vec R_inv_b(const arma::mat& R_chol,
                  const arma::vec& b) {
  arma::vec y = solve(trimatl(R_chol), b);
  arma::vec z = solve(trimatu(R_chol.t()), y);
  
  return (z);
  
} 

arma::mat R_inv_B(const arma::mat& R_chol,
                  const arma::mat& B) {
  arma::mat y = solve(trimatl(R_chol), B);
  arma::mat z = solve(trimatu(R_chol.t()), y);
  
  return (z);
  
} 

double softplus(double x) { return R::log1pexp(x); }

arma::mat softplus(arma::mat xMat) {
  return xMat.transform([](double x){return softplus(x);});
}

double logistic(double x) {
  return 1. / (1. + exp(-x));
}

double sigmoid(double x, 
               double lower, 
               double upper) {
  double u = (x-lower) / (upper-lower);
  return log(u) - log(1 - u);
}

double sigmoid_inv(double y, 
                   double lower, 
                   double upper) {
  double u = logistic(y);
  double x = lower + (upper - lower) * u;
  return x;
}

double sigmoid_inv_derivative(double y, 
                              double lower, 
                              double upper) {
  double u = logistic(y);
  double x_derivative = (upper - lower) * u * (1-u);
  return x_derivative;
}

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
