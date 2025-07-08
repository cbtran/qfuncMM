#include "helper.h"
#include <math.h>

// This function performs the "vec-trick" to compute the kronecker product.
// Instead of computing (A \otimes B) * v, we compute vec(B * V * A^T), where
// V is a matrix reshaped from v. This is more efficient than computing
// the kronecker product directly.
arma::vec kronecker_mvm(const arma::mat &A, const arma::mat &B,
                        const arma::vec &v, bool eye_B) {
  arma::mat V_mat(v);
  V_mat.reshape(B.n_cols, A.n_cols);
  if (eye_B) {
    return arma::vectorise(V_mat * A.t());
  }
  return arma::vectorise(B * V_mat * A.t());
}

// This function computes (A \otimes B) * C
// by applying the vec-trick to each column of C.
arma::mat kronecker_mmm(const arma::mat &A, const arma::mat &B,
                        const arma::mat &C, bool eye_B) {
  arma::mat V(A.n_rows * B.n_rows, C.n_cols);
  int n_cols = C.n_cols;
  for (int i = 0; i < n_cols; i++) {
    V.col(i) = kronecker_mvm(A, B, C.col(i), eye_B);
  }

  return V;
}

arma::vec kronecker_ovm(const arma::mat &B, const arma::vec &v,
                        arma::uword n_row_A, double scalar) {
  arma::uword n_col = v.n_elem / B.n_cols;
  arma::mat V_mat = arma::reshape(v, B.n_cols, n_col);
  arma::vec row_sums = sum(V_mat, 1) * scalar;
  arma::mat b_vec = B * row_sums;
  arma::vec result = arma::repmat(b_vec, n_row_A, 1);
  return result;
}

arma::mat kronecker_omm(const arma::mat &B, const arma::mat &C,
                        arma::uword n_row, double scalar) {
  arma::uword n_cols = C.n_cols;
  arma::mat V(n_row * B.n_rows, n_cols);
  for (arma::uword i = 0; i < n_cols; i++) {
    V.col(i) = kronecker_ovm(B, C.col(i), n_row, scalar);
  }
  return V;
}

arma::mat squared_distance(arma::mat coords) {
  int n = coords.n_rows;
  arma::mat result(n, n, arma::fill::zeros);

  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      arma::vec x_i = coords.row(i).t();
      arma::vec x_j = coords.row(j).t();
      double d_ij = sum(square(x_i - x_j));
      result(i, j) = d_ij;
      result(j, i) = d_ij;
    }
  }

  return result;
}

arma::mat R_inv_B(const arma::mat &R_chol, const arma::mat &B) {
  arma::mat y = solve(trimatl(R_chol), B);
  arma::mat z = solve(trimatu(R_chol.t()), y);

  return (z);
}

double softplus(double x) { return R::log1pexp(x); }

arma::mat softplus(arma::mat xMat) {
  return xMat.transform([](double x) { return softplus(x); });
}

double softminus(double x) { return x > 10 ? x : log(exp(x) - 1); }

arma::mat softminus(arma::mat xMat) {
  return xMat.transform([](double x) { return softminus(x); });
}

double logistic(double x) { return exp(-log1pexp(-x)); }

double sigmoid_inv(double y, double lower, double upper) {
  double u = logistic(y);
  double x = lower + (upper - lower) * u;
  return x;
}

double sigmoid_inv_derivative(double y, double lower, double upper) {
  double u = logistic(y);
  double x_derivative = (upper - lower) * u * (1 - u);
  return x_derivative;
}
