#ifndef MATERN_H
#define MATERN_H
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat matern_1_2(arma::mat xSqrd_mat, double tau);

arma::mat matern_1_2_deriv(arma::mat xSqrd_mat, double tau);

arma::mat matern_3_2(arma::mat xSqrd_mat, double tau);

arma::mat matern_3_2_deriv(arma::mat xSqrd_mat, double tau);

arma::mat matern_5_2(arma::mat xSqrd_mat, double tau);

arma::mat matern_5_2_deriv(arma::mat xSqrd_mat, double tau);

#endif