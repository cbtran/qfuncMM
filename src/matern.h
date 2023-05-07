#ifndef MATERN_H
#define MATERN_H
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

double matern_1_2(double xSqrd, double tau);

arma::vec matern_1_2(arma::vec xSqrd_mat, double tau);

arma::mat matern_1_2(arma::mat xSqrd_mat, double tau);

arma::mat matern_1_2_deriv(arma::mat xSqrd_mat, double tau);

double matern_3_2(double xSqrd, double tau);

arma::mat matern_3_2(arma::mat xSqrd_mat, double tau);

arma::vec matern_3_2(arma::vec xSqrd_vec, double tau);

arma::mat matern_3_2_deriv(arma::mat xSqrd_mat, double tau);

double matern_5_2(double xSqrd, double tau);

arma::mat matern_5_2(arma::mat xSqrd_mat, double tau);

arma::vec matern_5_2(arma::vec xSqrd_vec, double tau);

arma::mat matern_5_2_deriv(arma::mat xSqrd_mat, double tau);

#endif