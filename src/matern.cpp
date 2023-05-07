#include <math.h>
#include <RcppArmadillo.h>
#include "matern.h"

double matern_1_2(double xSqrd,
                  double tau)
{
  double x = sqrt(xSqrd);
  return exp(-tau * x);
}
arma::vec matern_1_2(arma::vec xSqrd_mat,
                     double tau)
{
  arma::vec x = arma::sqrt(xSqrd_mat);
  return arma::exp(-tau * x);
}

arma::mat matern_1_2(arma::mat xSqrd_mat,
                     double tau)
{
  arma::mat x = arma::sqrt(xSqrd_mat);
  return arma::exp(-tau * x);
}

arma::mat matern_1_2_deriv(arma::mat xSqrd_mat,
                           double tau)
{
  arma::mat x = arma::sqrt(xSqrd_mat);
  return arma::exp(-tau * x) % (-x);
}

double matern_3_2(double xSqrd,
                  double tau)
{
  double x = sqrt(xSqrd);
  return (1 + tau * sqrt(3.0) * x) * exp(-tau * sqrt(3.0) * x);
}

arma::mat matern_3_2(arma::mat xSqrd_mat,
                     double tau)
{
  arma::mat x = sqrt(xSqrd_mat);
  return (1 + tau * sqrt(3.0) * x) % arma::exp(-tau * sqrt(3.0) * x);
}

arma::vec matern_3_2(arma::vec xSqrd_vec,
                     double tau)
{
  arma::vec x = sqrt(xSqrd_vec);
  return (1 + tau * sqrt(3.0) * x) % arma::exp(-tau * sqrt(3.0) * x);
}

arma::mat matern_3_2_deriv(arma::mat xSqrd_mat,
                           double tau)
{
  arma::mat x = sqrt(xSqrd_mat);
  arma::mat exp_tau_x = exp(-tau * sqrt(3.0) * x);
  arma::mat deriv_1 = (sqrt(3.0) * x) % exp_tau_x;
  arma::mat deriv_2 = (1.0 + tau * sqrt(3.0) * x) % exp_tau_x % (-sqrt(3.0) * x);
  return (deriv_1 + deriv_2);
}

double matern_5_2(double xSqrd,
                  double tau)
{
  double x = sqrt(xSqrd);
  return (1.0 + tau * sqrt(5.0) * x + pow(tau, 2.0) * (5.0 / 3.0) * xSqrd) * exp(-tau * sqrt(5.0) * x);
}

arma::mat matern_5_2(arma::mat xSqrd_mat,
                     double tau)
{
  arma::mat x = arma::sqrt(xSqrd_mat);
  return (1.0 + tau * sqrt(5.0) * x + pow(tau, 2.0) * (5.0 / 3.0) * xSqrd_mat) % arma::exp(-tau * sqrt(5.0) * x);
}

arma::vec matern_5_2(arma::vec xSqrd_vec,
                     double tau)
{
  arma::vec x = arma::sqrt(xSqrd_vec);
  return (1.0 + tau * sqrt(5.0) * x + pow(tau, 2.0) * (5.0 / 3.0) * xSqrd_vec) % arma::exp(-tau * sqrt(5.0) * x);
}

arma::mat matern_5_2_deriv(arma::mat xSqrd_mat,
                           double tau)
{
  arma::mat x = arma::sqrt(xSqrd_mat);
  arma::mat exp_tau_x = arma::exp(-tau * sqrt(5.0) * x);
  arma::mat deriv_1 = (sqrt(5.0) * x + 10.0 / 3.0 * tau * xSqrd_mat) % exp_tau_x;
  arma::mat deriv_2 = (1.0 + tau * sqrt(5.0) * x + pow(tau, 2.0) * (5.0 / 3.0) * xSqrd_mat) % exp_tau_x % (-sqrt(5.0) * x);
  return (deriv_1 + deriv_2);
}