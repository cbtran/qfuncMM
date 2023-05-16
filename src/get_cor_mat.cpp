#include <math.h>
#include "get_cor_mat.h"
#include "rbf.h"
#include "matern.h"

arma::mat get_cor_mat(KernelType kernel_type,
                      arma::mat xSqrd_mat,
                      double tau) {
  arma::mat cormat;
  switch(kernel_type) {
    case KernelType::Rbf:
      cormat = rbf(xSqrd_mat, tau);
      break;
    case KernelType::Matern52:
      cormat = matern_5_2(xSqrd_mat, tau);
      break;
    case KernelType::Matern32:
      cormat = matern_3_2(xSqrd_mat, tau);
      break;
    case KernelType::Matern12:
      cormat = matern_1_2(xSqrd_mat, tau);
      break;
    default:
      throw(Rcpp::exception("Unrecognized Kernel"));
  }
  return cormat;
}

arma::mat get_cor_mat_deriv(KernelType kernel_type,
                            arma::mat xSqrd_mat,
                            double tau) {
  arma::mat cormat;
  switch(kernel_type) {
    case KernelType::Rbf:
      cormat = rbf_deriv(xSqrd_mat, tau);
      break;
    case KernelType::Matern52:
      cormat = matern_5_2_deriv(xSqrd_mat, tau);
      break;
    case KernelType::Matern32:
      cormat = matern_3_2_deriv(xSqrd_mat, tau);
      break;
    case KernelType::Matern12:
      cormat = matern_1_2_deriv(xSqrd_mat, tau);
      break;
    default:
      throw(Rcpp::exception("Unrecognized Kernel"));
  }
  return cormat;
}
