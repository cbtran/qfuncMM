#ifndef _GET_COR_MAT_H
#define _GET_COR_MAT_H
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <math.h>
#include "rbf.h"
#include "matern.h"


//' Compute correlation matrix
//' @param corfun_name_string Choice of kernel function
//' @param xSqrd_mat Matrix of squared difference between inputs x's
//' @param tau Inverse lengthscale parameter
//' @return cormat Correlation matrix from vector of inputs x's
//' @details The correlation matrix with corresponding choice of kernel: For RBF kernel \deqn{C(x_i,x_j) = \exp(-\tau^2/2 \times d^2),}
//' for Matern 1/2 kernel \deqn{C(x_i,x_j) = \exp(-\tau \times d),}
//' for Matern 3/2 kernel \deqn{C(x_i,x_j) = (1 + \sqrt{3} \times\tau \times d)\exp(-\sqrt{3}\times\tau \times d),}
//' and for Matern 5/2 kernel \deqn{C(x_i,x_j) = (1 + \sqrt{3} \times\tau \times d + \sqrt{5}/\sqrt{3} \times\tau^2 \times d^2)\exp(-\sqrt{5}\times\tau \times d),}
//' with \eqn{d = || x_i - x_j ||_2^2} is the squared distance between locations i and j.
//' @export
// [[Rcpp::export()]]
arma::mat get_cor_mat(std::string corfun_name_string, 
                      arma::mat xSqrd_mat, 
                      double tau) {
  arma::mat cormat;
  if( corfun_name_string.compare("rbf") == 0 )
  { 
    cormat = rbf(xSqrd_mat, tau);
  } 
  else if( corfun_name_string.compare("matern_5_2") == 0 )
  { 
    cormat = matern_5_2(xSqrd_mat, tau); 
  }
  else if( corfun_name_string.compare("matern_3_2") == 0 )
  { 
    cormat = matern_3_2(xSqrd_mat, tau); 
  }
  else if( corfun_name_string.compare("matern_1_2") == 0 )
  { 
    cormat = matern_1_2(xSqrd_mat, tau); 
  }
  else {
    Rcpp::Rcout << "Unrecognized Kernel \n";
  }
  return cormat;
  
}

// Compute derivative of correlation matrix w.r.t tau
// @param corfun_deriv_name_string Choice of kernel function
// @param xSqrd_mat Matrix of squared difference between inputs x's
// @param tau Inverse lengthscale parameter
// @return cormat_deriv Derivative of correlation matrix with respect to \eqn{\tau}
arma::mat get_cor_mat_deriv(std::string corfun_deriv_name_string,
                            arma::mat xSqrd_mat,
                            double tau) {
  arma::mat cormat;
  if( corfun_deriv_name_string.compare("rbf") == 0 )
  { 
    cormat = rbf_deriv(xSqrd_mat, tau);
  } 
  else if( corfun_deriv_name_string.compare("matern_5_2") == 0 )
  { 
    cormat = matern_5_2_deriv(xSqrd_mat, tau); 
  }
  else if( corfun_deriv_name_string.compare("matern_3_2") == 0 )
  { 
    cormat = matern_3_2_deriv(xSqrd_mat, tau); 
  }
  else if( corfun_deriv_name_string.compare("matern_1_2") == 0 )
  { 
    cormat = matern_1_2_deriv(xSqrd_mat, tau); 
  }
  else {
    Rcpp::Rcout << "Unrecognized Kernel \n";
  }
  return cormat;
  
}
#endif