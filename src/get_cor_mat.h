#ifndef GET_COR_MAT_H
#define GET_COR_MAT_H
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

enum KernelType
{
    Rbf = 0,
    Matern12 = 1,
    Matern32 = 2,
    Matern52 = 3
};

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
//' @noRd
arma::mat get_cor_mat(KernelType kernel_type,
                      arma::mat xSqrd_mat,
                      double tau);

// Compute derivative of correlation matrix w.r.t tau
// @param corfun_deriv_name_string Choice of kernel function
// @param xSqrd_mat Matrix of squared difference between inputs x's
// @param tau Inverse lengthscale parameter
// @return cormat_deriv Derivative of correlation matrix with respect to \eqn{\tau}
arma::mat get_cor_mat_deriv(KernelType kernel_type,
                            arma::mat xSqrd_mat,
                            double tau);

#endif
