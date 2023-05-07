#include <math.h>
#include "get_cor_mat.h"
#include "rbf.h"
#include "matern.h"

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
