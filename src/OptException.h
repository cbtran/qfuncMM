#ifndef OPT_EXCEPTION_H
#define OPT_EXCEPTION_H

#include <RcppArmadillo.h>
#include "Rcpp/exceptions.h"

class OptException : public Rcpp::exception
{
  public:
    OptException(std::string msg) : Rcpp::exception(msg.c_str()) {}
};

#endif