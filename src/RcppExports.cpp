// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// get_cor_mat
arma::mat get_cor_mat(std::string corfun_name_string, arma::mat xSqrd_mat, double tau);
RcppExport SEXP _qfuncMM_get_cor_mat(SEXP corfun_name_stringSEXP, SEXP xSqrd_matSEXP, SEXP tauSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::string >::type corfun_name_string(corfun_name_stringSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type xSqrd_mat(xSqrd_matSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    rcpp_result_gen = Rcpp::wrap(get_cor_mat(corfun_name_string, xSqrd_mat, tau));
    return rcpp_result_gen;
END_RCPP
}
// kronecker_mvm
arma::vec kronecker_mvm(const arma::mat& A, const arma::mat& B, const arma::vec& v);
RcppExport SEXP _qfuncMM_kronecker_mvm(SEXP ASEXP, SEXP BSEXP, SEXP vSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type B(BSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type v(vSEXP);
    rcpp_result_gen = Rcpp::wrap(kronecker_mvm(A, B, v));
    return rcpp_result_gen;
END_RCPP
}
// get_dist_sqrd_mat
arma::mat get_dist_sqrd_mat(arma::mat coords);
RcppExport SEXP _qfuncMM_get_dist_sqrd_mat(SEXP coordsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type coords(coordsSEXP);
    rcpp_result_gen = Rcpp::wrap(get_dist_sqrd_mat(coords));
    return rcpp_result_gen;
END_RCPP
}
// opt_intra
Rcpp::List opt_intra(const arma::vec& theta_init, const arma::mat& X_region, const arma::mat& Z_region, const arma::mat& dist_sqrd_mat, const arma::mat& time_sqrd_mat, int L, int M, std::string kernel_type);
RcppExport SEXP _qfuncMM_opt_intra(SEXP theta_initSEXP, SEXP X_regionSEXP, SEXP Z_regionSEXP, SEXP dist_sqrd_matSEXP, SEXP time_sqrd_matSEXP, SEXP LSEXP, SEXP MSEXP, SEXP kernel_typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type theta_init(theta_initSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_region(X_regionSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Z_region(Z_regionSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type dist_sqrd_mat(dist_sqrd_matSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type time_sqrd_mat(time_sqrd_matSEXP);
    Rcpp::traits::input_parameter< int >::type L(LSEXP);
    Rcpp::traits::input_parameter< int >::type M(MSEXP);
    Rcpp::traits::input_parameter< std::string >::type kernel_type(kernel_typeSEXP);
    rcpp_result_gen = Rcpp::wrap(opt_intra(theta_init, X_region, Z_region, dist_sqrd_mat, time_sqrd_mat, L, M, kernel_type));
    return rcpp_result_gen;
END_RCPP
}
// opt_inter
Rcpp::List opt_inter(const arma::vec& theta_init, const arma::mat& X, const arma::mat& Z, const arma::mat& dist_sqrdMat_1, const arma::mat& dist_sqrdMat_2, const arma::mat& time_sqrd_mat, const arma::vec& stage1_regional, std::string kernel_type);
RcppExport SEXP _qfuncMM_opt_inter(SEXP theta_initSEXP, SEXP XSEXP, SEXP ZSEXP, SEXP dist_sqrdMat_1SEXP, SEXP dist_sqrdMat_2SEXP, SEXP time_sqrd_matSEXP, SEXP stage1_regionalSEXP, SEXP kernel_typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type theta_init(theta_initSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type dist_sqrdMat_1(dist_sqrdMat_1SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type dist_sqrdMat_2(dist_sqrdMat_2SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type time_sqrd_mat(time_sqrd_matSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type stage1_regional(stage1_regionalSEXP);
    Rcpp::traits::input_parameter< std::string >::type kernel_type(kernel_typeSEXP);
    rcpp_result_gen = Rcpp::wrap(opt_inter(theta_init, X, Z, dist_sqrdMat_1, dist_sqrdMat_2, time_sqrd_mat, stage1_regional, kernel_type));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_qfuncMM_get_cor_mat", (DL_FUNC) &_qfuncMM_get_cor_mat, 3},
    {"_qfuncMM_kronecker_mvm", (DL_FUNC) &_qfuncMM_kronecker_mvm, 3},
    {"_qfuncMM_get_dist_sqrd_mat", (DL_FUNC) &_qfuncMM_get_dist_sqrd_mat, 1},
    {"_qfuncMM_opt_intra", (DL_FUNC) &_qfuncMM_opt_intra, 8},
    {"_qfuncMM_opt_inter", (DL_FUNC) &_qfuncMM_opt_inter, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_qfuncMM(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
