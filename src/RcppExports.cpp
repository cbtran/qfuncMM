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

// opt_intra
Rcpp::List opt_intra(const arma::vec& theta_init, const arma::mat& X_region, const arma::mat& voxel_coords, const arma::mat& time_sqrd_mat, int kernel_type_id, int cov_setting_id, bool verbose);
RcppExport SEXP _qfuncMM_opt_intra(SEXP theta_initSEXP, SEXP X_regionSEXP, SEXP voxel_coordsSEXP, SEXP time_sqrd_matSEXP, SEXP kernel_type_idSEXP, SEXP cov_setting_idSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type theta_init(theta_initSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_region(X_regionSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type voxel_coords(voxel_coordsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type time_sqrd_mat(time_sqrd_matSEXP);
    Rcpp::traits::input_parameter< int >::type kernel_type_id(kernel_type_idSEXP);
    Rcpp::traits::input_parameter< int >::type cov_setting_id(cov_setting_idSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(opt_intra(theta_init, X_region, voxel_coords, time_sqrd_mat, kernel_type_id, cov_setting_id, verbose));
    return rcpp_result_gen;
END_RCPP
}
// eval_stage1_nll
Rcpp::List eval_stage1_nll(const arma::vec& theta, const arma::mat& X_region, const arma::mat& voxel_coords, const arma::mat& time_sqrd_mat, int kernel_type_id);
RcppExport SEXP _qfuncMM_eval_stage1_nll(SEXP thetaSEXP, SEXP X_regionSEXP, SEXP voxel_coordsSEXP, SEXP time_sqrd_matSEXP, SEXP kernel_type_idSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_region(X_regionSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type voxel_coords(voxel_coordsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type time_sqrd_mat(time_sqrd_matSEXP);
    Rcpp::traits::input_parameter< int >::type kernel_type_id(kernel_type_idSEXP);
    rcpp_result_gen = Rcpp::wrap(eval_stage1_nll(theta, X_region, voxel_coords, time_sqrd_mat, kernel_type_id));
    return rcpp_result_gen;
END_RCPP
}
// opt_inter
Rcpp::List opt_inter(const arma::vec& theta_init, const arma::mat& data_r1, const arma::mat& data_r2, const arma::mat& coords_r1, const arma::mat& coords_r2, const arma::mat& time_sqrd_mat, const Rcpp::NumericVector& stage1_r1, const Rcpp::NumericVector& stage1_r2, int cov_setting_id1, int cov_setting_id2, int kernel_type_id, bool verbose);
RcppExport SEXP _qfuncMM_opt_inter(SEXP theta_initSEXP, SEXP data_r1SEXP, SEXP data_r2SEXP, SEXP coords_r1SEXP, SEXP coords_r2SEXP, SEXP time_sqrd_matSEXP, SEXP stage1_r1SEXP, SEXP stage1_r2SEXP, SEXP cov_setting_id1SEXP, SEXP cov_setting_id2SEXP, SEXP kernel_type_idSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type theta_init(theta_initSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type data_r1(data_r1SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type data_r2(data_r2SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords_r1(coords_r1SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords_r2(coords_r2SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type time_sqrd_mat(time_sqrd_matSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type stage1_r1(stage1_r1SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type stage1_r2(stage1_r2SEXP);
    Rcpp::traits::input_parameter< int >::type cov_setting_id1(cov_setting_id1SEXP);
    Rcpp::traits::input_parameter< int >::type cov_setting_id2(cov_setting_id2SEXP);
    Rcpp::traits::input_parameter< int >::type kernel_type_id(kernel_type_idSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(opt_inter(theta_init, data_r1, data_r2, coords_r1, coords_r2, time_sqrd_mat, stage1_r1, stage1_r2, cov_setting_id1, cov_setting_id2, kernel_type_id, verbose));
    return rcpp_result_gen;
END_RCPP
}
// get_fisher_info
Rcpp::NumericMatrix get_fisher_info(const arma::vec& theta, const arma::mat& coords_r1, const arma::mat& coords_r2, const arma::mat& time_sqrd_mat, const Rcpp::NumericVector& stage1_r1, const Rcpp::NumericVector& stage1_r2, int cov_setting_id1, int cov_setting_id2, int kernel_type_id, bool reml);
RcppExport SEXP _qfuncMM_get_fisher_info(SEXP thetaSEXP, SEXP coords_r1SEXP, SEXP coords_r2SEXP, SEXP time_sqrd_matSEXP, SEXP stage1_r1SEXP, SEXP stage1_r2SEXP, SEXP cov_setting_id1SEXP, SEXP cov_setting_id2SEXP, SEXP kernel_type_idSEXP, SEXP remlSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords_r1(coords_r1SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords_r2(coords_r2SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type time_sqrd_mat(time_sqrd_matSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type stage1_r1(stage1_r1SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type stage1_r2(stage1_r2SEXP);
    Rcpp::traits::input_parameter< int >::type cov_setting_id1(cov_setting_id1SEXP);
    Rcpp::traits::input_parameter< int >::type cov_setting_id2(cov_setting_id2SEXP);
    Rcpp::traits::input_parameter< int >::type kernel_type_id(kernel_type_idSEXP);
    Rcpp::traits::input_parameter< bool >::type reml(remlSEXP);
    rcpp_result_gen = Rcpp::wrap(get_fisher_info(theta, coords_r1, coords_r2, time_sqrd_mat, stage1_r1, stage1_r2, cov_setting_id1, cov_setting_id2, kernel_type_id, reml));
    return rcpp_result_gen;
END_RCPP
}
// get_asymp_var_rho_approx_cpp
double get_asymp_var_rho_approx_cpp(const arma::vec& theta, const arma::mat& coords_r1, const arma::mat& coords_r2, const arma::mat& time_sqrd_mat, const Rcpp::NumericVector& stage1_r1, const Rcpp::NumericVector& stage1_r2, int cov_setting_id1, int cov_setting_id2, int kernel_type_id, bool reml, bool fast);
RcppExport SEXP _qfuncMM_get_asymp_var_rho_approx_cpp(SEXP thetaSEXP, SEXP coords_r1SEXP, SEXP coords_r2SEXP, SEXP time_sqrd_matSEXP, SEXP stage1_r1SEXP, SEXP stage1_r2SEXP, SEXP cov_setting_id1SEXP, SEXP cov_setting_id2SEXP, SEXP kernel_type_idSEXP, SEXP remlSEXP, SEXP fastSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords_r1(coords_r1SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords_r2(coords_r2SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type time_sqrd_mat(time_sqrd_matSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type stage1_r1(stage1_r1SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type stage1_r2(stage1_r2SEXP);
    Rcpp::traits::input_parameter< int >::type cov_setting_id1(cov_setting_id1SEXP);
    Rcpp::traits::input_parameter< int >::type cov_setting_id2(cov_setting_id2SEXP);
    Rcpp::traits::input_parameter< int >::type kernel_type_id(kernel_type_idSEXP);
    Rcpp::traits::input_parameter< bool >::type reml(remlSEXP);
    Rcpp::traits::input_parameter< bool >::type fast(fastSEXP);
    rcpp_result_gen = Rcpp::wrap(get_asymp_var_rho_approx_cpp(theta, coords_r1, coords_r2, time_sqrd_mat, stage1_r1, stage1_r2, cov_setting_id1, cov_setting_id2, kernel_type_id, reml, fast));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_qfuncMM_opt_intra", (DL_FUNC) &_qfuncMM_opt_intra, 7},
    {"_qfuncMM_eval_stage1_nll", (DL_FUNC) &_qfuncMM_eval_stage1_nll, 5},
    {"_qfuncMM_opt_inter", (DL_FUNC) &_qfuncMM_opt_inter, 12},
    {"_qfuncMM_get_fisher_info", (DL_FUNC) &_qfuncMM_get_fisher_info, 10},
    {"_qfuncMM_get_asymp_var_rho_approx_cpp", (DL_FUNC) &_qfuncMM_get_asymp_var_rho_approx_cpp, 11},
    {NULL, NULL, 0}
};

RcppExport void R_init_qfuncMM(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
