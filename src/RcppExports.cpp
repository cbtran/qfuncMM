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
Rcpp::List opt_intra(const arma::vec& theta_init, const arma::mat& X_region, const arma::mat& voxel_coords, const arma::mat& time_sqrd_mat, int kernel_type_id, bool nugget_only, bool noiseless);
RcppExport SEXP _qfuncMM_opt_intra(SEXP theta_initSEXP, SEXP X_regionSEXP, SEXP voxel_coordsSEXP, SEXP time_sqrd_matSEXP, SEXP kernel_type_idSEXP, SEXP nugget_onlySEXP, SEXP noiselessSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type theta_init(theta_initSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_region(X_regionSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type voxel_coords(voxel_coordsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type time_sqrd_mat(time_sqrd_matSEXP);
    Rcpp::traits::input_parameter< int >::type kernel_type_id(kernel_type_idSEXP);
    Rcpp::traits::input_parameter< bool >::type nugget_only(nugget_onlySEXP);
    Rcpp::traits::input_parameter< bool >::type noiseless(noiselessSEXP);
    rcpp_result_gen = Rcpp::wrap(opt_intra(theta_init, X_region, voxel_coords, time_sqrd_mat, kernel_type_id, nugget_only, noiseless));
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
Rcpp::List opt_inter(const arma::vec& theta_init, const arma::mat& dataRegion1, const arma::mat& dataRegion2, const arma::mat& voxel_coords_1, const arma::mat& voxel_coords_2, const arma::mat& time_sqrd_mat, const arma::vec& stage1ParamsRegion1, const arma::vec& stage1ParamsRegion2, int kernel_type_id, bool diag_time);
RcppExport SEXP _qfuncMM_opt_inter(SEXP theta_initSEXP, SEXP dataRegion1SEXP, SEXP dataRegion2SEXP, SEXP voxel_coords_1SEXP, SEXP voxel_coords_2SEXP, SEXP time_sqrd_matSEXP, SEXP stage1ParamsRegion1SEXP, SEXP stage1ParamsRegion2SEXP, SEXP kernel_type_idSEXP, SEXP diag_timeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type theta_init(theta_initSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type dataRegion1(dataRegion1SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type dataRegion2(dataRegion2SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type voxel_coords_1(voxel_coords_1SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type voxel_coords_2(voxel_coords_2SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type time_sqrd_mat(time_sqrd_matSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type stage1ParamsRegion1(stage1ParamsRegion1SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type stage1ParamsRegion2(stage1ParamsRegion2SEXP);
    Rcpp::traits::input_parameter< int >::type kernel_type_id(kernel_type_idSEXP);
    Rcpp::traits::input_parameter< bool >::type diag_time(diag_timeSEXP);
    rcpp_result_gen = Rcpp::wrap(opt_inter(theta_init, dataRegion1, dataRegion2, voxel_coords_1, voxel_coords_2, time_sqrd_mat, stage1ParamsRegion1, stage1ParamsRegion2, kernel_type_id, diag_time));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_qfuncMM_opt_intra", (DL_FUNC) &_qfuncMM_opt_intra, 7},
    {"_qfuncMM_eval_stage1_nll", (DL_FUNC) &_qfuncMM_eval_stage1_nll, 5},
    {"_qfuncMM_opt_inter", (DL_FUNC) &_qfuncMM_opt_inter, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_qfuncMM(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
