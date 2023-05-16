#include "OptIntra.h"
#include <math.h>
#include "helper.h"
#include "rbf.h"

/*****************************************************************************
 Intra-regional model
*****************************************************************************/

OptIntra::OptIntra(const arma::mat &data,
                   const arma::mat &design,
                   const arma::mat &distSqrd,
                   const arma::mat &timeSqrd,
                   int numVoxel,
                   int numTimePt,
                   const arma::mat &fixedEffect,
                   KernelType kernelType)
    : data_(data),
      design_(design),
      distSqrd_(distSqrd),
      timeSqrd_(timeSqrd),
      numVoxel_(numVoxel),
      numTimePt_(numTimePt),
      fixedEffect_(fixedEffect),
      kernelType_(kernelType)
{}


// Compute both objective function and its gradient
double OptIntra::EvaluateWithGradient(
  const arma::mat &theta_unrestrict, arma::mat &gradient)
{
  int length_nu = fixedEffect_.n_rows;
  fixedEffect_ = theta_unrestrict.tail_rows(length_nu);

  // Parameter list:
  // phi_gamma, tau_gamma, kGammaj
  // Transform unrestricted parameters to original forms.
  double phi_gamma = softplus(theta_unrestrict(0));
  double tau_gamma = softplus(theta_unrestrict(1));
  double k_gamma = softplus(theta_unrestrict(2));

  // Create necessary components in likelihood evaluation.
  int N = data_.n_rows;
  arma::mat r_region = data_ - design_ * fixedEffect_;
  arma::mat I = arma::eye(N,N);

  // log-likelihood components
  double l1, l2, l3;

  // Construct the covariance matrices

  // Block matrices of B(m1, m2)
  arma::mat B_Region = k_gamma * rbf(timeSqrd_, tau_gamma);
  arma::mat dB_dk_gamma = logistic(theta_unrestrict(2)) * rbf(timeSqrd_, tau_gamma);
  arma::mat dB_dtau_gamma = k_gamma * (logistic(theta_unrestrict(1)) * rbf_deriv(timeSqrd_, tau_gamma)) % B_Region;

  arma::vec eigval_B;
  arma::mat eigvec_B;

  arma::eig_sym(eigval_B, eigvec_B, B_Region);

  // Block matrices of C(v1, v2)
  arma::mat C_Region = get_cor_mat(kernelType_, distSqrd_, phi_gamma);
  arma::mat dC_dphi_gamma = logistic(theta_unrestrict(0)) * get_cor_mat_deriv(kernelType_, distSqrd_, phi_gamma);

  arma::vec eigval_C;
  arma::mat eigvec_C;

  arma::eig_sym(eigval_C, eigvec_C, C_Region);

  // V matrix.
  arma::vec lambda_inv = 1. / (arma::kron(eigval_C, eigval_B) + 1.);
  arma::mat VInv_r_region = kronecker_mvm(eigvec_C, eigvec_B, lambda_inv % kronecker_mvm(eigvec_C.t(), eigvec_B.t(), r_region)) ;

  arma::mat VInv_Z_region = design_;
  VInv_Z_region = VInv_Z_region.each_col( [&eigvec_C, &eigvec_B, &lambda_inv](arma::vec& a){
    a = kronecker_mvm(eigvec_C, eigvec_B, lambda_inv % kronecker_mvm(eigvec_C.t(), eigvec_B.t(), a) );
    });


  // log determinant of V
  l1 = arma::sum(arma::log(1./lambda_inv));

  // log determinant of Z.t() * VInv * Z
  l2 = arma::log_det_sympd(design_.t() * VInv_Z_region);

  // quadratic form of residuals
  arma::mat qdr = r_region.t() * VInv_r_region;
  l3 = log(qdr(0,0));


  double result = 0.5 * (l1 + l2 + (numVoxel_*numTimePt_ - length_nu) * l3);


  // Gradient of component 1: log_det_sympd(Z.t() * VInv * Z)
  arma::vec comp1(3);
  arma::mat HInv = arma::inv_sympd(design_.t() * VInv_Z_region);
  arma::mat dV_phi_gamma_Z = kronecker_mmm(dC_dphi_gamma, B_Region, VInv_Z_region);
  arma::mat dV_tau_gamma_Z = kronecker_mmm(C_Region, dB_dtau_gamma, VInv_Z_region);
  arma::mat dV_kGammaj_Z = kronecker_mmm(C_Region, dB_dk_gamma, VInv_Z_region);

  comp1(0) = arma::trace(-HInv * (VInv_Z_region.t() * dV_phi_gamma_Z));
  comp1(1) = arma::trace(-HInv * (VInv_Z_region.t() * dV_tau_gamma_Z));
  comp1(2) = arma::trace(-HInv * (VInv_Z_region.t() * dV_kGammaj_Z));

  // Gradient of component 2: log_det_sympd(V)
  arma::vec comp2(3);
  arma::vec dC_phi_temp = arma::kron(arma::diagvec(eigvec_C.t() * dC_dphi_gamma * eigvec_C),
                                      arma::diagvec(eigvec_B.t() * B_Region * eigvec_B) );
  comp2(0) = arma::sum(lambda_inv % dC_phi_temp);

  arma::vec dB_tau_temp = arma::kron(arma::diagvec(eigvec_C.t() * C_Region * eigvec_C),
                                      arma::diagvec(eigvec_B.t() * dB_dtau_gamma * eigvec_B) );
  comp2(1) = arma::sum(lambda_inv % dB_tau_temp);

  arma::vec dB_k_temp = arma::kron(arma::diagvec(eigvec_C.t() * C_Region * eigvec_C),
                                    arma::diagvec(eigvec_B.t() * dB_dk_gamma * eigvec_B) );
  comp2(2) = arma::sum(lambda_inv % dB_k_temp);

  // Gradient of component 3: r.t() * VInv * r
  // w.r.t. theta.
  arma::vec comp3_1(3);
  arma::mat qdr_phi_gamma = -VInv_r_region.t() * kronecker_mvm(dC_dphi_gamma, B_Region, VInv_r_region);
  arma::mat qdr_tau_gamma = -VInv_r_region.t() * kronecker_mvm(C_Region, dB_dtau_gamma, VInv_r_region) ;
  arma::mat qdr_k_gamma = -VInv_r_region.t() * kronecker_mvm(C_Region, dB_dk_gamma, VInv_r_region);

  comp3_1(0)= qdr_phi_gamma(0,0)/qdr(0,0);
  comp3_1(1)= qdr_tau_gamma(0,0)/qdr(0,0);
  comp3_1(2)= qdr_k_gamma(0,0)/qdr(0,0);

  comp3_1 = (numVoxel_ * numTimePt_ - length_nu) * comp3_1;

  // w.r.t. nu
  arma::vec comp3_2(length_nu);
  arma::mat comp3_2_temp = -2 * design_.t() * VInv_r_region;
  for (int i = 0; i < length_nu; i++){
    comp3_2(i) = comp3_2_temp(i,0)/qdr(0,0);
  }
  comp3_2 = (numVoxel_ * numTimePt_ - length_nu) * comp3_2;


  // theta gradients
  gradient.head_rows(3) = 0.5 * (comp1 + comp2 + comp3_1);

  // nu gradients
  gradient.tail_rows(length_nu) = 0.5 * comp3_2;
  return (result);
}