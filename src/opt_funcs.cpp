#include <iostream>
#include <RcppEnsmallen.h>
#include <math.h>
#include "helper.h"
// [[Rcpp::depends(RcppEnsmallen)]]

/*****************************************************************************
 Intra-regional model
*****************************************************************************/

class OptIntra
{
  
private:
  const arma::mat& X_region; // The data matrix.
  const arma::mat& Z_region; // The design matrix.
  const arma::mat& dist_sqrd_mat; // Square spatial distance matrix
  const arma::mat& time_sqrd_mat; // Square temporal distance matrix
  int L; // Number of voxels 
  int M; // Number of time points 
  arma::mat& nu; // Fixed-effect vector
  std::string kernel_type; // Choice of spatial kernel
  
public:
  // Construct the object with the given data
  OptIntra(const arma::mat& X_region, 
                const arma::mat& Z_region, 
                const arma::mat& dist_sqrd_mat, 
                const arma::mat& time_sqrd_mat,
                int L, int M, 
                arma::mat& nu, 
                std::string kernel_type) :
  
  X_region(X_region), 
  Z_region(Z_region), 
  dist_sqrd_mat(dist_sqrd_mat), 
  time_sqrd_mat(time_sqrd_mat),
  L(L), M(M),
  nu(nu), 
  kernel_type(kernel_type){}
  
  // Compute both objective function and its gradient
  double EvaluateWithGradient(const arma::mat &theta_unrestrict,
                              arma::mat &gradient)
  {
    int length_nu = nu.n_rows;
    nu = theta_unrestrict.tail_rows(length_nu);
    
    // Parameter list:
    // phi_gamma, tau_gamma, kGammaj
    // Transform unrestricted parameters to original forms.
    double phi_gamma = softplus(theta_unrestrict(0));
    double tau_gamma = softplus(theta_unrestrict(1));
    double k_gamma = softplus(theta_unrestrict(2));
    
    // Create necessary components in likelihood evaluation.
    int N = X_region.n_rows;
    arma::mat r_region = X_region - Z_region * nu;
    arma::mat I = arma::eye(N,N);
    
    // log-likelihood components
    double l1, l2, l3;
    
    // Construct the covariance matrices
    
    // Block matrices of B(m1, m2)
    arma::mat B_Region = k_gamma * rbf(time_sqrd_mat, tau_gamma);
    arma::mat dB_dk_gamma = logistic(theta_unrestrict(2)) * rbf(time_sqrd_mat, tau_gamma);
    arma::mat dB_dtau_gamma = k_gamma * (logistic(theta_unrestrict(1)) * rbf_deriv(time_sqrd_mat, tau_gamma)) % B_Region;
    
    arma::vec eigval_B;
    arma::mat eigvec_B;
    
    arma::eig_sym(eigval_B, eigvec_B, B_Region);

    // Block matrices of C(v1, v2)
    arma::mat C_Region = get_cor_mat(kernel_type, dist_sqrd_mat, phi_gamma); 
    arma::mat dC_dphi_gamma = logistic(theta_unrestrict(0)) * get_cor_mat_deriv(kernel_type, dist_sqrd_mat, phi_gamma);
    
    arma::vec eigval_C;
    arma::mat eigvec_C;
    
    arma::eig_sym(eigval_C, eigvec_C, C_Region);
    
    // V matrix.
    arma::vec lambda_inv = 1. / (arma::kron(eigval_C, eigval_B) + 1.);
    arma::mat VInv_r_region = kronecker_mvm(eigvec_C, eigvec_B, lambda_inv % kronecker_mvm(eigvec_C.t(), eigvec_B.t(), r_region)) ;
    
    arma::mat VInv_Z_region = Z_region;
    VInv_Z_region = VInv_Z_region.each_col( [&eigvec_C, &eigvec_B, &lambda_inv](arma::vec& a){ 
      a = kronecker_mvm(eigvec_C, eigvec_B, lambda_inv % kronecker_mvm(eigvec_C.t(), eigvec_B.t(), a) ); 
      });

    
    // log determinant of V
    l1 = arma::sum(arma::log(1./lambda_inv)); 
    
    // log determinant of Z.t() * VInv * Z
    l2 = arma::log_det_sympd(Z_region.t() * VInv_Z_region);
    
    // quadratic form of residuals
    arma::mat qdr = r_region.t() * VInv_r_region;
    l3 = log(qdr(0,0)); 
    
    
    double result = 0.5 * (l1 + l2 + (L*M - length_nu) * l3);
    
    
    // Gradient of component 1: log_det_sympd(Z.t() * VInv * Z)
    arma::vec comp1(3);
    arma::mat HInv = arma::inv_sympd(Z_region.t() * VInv_Z_region);
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
    
    comp3_1 = (L * M - length_nu) * comp3_1;
    
    // w.r.t. nu
    arma::vec comp3_2(length_nu);
    arma::mat comp3_2_temp = -2 * Z_region.t() * VInv_r_region;
    for (int i = 0; i < length_nu; i++){
      comp3_2(i) = comp3_2_temp(i,0)/qdr(0,0);
    }
    comp3_2 = (L * M - length_nu) * comp3_2;
    
    
    // theta gradients
    gradient.head_rows(3) = 0.5 * (comp1 + comp2 + comp3_1);
    
    // nu gradients
    gradient.tail_rows(length_nu) = 0.5 * comp3_2;
    return (result);
  }
};


//' @title Fit intra-regional model using L-BFGS
//' @param theta_init unrestricted initialization of parameters for 1 region
//' @param X_region Data matrix of signals of 1 region
//' @param Z_region fixed-effects design matrix of 1 region
//' @param dist_sqrd_mat Spatial squared distance matrix
//' @param time_sqrd_mat Temporal squared distance matrix
//' @param L Number of voxels 
//' @param M Number of time points
//' @param kernel_type Choice of spatial kernel
//' @return List of 2 components:
//' \item{theta}{estimated intra-regional parameters}
//' \item{nu}{fixed-effect estimate}
//' @export
// [[Rcpp::export]]
Rcpp::List opt_intra(const arma::vec& theta_init,
                     const arma::mat& X_region, 
                     const arma::mat& Z_region,
                     const arma::mat& dist_sqrd_mat, 
                     const arma::mat& time_sqrd_mat,
                     int L, int M, 
                     std::string kernel_type) {
                      
  
  // Read in parameters inits
  arma::mat nu = theta_init.tail_rows(theta_init.n_elem - 3);
  
  // Update basis coefficents
  arma::mat theta_vec(theta_init.n_elem, 1);
  theta_vec.col(0) = theta_init;
  
  // Construct the objective function.
  OptIntra opt_intra(X_region,
                     Z_region, 
                     dist_sqrd_mat, 
                     time_sqrd_mat,
                     L, M, nu, 
                     kernel_type);
                           
  
  // Create the L_BFGS optimizer with default parameters.
  ens::L_BFGS optimizer(20); // L-BFGS optimizer with 10 memory points
  // Maximum number of iterations
  optimizer.MaxIterations() = 100;
  optimizer.MaxLineSearchTrials() = 10;
  // Relative error
  optimizer.MinGradientNorm() = 1e-4;
  
  // Run the optimization
  optimizer.Optimize(opt_intra, theta_vec);
  arma::vec theta = softplus(theta_vec.head_rows(3));
  nu = theta_vec.tail_rows(Z_region.n_cols);
  
  // Return
  return Rcpp::List::create(Rcpp::Named("theta") = theta,
                            Rcpp::Named("nu") = nu);
}



/*****************************************************************************
 Inter-regional model
*****************************************************************************/

class OptInter
{
  
private:
  const arma::mat& X; // The data matrix from 2 regions.
  const arma::mat& Z; // The design matrix.
  int L_1; // Number of voxels in region 1
  int L_2; // Number of voxels in region 2
  int M; // Number of time points 
  const arma::mat& block_region_1; //Spatial kernel matrix for region 1
  const arma::mat& block_region_2; //Spatial kernel matrix for region 2
  const arma::mat& time_sqrd_mat;
public:
  // Construct the object with the given data
  OptInter(const arma::mat& X, 
           const arma::mat& Z, 
           int L_1, int L_2, int M, 
           const arma::mat& block_region_1, 
           const arma::mat& block_region_2, 
           const arma::mat& time_sqrd_mat) :
  X(X), Z(Z), L_1(L_1), L_2(L_2), M(M), 
  block_region_1(block_region_1), 
  block_region_2(block_region_2),
  time_sqrd_mat(time_sqrd_mat) {}
  
  // Compute both objective function and its gradient
  double EvaluateWithGradient(const arma::mat &theta_unrestrict,
                              arma::mat &gradient)
  {
    // Parameter list:
    // rho, tauEta, kEta, nugget, mu1, mu2
    // Transform unrestricted parameters to original forms.
    double theta [4];
    theta[0] = sigmoid_inv(theta_unrestrict(0), -1, 1);
    
    for (int i = 1; i < 4; i++) {
      theta[i] = softplus(theta_unrestrict(i));
    }
    arma::mat mu(2,1);
    mu(0,0) = theta_unrestrict(4); 
    mu(1,0) = theta_unrestrict(5);
    
    // Create necessary components in likelihood evaluation.
    int N = X.n_rows;
    arma::mat r = X - Z * mu;
    arma::mat I = arma::eye(N,N);
    
    // log-likelihood components
    double l1, l2, l3;
    int M_L1 = M*L_1;
    int M_L2 = M*L_2;
    
    // Construct the Sigma_alpha matrix.
    
    // A Matrix
    arma::mat dAt_dk_eta = rbf(time_sqrd_mat, theta[1]);
    arma::mat At = theta[2] * dAt_dk_eta + theta[3] * arma::eye(M,M);
    arma::mat dAt_dnugget = arma::eye(M,M);
    arma::mat dAt_dtau_eta = rbf_deriv(time_sqrd_mat, theta[1]);
    
    arma::mat M_12 = arma::repmat(theta[0]*At, L_1, L_2);
    
    arma::mat M_11 = block_region_1 + arma::repmat(At, L_1, L_1) + arma::eye(L_1*M, L_1*M);
    arma::mat M_22 = block_region_2 + arma::repmat(At, L_2, L_2) + arma::eye(L_2*M, L_2*M);
    
    
    arma::mat M_22_chol = arma::chol(M_22, "lower");
    arma::mat M_22_inv = R_inv_B(M_22_chol, arma::eye(M_L2, M_L2));
    
    arma::mat M_12_M_22_inv = arma::repmat(M_12.head_rows(M) * M_22_inv, L_1, 1);
    arma::mat C_11_chol = arma::chol(M_11 - M_12_M_22_inv * M_12.t(), "lower");
    arma::mat C_11_inv = R_inv_B(C_11_chol, arma::eye(M_L1, M_L1));
    
    arma::mat VInv_r_1 = C_11_inv * (r.head_rows(M_L1) - M_12_M_22_inv*r.tail_rows(M_L2));
    arma::mat VInv_r_2 = M_22_inv * (r.tail_rows(M_L2) - M_12.t() * VInv_r_1);
    
    arma::mat VInv_Z_1 = C_11_inv * (Z.head_rows(M_L1) - M_12_M_22_inv*Z.tail_rows(M_L2));
    arma::mat VInv_Z_2 = M_22_inv * (Z.tail_rows(M_L2) - M_12.t() * VInv_Z_1);
    
    // VInv matrix.
    arma::mat VInv_Z = arma::join_cols(VInv_Z_1, VInv_Z_2);
    arma::mat VInv_r = arma::join_cols(VInv_r_1, VInv_r_2);
    arma::mat qdr = r.t() * VInv_r;
    
    l1 = 2*arma::sum(arma::log(M_22_chol.diag())) + 2*arma::sum(arma::log(C_11_chol.diag()));
    l2 = arma::log_det_sympd(Z.t() * VInv_Z);
    l3 = log(qdr(0,0));
    
    double result = (0.5 * (l1 + l2 + ((L_1+L_2)*M-2)*l3));
    
    // Get gradient for each component of the REML function.
    
    double rho_deriv = sigmoid_inv_derivative(theta_unrestrict(0), -1, 1);
    
    arma::mat rhoDeriv_At = arma::repmat(At, L_1, L_2);
    arma::mat J_L1(L_1, L_1, arma::fill::ones);
    arma::mat J_L2(L_2, L_2, arma::fill::ones);
    arma::mat leftBlock = arma::join_vert(arma::join_horiz(J_L1, arma::mat(L_1, L_2, arma::fill::value(theta[0]))),
                                          arma::join_horiz(arma::mat(L_2, L_1, arma::fill::value(theta[0])), J_L2));
    
    // Gradient of component 1: log_det_sympd(Z.t() * VInv * Z)
    arma::vec comp1(4);
    arma::mat HInv = arma::inv_sympd(Z.t() * VInv_Z);
    arma::mat dV_rho_VInvZ = arma::join_cols(rhoDeriv_At * VInv_Z.tail_rows(M_L2), 
                                             rhoDeriv_At.t() * VInv_Z.head_rows(M_L1));
    
    comp1(0) = arma::trace(-HInv * VInv_Z.t() * dV_rho_VInvZ);
    comp1(1) = arma::trace(-HInv * VInv_Z.t() * kronecker_mmm(leftBlock, dAt_dtau_eta, VInv_Z));
    comp1(2) = arma::trace(-HInv * VInv_Z.t() * kronecker_mmm(leftBlock, dAt_dk_eta, VInv_Z));
    comp1(3) = arma::trace(-HInv * VInv_Z.t() * kronecker_mmm(leftBlock, dAt_dnugget, VInv_Z));
    
    // Gradient of component 2: log_det_sympd(V)
    arma::vec comp2(4);
    arma::mat VInv_12 = -C_11_inv * M_12_M_22_inv;
    arma::mat VInv_22 = M_22_inv + M_12_M_22_inv.t() * (-VInv_12);
    
    comp2(0) = 2 * arma::trace(VInv_12 * rhoDeriv_At);
    
    comp2(1) = arma::trace(C_11_inv * arma::repmat(dAt_dtau_eta, L_1, L_1)) 
      + 2 * arma::trace(VInv_12 * arma::repmat(theta[0]*dAt_dtau_eta, L_1, L_2)) 
      + arma::trace(VInv_22 * arma::repmat(dAt_dtau_eta, L_2, L_2));
      
    comp2(2) = arma::trace(C_11_inv * arma::repmat(dAt_dk_eta, L_1, L_1)) 
      + 2 * arma::trace(VInv_12 * arma::repmat(theta[0]*dAt_dk_eta, L_1, L_2)) 
      + arma::trace(VInv_22 * arma::repmat(dAt_dk_eta, L_2, L_2));
        
    comp2(3) = arma::trace(C_11_inv * arma::repmat(dAt_dnugget, L_1, L_1)) 
      + 2 * arma::trace(VInv_12 * arma::repmat(theta[0]*dAt_dnugget, L_1, L_2)) 
      + arma::trace(VInv_22 * arma::repmat(dAt_dnugget, L_2, L_2));
          
    // Gradient of component 3: r.t() * VInv * r
    
    // w.r.t. theta.
    arma::vec comp3_1(4);
    arma::mat qdr_rho = -VInv_r.t() * arma::join_cols(rhoDeriv_At * VInv_r.tail_rows(M_L2), 
                                                      rhoDeriv_At.t() * VInv_r.head_rows(M_L1));
    arma::mat qdr_tauEta = -VInv_r.t() * kronecker_mvm(leftBlock, dAt_dtau_eta, VInv_r);
    arma::mat qdr_kEta = -VInv_r.t() * kronecker_mvm(leftBlock, dAt_dk_eta, VInv_r);
    arma::mat qdr_nugget = -VInv_r.t() * kronecker_mvm(leftBlock, dAt_dnugget, VInv_r);
    
    comp3_1(0) = qdr_rho(0,0)/qdr(0,0);
    comp3_1(1) = qdr_tauEta(0,0)/qdr(0,0);
    comp3_1(2) = qdr_kEta(0,0)/qdr(0,0);
    comp3_1(3) = qdr_nugget(0,0)/qdr(0,0);
    
    comp3_1 = ((L_1+L_2) * M - 2) * comp3_1;
    arma::vec gradient_wrt_theta = 0.5 * (comp1 + comp2 + comp3_1);
    
    // w.r.t. mu.
    arma::vec comp3_2(2);
    arma::mat comp3_2_temp = -2 * Z.t() * VInv_r;
    comp3_2(0) = comp3_2_temp(0)/qdr(0,0);
    comp3_2(1) = comp3_2_temp(1)/qdr(0,0);
    comp3_2 = 0.5 * ((L_1+L_2) * M - 2) * comp3_2;
    
    // Final gradient
    gradient(0) =  rho_deriv * gradient_wrt_theta(0);
    gradient(1) =  logistic(theta_unrestrict(1)) * gradient_wrt_theta(1);
    gradient(2) =  logistic(theta_unrestrict(2)) * gradient_wrt_theta(2);
    gradient(3) =  logistic(theta_unrestrict(3)) * gradient_wrt_theta(3);
    gradient(4) =  comp3_2(0);
    gradient(5) =  comp3_2(1);
    
    return result;
  }
};


//' @title Fit inter-regional model using L-BFGS
//' @param theta_init unrestricted initialization of parameters  for inter-regional model
//' @param X Data matrix of signals of 2 regions
//' @param Z fixed-effects design matrix of 2 regions
//' @param L_1 Number of voxels in region 1
//' @param L_2 Number of voxels in region 2
//' @param M Number of time points from each voxel
//' @param dist_sqrdMat_1 Block component for that region 1
//' @param dist_sqrdMat_2 Block component for that region 2
//' @param kernel_type Choice of spatial kernel
//' @return List of 3 components:
//' \item{theta}{estimated inter-regional parameters}
//' \item{asymptotic_var}{asymptotic variance of transformed correlation coefficient}
//' \item{rho_transformed}{Fisher transformation of correlation coefficient}
//' @export
// [[Rcpp::export]]
Rcpp::List opt_inter(const arma::vec& theta_init,
                     const arma::mat& X,
                     const arma::mat& Z,
                     const arma::mat& dist_sqrdMat_1, 
                     const arma::mat& dist_sqrdMat_2, 
                     const arma::mat& time_sqrd_mat, 
                     const arma::vec& gamma_vec, 
                     std::string kernel_type) {
                               
  // Read in parameters inits
  arma::mat theta_vec(6, 1);
  theta_vec.col(0) = theta_init;
  
  const arma::mat block_region_1 = arma::kron(get_cor_mat(kernel_type, dist_sqrdMat_1, gamma_vec(0)),
                                      gamma_vec(2) * get_cor_mat("rbf", time_sqrd_mat, gamma_vec(1)));
  
  const arma::mat block_region_2 = arma::kron(get_cor_mat(kernel_type, dist_sqrdMat_2, gamma_vec(3)),
                                      gamma_vec(5) * get_cor_mat("rbf", time_sqrd_mat, gamma_vec(4)));
  
  int L1 = dist_sqrdMat_1.n_cols;
  int L2 = dist_sqrdMat_2.n_cols;
  int M = time_sqrd_mat.n_cols;
  // Construct the objective function.
  OptInter opt_schur_rho_f(X, Z, L1, L2, M, block_region_1, block_region_2, time_sqrd_mat);
  
  // Create the L_BFGS optimizer with default parameters.
  ens::L_BFGS optimizer(10); // L-BFGS optimizer with 10 memory points
  // Maximum number of iterations
  optimizer.MaxIterations() = 50;
  optimizer.MaxLineSearchTrials() = 10;
  // Relative error
  optimizer.MinGradientNorm() = 1e-4;
  
  // Run the optimization
  optimizer.Optimize(opt_schur_rho_f, theta_vec);
  
  //Return rho
  theta_vec(0) = sigmoid_inv(theta_vec(0), -1 ,1);
  theta_vec(1) = softplus(theta_vec(1));
  theta_vec(2) = softplus(theta_vec(2));
  theta_vec(3) = softplus(theta_vec(3));

  Rcpp::List asymp_var = asymptotic_variance(block_region_1,
                                             block_region_2, 
                                             time_sqrd_mat,
                                             L1, L2, M,
                                             theta_vec(2), 
                                             theta_vec(1), 
                                             theta_vec(3),
                                             theta_vec(0),
                                             Z,
                                             kernel_type);
  
  return Rcpp::List::create(Rcpp::Named("theta") = theta_vec,
                            Rcpp::Named("asymptotic_var") = asymp_var[0],
                            Rcpp::Named("rho_transformed") = asymp_var[1]);
  
}