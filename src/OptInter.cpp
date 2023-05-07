#include "OptInter.h"
#include <math.h>
#include "rbf.h"
#include "helper.h"

/*****************************************************************************
 Inter-regional model
*****************************************************************************/

OptInter::OptInter(const arma::mat& dataRegionPair, 
                   const arma::mat& design, 
                   int numVoxelRegion1,
                   int numVoxelRegion2,
                   int numTimePt, 
                   const arma::mat& spatialKernelRegion1, 
                   const arma::mat& spatialKernelRegion2, 
                   const arma::mat& timeSqrd)
    : dataRegionPair_(dataRegionPair),
      design_(design),
      numVoxelRegion1_(numVoxelRegion1),
      numVoxelRegion2_(numVoxelRegion2),
      numTimePt_(numTimePt), 
      spatialKernelRegion1_(spatialKernelRegion1), 
      spatialKernelRegion2_(spatialKernelRegion2),
      timeSqrd_(timeSqrd)
{
}

  
  // Compute both objective function and its gradient
double OptInter::EvaluateWithGradient(
    const arma::mat &theta_unrestrict, arma::mat &gradient)
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
int N = dataRegionPair_.n_rows;
arma::mat r = dataRegionPair_ - design_ * mu;
arma::mat I = arma::eye(N,N);

// log-likelihood components
double l1, l2, l3;
int M_L1 = numTimePt_*numVoxelRegion1_;
int M_L2 = numTimePt_*numVoxelRegion2_;

// Construct the Sigma_alpha matrix.

// A Matrix
arma::mat dAt_dk_eta = rbf(timeSqrd_, theta[1]);
arma::mat At = theta[2] * dAt_dk_eta + theta[3] * arma::eye(numTimePt_, numTimePt_);
arma::mat dAt_dnugget = arma::eye(numTimePt_, numTimePt_);
arma::mat dAt_dtau_eta = rbf_deriv(timeSqrd_, theta[1]);

arma::mat M_12 = arma::repmat(theta[0]*At, numVoxelRegion1_, numVoxelRegion2_);

arma::mat M_11 = spatialKernelRegion1_ +
    arma::repmat(At, numVoxelRegion1_, numVoxelRegion1_) +
    arma::eye(numVoxelRegion1_ * numTimePt_, numVoxelRegion1_ * numTimePt_);
arma::mat M_22 = spatialKernelRegion2_ +
    arma::repmat(At, numVoxelRegion2_, numVoxelRegion2_) +
    arma::eye(numVoxelRegion2_ * numTimePt_, numVoxelRegion2_ * numTimePt_);


arma::mat M_22_chol = arma::chol(M_22, "lower");
arma::mat M_22_inv = R_inv_B(M_22_chol, arma::eye(M_L2, M_L2));

arma::mat M_12_M_22_inv = arma::repmat(M_12.head_rows(numTimePt_) * M_22_inv, numVoxelRegion1_, 1);
arma::mat C_11_chol = arma::chol(M_11 - M_12_M_22_inv * M_12.t(), "lower");
arma::mat C_11_inv = R_inv_B(C_11_chol, arma::eye(M_L1, M_L1));

arma::mat VInv_r_1 = C_11_inv * (r.head_rows(M_L1) - M_12_M_22_inv*r.tail_rows(M_L2));
arma::mat VInv_r_2 = M_22_inv * (r.tail_rows(M_L2) - M_12.t() * VInv_r_1);

arma::mat VInv_Z_1 = C_11_inv * (design_.head_rows(M_L1) - M_12_M_22_inv*design_.tail_rows(M_L2));
arma::mat VInv_Z_2 = M_22_inv * (design_.tail_rows(M_L2) - M_12.t() * VInv_Z_1);

// VInv matrix.
arma::mat VInv_Z = arma::join_cols(VInv_Z_1, VInv_Z_2);
arma::mat VInv_r = arma::join_cols(VInv_r_1, VInv_r_2);
arma::mat qdr = r.t() * VInv_r;

l1 = 2*arma::sum(arma::log(M_22_chol.diag())) + 2*arma::sum(arma::log(C_11_chol.diag()));
l2 = arma::log_det_sympd(design_.t() * VInv_Z);
l3 = log(qdr(0,0));

double result = (0.5 * (l1 + l2 + ((numVoxelRegion1_+numVoxelRegion2_)*numTimePt_-2)*l3));

// Get gradient for each component of the REML function.

double rho_deriv = sigmoid_inv_derivative(theta_unrestrict(0), -1, 1);

arma::mat rhoDeriv_At = arma::repmat(At, numVoxelRegion1_, numVoxelRegion2_);
arma::mat J_L1(numVoxelRegion1_, numVoxelRegion1_, arma::fill::ones);
arma::mat J_L2(numVoxelRegion2_, numVoxelRegion2_, arma::fill::ones);
arma::mat leftBlock = arma::join_vert(arma::join_horiz(J_L1, arma::mat(numVoxelRegion1_, numVoxelRegion2_, arma::fill::value(theta[0]))),
                                        arma::join_horiz(arma::mat(numVoxelRegion2_, numVoxelRegion1_, arma::fill::value(theta[0])), J_L2));

// Gradient of component 1: log_det_sympd(Z.t() * VInv * Z)
arma::vec comp1(4);
arma::mat HInv = arma::inv_sympd(design_.t() * VInv_Z);
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

comp2(1) = arma::trace(C_11_inv * arma::repmat(dAt_dtau_eta, numVoxelRegion1_, numVoxelRegion1_)) 
    + 2 * arma::trace(VInv_12 * arma::repmat(theta[0]*dAt_dtau_eta, numVoxelRegion1_, numVoxelRegion2_)) 
    + arma::trace(VInv_22 * arma::repmat(dAt_dtau_eta, numVoxelRegion2_, numVoxelRegion2_));
    
comp2(2) = arma::trace(C_11_inv * arma::repmat(dAt_dk_eta, numVoxelRegion1_, numVoxelRegion1_)) 
    + 2 * arma::trace(VInv_12 * arma::repmat(theta[0]*dAt_dk_eta, numVoxelRegion1_, numVoxelRegion2_)) 
    + arma::trace(VInv_22 * arma::repmat(dAt_dk_eta, numVoxelRegion2_, numVoxelRegion2_));
    
comp2(3) = arma::trace(C_11_inv * arma::repmat(dAt_dnugget, numVoxelRegion1_, numVoxelRegion1_)) 
    + 2 * arma::trace(VInv_12 * arma::repmat(theta[0]*dAt_dnugget, numVoxelRegion1_, numVoxelRegion2_)) 
    + arma::trace(VInv_22 * arma::repmat(dAt_dnugget, numVoxelRegion2_, numVoxelRegion2_));
        
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

comp3_1 = ((numVoxelRegion1_+numVoxelRegion2_) * numTimePt_ - 2) * comp3_1;
arma::vec gradient_wrt_theta = 0.5 * (comp1 + comp2 + comp3_1);

// w.r.t. mu.
arma::vec comp3_2(2);
arma::mat comp3_2_temp = -2 * design_.t() * VInv_r;
comp3_2(0) = comp3_2_temp(0)/qdr(0,0);
comp3_2(1) = comp3_2_temp(1)/qdr(0,0);
comp3_2 = 0.5 * ((numVoxelRegion1_+numVoxelRegion2_) * numTimePt_ - 2) * comp3_2;

// Final gradient
gradient(0) =  rho_deriv * gradient_wrt_theta(0);
gradient(1) =  logistic(theta_unrestrict(1)) * gradient_wrt_theta(1);
gradient(2) =  logistic(theta_unrestrict(2)) * gradient_wrt_theta(2);
gradient(3) =  logistic(theta_unrestrict(3)) * gradient_wrt_theta(3);
gradient(4) =  comp3_2(0);
gradient(5) =  comp3_2(1);

return result;
}
