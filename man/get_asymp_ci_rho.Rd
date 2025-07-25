% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/asymptotic_inference.R
\name{get_asymp_ci_rho}
\alias{get_asymp_ci_rho}
\title{Compute Asymptotic Confidence Interval for Rho}
\usage{
get_asymp_ci_rho(
  theta,
  level,
  asympvar_rho = NULL,
  region1_info = NULL,
  region2_info = NULL
)
}
\arguments{
\item{theta}{Named numeric vector containing "rho".}

\item{level}{The confidence level for the interval (e.g., 0.95 for
95\% confidence interval).}

\item{asympvar_rho}{Numeric value or NULL. Pre-computed asymptotic variance
for rho. If NULL, variance will be computed using region information.}

\item{region1_info}{List or NULL. Information about the first region used
for variance calculation when asympvar_rho is NULL.}

\item{region2_info}{List or NULL. Information about the second region used
for variance calculation when asympvar_rho is NULL.}
}
\value{
A numeric vector containing the lower and upper bounds of the
  confidence interval for the rho parameter. The confidence interval is computed
 on the Fisher Z scale using the delta method and then transformed back to the correlation scale.
}
\description{
Get Asymptotic Confidence Interval for Rho Parameter
}
\details{
Computes asymptotic confidence intervals for the rho parameter using
asymptotic variance estimates. The function supports different methods
for variance calculation based on provided region information.
}
\seealso{
\code{\link{get_asymp_var_rho}} for computing asymptotic variance
}
