# qfuncMM

The `qfuncMM` package estimates the functional connectivity of brain regions. A two-stage procedure is used to fit a mixed model from voxel-level BOLD signals.

Our paper is forthcoming. An older version of the manuscript is available [here](https://arxiv.org/abs/2211.02192).

To install `qfuncMM` from github, type in R console
```r
if (!require("devtools")){
    install.packages("devtools")
}
devtools::install_github("cbtran/qfuncMM")
```

The package includes simulated data for testing. To run the method on the example data, run the following:
```r
library(qfuncMM)
result <- qfuncMM(qfunc_sim_data$data, qfunc_sim_data$coords)
result$rho
```

Code to reproduce simulation results from the paper is available on github at [https://github.com/cbtran/qfuncMM-reproducible](https://github.com/cbtran/qfuncMM-reproducible).
