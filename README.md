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
out_dir <- "./test_out"
qfuncMM_stage1(1, 1, "region1", qfunc_sim_data$data[[1]], qfunc_sim_data$coords[[1]], out_dir)
qfuncMM_stage1(1, 2, "region2", qfunc_sim_data$data[[2]], qfunc_sim_data$coords[[2]], out_dir)
outfile1 <- file.path(out_dir, file.path("qfuncMM_stage1_intra_region_1_1.json"))
outfile2 <- file.path(out_dir, file.path("qfuncMM_stage1_intra_region_1_2.json"))
result <- qfuncMM_stage2_reml(outfile1, outfile2, out_dir)
print(paste("Estimated rho:", round(result$stage2$rho, 3)))
```

Code to reproduce simulation results from the paper is available on github at [https://github.com/cbtran/qfuncMM-reproducible](https://github.com/cbtran/qfuncMM-reproducible).
