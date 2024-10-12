# Lookup covariance setting by name
cov_setting_dict <- function(name) {
  switch(name,
    noisy = {
      return(0L)
    },
    diag_time = {
      return(1L)
    },
    noiseless = {
      return(2L)
    },
    noiseless_profiled = {
      return(3L)
    },
    {
      stop(paste("Invalid covariance specification:", name))
    }
  )
}
