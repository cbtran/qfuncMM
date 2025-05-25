# Lookup covariance setting by name
cov_setting_dict <- function(name) {
  switch(name,
    noisy = {
      return(0L)
    },
    noiseless = {
      return(2L)
    },
    {
      stop(paste("Invalid covariance specification:", name))
    }
  )
}
