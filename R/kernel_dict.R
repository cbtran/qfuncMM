# Lookup kernel id by name
kernel_dict <- function(name) {
  switch(name,
    rbf = {
      return(0L)
    },
    matern_1_2 = {
      return(1L)
    },
    matern_3_2 = {
      return(2L)
    },
    matern_5_2 = {
      return(3L)
    },
    {
      stop(paste("Invalid covariance kernel:", name))
    }
  )
}