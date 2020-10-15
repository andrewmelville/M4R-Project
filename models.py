# This function takes as inputs a d dimensional brownian random walk, some function of beta, and some
# appropriate random noise variance parameter, and models the continuous time series of a commoditie's
# futures

# 15/10/20 Andrew Melville

import pandas as pd
import numpy as np

from brownian_motion import walk_generator
from beta_functions import beta_generator

def linear_model_generator(num_obs = 1000, num_cofactors = 3, noise = 1):
    
    # Generate beta variables
    beta = beta_generator(n = num_obs, d = num_cofactors)
    
    # Genearte covariates through brownian motion generator function and taking difference to get
    # returns data (which is approx normally distributed)
    covariates = walk_generator(n = num_obs, d = num_cofactors).diff()
    
    # Create time series model using covariates
    # y = XB + E
    output = (covariates * beta).values.sum(axis = 1) + np.random.normal(loc = 0, scale = noise, size = num_obs)
    output = pd.DataFrame(output, index = [l for l in range(num_obs)], columns = ['Y'])
    return output


## Notes


# Need to add ARMA time series model functionality