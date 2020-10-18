# This function takes as inputs a d dimensional brownian random walk, some function of beta, and some
# appropriate random noise variance parameter, and models the continuous time series of a commoditie's
# futures

# 15/10/20 Andrew Melville

import pandas as pd
import numpy as np

from brownian_motion import walk_generator
from beta_functions import beta_generator

class model_generator():
    
    
    ## This class holds functions for generating realisations of models and 
    ## functions for returning information on their specifications
    
    
    def __init__(self):
        
        # Has a linear model been generated yet? No.
        self.lin_model_made = False

        # Initialise variables that may be asked for as output
        self.covariates = []
        self.params = []
        self.output = []
    
    def linear_model(self, num_obs = 1000, num_covariates = 3, noise = 1):
        
        ## Generate an observation of a linear model according to the 
        ## specifications taken as input.
        
        ## This linear model defaults to 1000 observartions with 3 covariates,
        ## and noise variance of 1.
    
        # Generate beta variables
        self.params = beta_generator(n = num_obs, d = num_covariates)
        
        # Genearte covariates through brownian motion generator function and taking difference to get
        # returns data (which is approx normally distributed)
        self.covariates = walk_generator(n = num_obs, d = num_covariates).diff(periods=1).fillna(method='backfill')
        
        # Create time series model using covariates and beta series
        # y = XB + E
        self.output = (self.covariates * self.params).values.sum(axis = 1) + np.random.normal(loc = 0, scale = noise, size = num_obs)
        self.output = pd.DataFrame(self.output, index = [l for l in range(num_obs)], columns = ['Y'])
        
        # Confirm we just made a linear model
        self.lin_model_made = True
        
        return self.output
    
    def params(self):
        
        # Return the series of beta values that the model used to generate the
        # realisation
        
        if self.lin_model_made == True:
            return self.params
        else:
            print('No linear model generated yet')
        
    def covariates(self):
        
        # Return the series of beta values that the model used to generate the
        # realisation
        
        if self.lin_model_made == True:
            return self.covariates
        else:
            print('No linear model generated yet')
        

## Notes


# Need to add ARMA time series model functionality