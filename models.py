# This function takes as inputs a d dimensional brownian random walk, some function of beta, and some
# appropriate random noise variance parameter, and models the continuous time series of a commoditie's
# futures

# 15/10/20 Andrew Melville

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plotting_functions import series_plot
from brownian_motion import walk_generator
from beta_functions import beta_generator

class model_generator():
    
    
    ## This class holds functions for generating realisations of models and 
    ## functions for returning information on their specifications
    
    
    def __init__(self):
        
        # Has a linear model been generated yet? No.
        self.lin_model_made = False

        # Initialise variables that may be asked for as output
        self.beta_type = []
        self.noisy_covariates = []
        self.params = []
        self.output = []
        self.true_covariates = []
        self.noise = []
    
    def linear_model(self, beta_type = 'sin_range', num_obs = 1000, num_covariates = 3, noise = 1):
        
        ## Generate an observation of a linear model according to the 
        ## specifications taken as input.
        ## This linear model defaults to 1000 observartions with 3 covariates,
        ## and noise variance of 1.
    
    
        # Generate beta variables
        gen = beta_generator(number = num_obs, dimensions = num_covariates, beta_type = beta_type)
        self.params = gen()

        # Generate output model time series using Brownian Motion generator function
        # and taking difference to get returns data (which is approx normally distributed)
        hold = walk_generator(n = num_obs, d = 1, drift=-0.000002, sigma=0.002, initial_range=[0,1])

        self.output = hold.copy()
        self.output = self.output.diff(periods=1).fillna(method='backfill')
        self.output.iloc[0] = hold.iloc[0]

        
        self.noise = pd.DataFrame([]).reindex_like(self.params)
        
        # Create each covariate time series model using output and beta series
        # commodity = currency * beta + noise
        self.true_covariates = pd.DataFrame([]).reindex_like(self.params)
        self.noisy_covariates = self.true_covariates.copy()
        
        # Loop through each commodity
        for commod in self.params:
            
            # Generate and save vector of noise 
            self.noise[commod] = np.random.normal(loc = 0, scale = noise, size = num_obs)
            
            # Generate and save true covariates and noisy covariates
            self.true_covariates[commod] = (self.output[1].iloc[1:] * self.params[commod])
            self.noisy_covariates[commod] = (self.output[1].iloc[1:] * self.params[commod]) + self.noise[commod]
        
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
            
            self.covariates_dict = {'True':[],'Noise':[]}
            self.covariates_dict['True'] = self.true_covariates
            self.covariates_dict['Noisy'] = self.noisy_covariates
            
            return self.covariates_dict
        else:
            print('No linear model generated yet')
        
    def model_plot(self):
        
        ## Plot time series of model and its produced covariates coefficients
        
        if self.lin_model_made == True:
            # Plot beta time series
            plt.figure(figsize=(20,10))
              
            # plt.plot(selfoutput.cumsum(), lw=1, label=col)
            series_plot(self.output.cumsum(),'Currency Model')
            # plt.plot(self.output.cumsum(), 'b', lw=1, label='Currency')
            # plt.xlabel('Index')
            # plt.ylabel('Price')
            # plt.title('Currency Price Compared to Generated Commodities PRices')
            # plt.legend(loc=3)
            # plt.show()
        else:
            print('Please fit a regression first!')
            
    def beta_plot(self):
        
        ## Plot time series of beta coefficients
        
        if self.lin_model_made == True:
            
            # Plot beta time series
            plt.figure(figsize=(20,10))
            for col in self.params.columns:    
                plt.plot(self.params[col], lw=1, label=col)
           
            # plt.plot(self.output.cumsum(), 'b', lw=2, label='')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Beta Coefficients that generated the model')
            plt.legend(loc=3)
            plt.show()
        else:
            print('Please fit a regression first!')
            
    def noisy_covariates_plot(self):
        
        ## Plot time series of generated covariates and added noise
        
        if self.lin_model_made == True:
            
            # Plot beta time series
            plt.figure(figsize=(20,10))
            for col in self.noisy_covariates.columns:    
                plt.plot(self.noisy_covariates, lw=1, label=col)
           
            # plt.plot(self.output.cumsum(), 'b', lw=2, label='')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Noisy Covariates generated by the Beta coefficients')
            plt.legend(loc=3)
            plt.show()
        else:
            print('Please fit a regression first!')
    
    
    def true_covariates_plot(self):
        
        ## Plot time series of generated covariates and added noise
        
        if self.lin_model_made == True:
            
            # Plot beta time series
            plt.figure(figsize=(20,10))
            for col in self.true_covariates.columns:    
                plt.plot(self.true_covariates, lw=1, label=col)
           
            # plt.plot(self.output.cumsum(), 'b', lw=2, label='')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('True Covariates generated by the Beta coefficients')
            plt.legend(loc=3)
            plt.show()
        else:
            print('Please fit a regression first!')

## Notes


# Need to add ARMA time series model functionality