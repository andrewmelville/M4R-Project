# This function takes as inputs a d dimensional brownian random walk, some function of beta, and some
# appropriate random noise variance parameter, and models the continuous time series of a commoditie's
# futures

# 15/10/20 Andrew Melville

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plotting_functions import series_plot
from brownian_motion import geo_bm
from beta_functions import beta_generator
from ARMA import ARMAmodel
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
    
    def linear_model(self, beta_type = 'bm_std', beta_sigma = 0.00035, num_obs = 1000, num_covariates = 3, noise = 0.5):
        
        ## Generate an observation of a linear model according to the 
        ## specifications taken as input.
        ## This linear model defaults to 1000 observartions with 3 covariates,
        ## and noise variance of 1.
    
    
        # Generate beta variables
        gen = beta_generator(number=num_obs+1,
                             dimensions=num_covariates,
                             beta_type=beta_type,
                             noise=beta_sigma)
        self.params = gen()

        # Generate output model time series using Brownian Motion generator function
        # and taking difference to get returns data (which is approx normally distributed)
        hold = geo_bm(n = num_obs+1,
                      d = 1,
                      drift=-0.000002,
                      sigma=0.001,
                      initial_range=[0,1])

        self.output = hold.copy()
        self.output = np.log(self.output.pct_change() + 1)
        self.output.iloc[0] = np.log(hold.iloc[0])
        
        self.arma_noise = pd.DataFrame([]).reindex_like(self.params.drop(index=10000))
        
        # Create each covariate time series model using output and beta series
        # commodity = currency * beta + noise
        self.true_covariates = pd.DataFrame([]).reindex_like(self.params)
        self.noisy_covariates = self.true_covariates.copy()
        
        # Loop through each commodity
        for commod in self.params:
            
            # Generate and save true covariates 
            self.true_covariates[commod].iloc[1:] = (self.output[1].iloc[1:] * self.params[commod])
            
            # Estimate standard deviation of the commodities
            commod_sigma = self.true_covariates[commod][1:].std()
            
            # Generate and save vector of noise proportional to the standard deviation of each commodity
            arma = ARMAmodel()
            self.arma_noise[commod] = arma(n=10000, phi=[0.98], theta=[(i+1)**(-2) for i in range(60)], sigma=0.001, burnin=10000)
            # np.random.normal(loc = 0, scale = noise * commod_sigma, size = num_obs)
            
            # Add price signal to covariates price series
            hold_prices = self.output[1].iloc[1:] * self.params[commod][1:]
            hold_prices = np.exp([i for i in hold_prices]).cumprod()
            hold_prices += self.arma_noise[commod][:]

            # Turn noisy covariate prices into series of log returns once more
            hold_log_returns = hold_prices / hold_prices.shift(1)
            hold_log_returns = np.log([i for i in hold_log_returns])
            self.noisy_covariates[commod].iloc[1:-1] = hold_log_returns[1:]
            
            # Set initial conditions for value of commodities
            initial_cond = np.random.gamma(1,0.2)            

            # Set initial condition to be some random positive value
            self.true_covariates[commod].iloc[0] = float(initial_cond)
            self.noisy_covariates[commod].iloc[0] = float(initial_cond)
            
        # Drop excess days
        self.output.drop(index=10000, inplace=True)
        self.params.drop(index=10000, inplace=True)
            
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
            
            # Plot currency time series
            plt.figure(figsize=(20,10))
            
            plt.title('Simulated Currency Price')
            plt.xlabel('Index')
            plt.ylabel('Price')
    
            plt.plot(np.exp(self.output).cumprod(), label='Currency Model', lw=1)
            
            plt.legend(loc=3)

        else:
            print('Please generate a model first!')
            
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
                plt.plot(np.exp([i for i in self.noisy_covariates[col]]).cumprod(), lw=1, label=col)
           
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
                plt.plot(np.exp([i for i in self.true_covariates[col]]).cumprod(), lw=1, label=col)
           
            # plt.plot(self.output.cumsum(), 'b', lw=2, label='')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('True Covariates generated by the Beta coefficients')
            plt.legend(loc=3)
            plt.show()
        else:
            print('Please fit a regression first!')

def log_returns(prices):
    
    # This function takes as input a dataframe of prices and returns a dataframe of log returns
    return np.log(prices / prices.shift())
    
def prices(log_returns):
    
    # This function takes as input a series of log returns and returns a serires of prices
    return pd.DataFrame(np.exp([i for i in log_returns]).cumprod())
    
    
## Notes


# Need to add ARMA time series model functionality