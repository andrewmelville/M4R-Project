# This function generates an array of d beta values over n time points to be used in a time series
# generating model

# 15/10/20 Andrew Melville

import pandas as pd
from brownian_motion import geo_bm, bm_std
import numpy as np

class beta_generator:
    
    
    def __init__(self, beta_type, number, dimensions, freq = 10, noise = 0.0035):
        
        # Initialise class variables determining vector size, dimensions, and beta generator type
        self.beta_type = beta_type
        self.n = number
        self.d = dimensions
        self.freq = freq
        self.noise = noise
        
        # Initialise empty beta array and array of linespace to be operated on
        self.beta_df = pd.DataFrame([], index = [l for l in range(self.n)], columns = [m for m in range(self.d)])
        self.line = np.linspace(0,self.n,self.n)
        
        
    def __call__(self):
        
        # Call the appropriate generation function
        if self.beta_type == 'sin_range':
            return self.sin_range()
        elif self.beta_type == 'sin_correlated':
            return self.sin_correlated()
        elif self.beta_type == 'linear':
            return self.linear()
        elif self.beta_type == 'high_freq':
            return self.high_freq()
        elif self.beta_type == 'geo_bm':
            return self.brownian()
        elif self.beta_type == 'bm_std':
            return self.brownian()
        elif self.beta_type == 'constant':
            return self.constant()
    
    
    def sin_range(self):
        
        n, d, line = self.n, self.d, self.line
        
        # Loop through dimensions and generate the same beta vector for each
        for j in range(d):
            
            # Generate random periodic function for each beta
            self.beta_df[j] = np.sin((2*np.pi*line)*(j+1)/n)
        
        return self.beta_df
    
    
    def sin_correlated(self):
        
        d, line = self.d, self.line
        
        # Loop through dimensions and generate the same beta vector for each
        for j in range(d):
            
            # Generate standard sin periodic function for each beta
            self.beta_df[j] = np.sin((2*np.pi*line))
        
        return self.beta_df
    
    
    def linear(self):        
                
        d, line = self.d, self.line
        
        # Loop through dimensions and generate the same beta vector for each
        for j in range(d):
            
            # Generate random periodic function for each beta
            self.beta_df[j] = 2*line
        
        return self.beta_df

    def high_freq(self):
        
        n, d, line, freq = self.n, self.d, self.line, self.freq
        
        # Loop through dimensions and generate the same beta vector for each
        for j in range(d):
            
            # Generate standard sin periodic function for each beta
            self.beta_df[j] = np.sin((freq * 2 * np.pi * line) / n)
        
        return self.beta_df
    
    def brownian(self):
        
        n, d = self.n, self.d
        
        self.beta_df = bm_std(d=d, n=n, sigma=self.noise, initial_range=[-0.1,0.6])
    
        return self.beta_df

    def constant(self):
        
        n, d = self.n, self.d
        
        self.beta_df = bm_std(d=d, n=n, sigma=self.noise, initial_range=[-0.1,0.6])
    
        return self.beta_df


## Notes


# Need to add more flexible beta functions using combinations of
# sin and cos periodic functions