# This function generates an array of d beta values over n time points to be used in a time series
# generating model

# 15/10/20 Andrew Melville

import pandas as pd
import numpy as np

class beta_generator:
    
    
    def __init__(self, beta_type, number, dimensions):
        
        # Initialise class variables determining vector size, dimensions, and beta generator type
        self.beta_type = beta_type
        self.n = number
        self.d = dimensions
        
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
    



## Notes


# Need to add more flexible beta functions using combinations of
# sin and cos periodic functions