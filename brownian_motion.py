import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotting_functions import series_plot

#%% Brownian Motion Generator

# This function generates a d-dimensional random walk, returned as a pandas dataframe"

# 15/10/20 Andrew Melville

def walk_generator(x0 = 0, d = 3, n = 100, drift = 0, sigma = 1):
   
    # x0 -- Initial starting point for walk (default = 100)
    # d -- number of dimensions of walk (default = 1)
    # n -- length of walk
    # drift -- mean of indpendent increments (default = 0)
    # sigma -- variance (volatility) of independent increments 
    
    
    # Create empty dataframe for assignment
    increments_df = pd.DataFrame([], index = [l for l in range(n)], columns = [m for m in range(d)])
    
    # Loop through each dimension and generate n independent increments
    for j in range(d):
        increments_df[j] = np.random.normal(loc = drift, scale = sigma, size = n)

    # Define the random walk as the cumulative sum of the increments
    # Note, may need to rescale thisw appropriately for CLT purposes
    brownian_df = increments_df.cumsum(axis = 0)
    
    return brownian_df

## Notes


# Need to add ability to control structure between dimensions in the random walk
# using a covariance matrix