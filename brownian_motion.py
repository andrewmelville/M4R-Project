import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from plotting_functions import series_plot


def geo_bm(d = 3, n = 100, drift = 0, sigma = 1, initial_range = [0,1]):

    # Geometric Brownian Motion Generator
    # This function generates a d-dimensional random walk, returned as a pandas dataframe
    # 15/10/20 Andrew Melville

   
    # x0 -- Initial starting point for walk (default = 100)
    # d -- number of dimensions of walk (default = 1)
    # n -- length of walk
    # drift -- mean of indpendent increments (default = 0)
    # sigma -- variance (volatility) of independent increments 
    
    # Define intial conditions for walk
    S0 = [(initial_range[1]-initial_range[0]) * random.random() + initial_range[0] for dim in range(d)]
    
    # Create empty dataframes for assignment
    increments_df = pd.DataFrame([], index = [l for l in range(n)], columns = [dim+1 for dim in range(d)])
    z_df = pd.DataFrame([], index = [l for l in range(n)], columns = [dim+1 for dim in range(d)])
    brownian_df = pd.DataFrame([], index = [l for l in range(n)], columns = [dim+1 for dim in range(d)])
    
    # Loop through each dimension and generate n independent increments
    for dim in range(d):
        
        # Set initial condition
        increments_df[dim+1].iloc[0] = 0
        
        # Generate iid normal increments
        increments_df[dim+1].iloc[1:] = np.random.normal(loc = 0, scale = 1, size = n-1)
         
        # Define Zk dataframe of brownian motion
        z_df[dim+1] = increments_df[dim+1].cumsum(axis = 0)
        
        # Generate final model
        brownian_df[dim+1] = S0[dim] * np.exp(sigma * np.array(z_df[dim+1], dtype=float) + drift * np.array(range(n)), dtype=float)
        # brownian_df[dim+1] = sigma * np.array(z_df[dim+1], dtype=float)
    return brownian_df

def bm_std(d = 3, n = 100, sigma = 1, initial_range = [0,1]):
    
    # x0 -- Initial starting point for walk (default = 100)
    # d -- number of dimensions of walk (default = 1)
    # n -- length of walk
    # drift -- mean of indpendent increments (default = 0)
    # sigma -- variance (volatility) of independent increments 
    
    # Define intial conditions for walk
    S0 = [(initial_range[1]-initial_range[0]) * random.random() + initial_range[0] for dim in range(d)]
    
    # Create empty dataframes for assignment
    increments_df, z_df, brownian_df = [pd.DataFrame([], index = [l for l in range(n)], columns = [dim+1 for dim in range(d)]) for i in range(3)]
    
    # Loop through each dimension and generate n independent increments
    for dim in range(d):
        
        # Set initial condition
        increments_df[dim+1].iloc[0] = S0[dim]
        
        # Generate iid normal increments
        increments_df[dim+1].iloc[1:] = np.random.normal(loc = 0, scale = sigma, size = n-1)
         
        # Define Zk dataframe of brownian motion
        z_df[dim+1] = increments_df[dim+1].cumsum(axis = 0)
    
    # Generate final model
    brownian_df = increments_df.cumsum()
       
    return brownian_df


## Notes

# Need to add ability to control structure between dimensions in the random walk
# using a covariance matrix