# This function generates an array of d beta values over n time points to be used in a time series
# generating model

# 15/10/20 Andrew Melville

import pandas as pd
import numpy as np
import sympy as sp

def beta_generator(n = 1000, d = 3):
    
    # Initialise empty beta array
    beta_df = pd.DataFrame([], index = [l for l in range(n)], columns = [m for m in range(d)])
    

    for j in range(d):
        
        # Creat array of linspace
        line = np.array(range(n))
        
        # Generate random periodic function for each beta
        beta_df[j] = np.sin(((j+10)/d)*line)
    
    return beta_df


## Notes


# Need to add more flexible beta functions using combinations of
# sin and cos periodic functions