# This is where I will bring together all the modules of my code project for integrated testing

# 15/10/20 Andrew Melville
from brownian_motion import walk_generator
from plotting_functions import series_plot
from beta_functions import beta_generator
from plotting_functions import series_plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
from brownian_motion import walk_generator
brown = walk_generator(n=10000, d=100)


series_plot(brown, 'Random Walk Plot', xlab = 'Value', ylab= 'T')

#%%
from beta_functions import beta_generator
test =  beta_generator()
#%%
from models import linear_model_generator
test = linear_model_generator()
#%%
from models import model_generator

test_model = model_generator()

model = test_model.linear_model(num_obs = 10000, num_covariates = 1, beta_type = 'sin_range')
betas = test_model.params
covs = test_model.covariates()
series_plot(test_model.params,'')
#%%

from rolling_functions import Rolling_LR_OneD

reg_oneD = Rolling_LR_OneD()

test = reg_oneD.fit(covs['Noisy'], model, 625)
reg_oneD.beta_plot()
#%%

from rolling_functions import Rolling_LR

reg = Rolling_LR()

reg.fit(covs['Noisy'], model, 625)
reg.beta_plot()
#%%
from models import model_generator
high_freq_model = model_generator()

model = high_freq_model.linear_model(num_obs=10000, num_covariates=30, beta_type='brownian', noise=1)

betas = high_freq_model.params
noisy_covs = high_freq_model.covariates()['Noisy']
true_covs = high_freq_model.covariates()['True']
noise = high_freq_model.noise

from trading_strats import MeanReversion
mean_rev = MeanReversion()
test = mean_rev.back_test(model, noisy_covs, noise, chunk_size = 20, lookback = 625)
