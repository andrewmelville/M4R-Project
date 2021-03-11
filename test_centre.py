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
from brownian_motion import geo_bm, bm_std
# brown = geo_bm(n=10000, d=1, drift = 0.000001, sigma=0.005)
brown2 = brownian_motion(d=1, n=10000, sigma=0.0035, initial_range=[-0.1,0.6])

series_plot(brown2, 'Random Walk Plot')
#%%
from models import model_generator

test_model = model_generator()

model = test_model.linear_model(num_obs=10000, 
                                num_covariates=1, 
                                beta_type='bm_std',
                                noise=0.0001)
betas = test_model.params
covs = test_model.covariates()
test_model.beta_plot()
# test_model.model_plot()
# test_model.noisy_covariates_plot()
# test_model.true_covariates_plot()
#%%

from rolling_functions import Rolling_LR_OneD

reg_oneD = Rolling_LR_OneD()

reg_oneD.fit(covs['Noisy'], model, 120)
# reg_oneD.beta_plot()

preds = reg_oneD.pred_series()

plt.scatter(covs['Noisy'], covs['Noisy'].values-preds, s=1)


#%%
from models import model_generator
high_freq_model = model_generator()

model = high_freq_model.linear_model(num_obs=10000, num_covariates=30, beta_type='bm_std', noise=0.2)

betas = high_freq_model.params
noisy_covs = high_freq_model.covariates()['Noisy']
true_covs = high_freq_model.covariates()['True']
noise = high_freq_model.noise

from trading_strats import MeanReversion
mean_rev = MeanReversion()
test = mean_rev.back_test(model, noisy_covs, noise, chunk_size = 20, lookback = 300)

# high_freq_model.beta_plot()
# mean_rev.Residuals()
#%%
from models import model_generator

test_model = model_generator()

model = test_model.linear_model(num_obs=10000,
                                num_covariates=1,
                                beta_type='bm_std',
                                noise=0.0002)
betas = test_model.params
covs = test_model.covariates()
series_plot([covs['Noisy'].cumsum() - test_model.noise, covs['Noisy'].cumsum()],'', legend=True)
# series_plot(covs['Noisy'].cumsum(),'')
# test_model.model_plot()
# test_model.noisy_covariates_plot()
# test_model.true_covariates_plot()

#%% Simulating Data
from models import model_generator

# Define a generative model to simulate 10000 days of data for 1 currency basket and 30 commodities
model = model_generator()
sim_currency = model.linear_model(num_obs=10000, num_covariates=1, beta_type='bm_std', noise=0)
sim_betas = model.params
sim_commods = model.covariates()['Noisy']
# model.model_plot()

next_day_returns = pd.DataFrame(sim_commods[1])
# pd.plotting.autocorrelation_plot(np.exp(next_day_returns[1].astype(float)).cumprod())


plt.plot(np.exp(next_day_returns[1].astype(float)).cumprod())
# series_plot(next_day_returns.cumsum(),'')
# CBOT_OATS_df = pd.read_csv('Data/Continuous Futures Series/CBOT Rough Rice.csv',
#                            index_col = 0, skiprows = 0, skipfooter = 1, header = 1, engine = 'python')
# next_day_returns = pd.DataFrame(CBOT_OATS_df['Close'].diff(1))