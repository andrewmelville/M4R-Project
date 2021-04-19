# This is where I will bring together all the modules of my code project for integrated testing

# 15/10/20 Andrew Melville
# from brownian_motion import walk_generator
from plotting_functions import series_plot, pred_truth_vis
from beta_functions import beta_generator
from models import model_generator
from rolling_functions import Rolling_LR_OneD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
from brownian_motion import geo_bm, bm_std
# brown = geo_bm(n=10000, d=1, drift = 0.000001, sigma=0.005)
brown2 = geo_bm(d=1, n=10000, sigma=0.0035, initial_range=[-0.1,0.6])

series_plot(brown2, 'Random Walk Plot')

#%%
from ARMA import ARMAmodel
from statsmodels.tsa.stattools import acf
arma = ARMAmodel()
l=1
x=arma(n=10000,
      phi=[.98],
      theta=[(i+1)**(-2) for i in range(60)],
      sigma=0.001,
      burnin=10000)

plt.figure(figsize=(15,10))
plt.xlim((0,100))
plt.plot(acf(x[0],fft=True, nlags=100))


series_plot(x,'')
#%%

from models import model_generator
test_model = model_generator()

model = test_model.linear_model(num_obs=10000,
                                num_covariates=1, 
                                beta_type='bm_std',
                                beta_sigma=0.0035,
                                noise=0.0001)
# betas = test_model.params
covs = test_model.covariates()

# test_model.beta_plot()
# test_model.model_plot()
# test_model.noisy_covariates_plot()
# test_model.true_covariates_plot()

plt.figure(figsize=(20,10))
plt.plot(np.exp([i for i in covs['Noisy'][1]]).cumprod()), plt.plot(np.exp([i for i in covs['True'][1]]).cumprod())
# plt.scatter(covs['Noisy'][1][1:], covs['True'][1][1:])
#%%
from rolling_functions import Rolling_LR_OneD
t = 1
reg_oneD = Rolling_LR_OneD()
reg_oneD.fit((1-t)*covs['True'][1:] + t*covs['Noisy'][1:], model, 2)
# reg_oneD.beta_plot()
series_plot(test_model.params.join(reg_oneD.beta_df), '', legend=True)
# plt.scatter(covs['Noisy'][1][600:], covs['Noisy'][1][600:] - reg_oneD.pred_ts['Prediction'].dropna())
# pred_truth_vis(covs['Noisy'][1][200:], [i for i in (covs['Noisy'][1][200:]-reg_oneD.pred_ts['Prediction'].dropna())])

# preds = reg_oneD.pred_series()
# pred_truth_vis(covs['Noisy'][1], preds)
#%%
from trading_strats import MeanReversion
from models import model_generator

high_freq_model = model_generator()
cur_ret = high_freq_model.linear_model(num_obs=10000, 
                                       num_covariates=30, 
                                       beta_type='cor_bb', 
                                       beta_sigma=0.0035, 
                                       noise=0.0001,
                                       t=0.6)
betas = high_freq_model.params
noisy_covs = high_freq_model.covariates()['Noisy']
true_covs = high_freq_model.covariates()['True']
noise = high_freq_model.arma_noise
# high_freq_model.beta_plot()
#%%
from trading_strats import MeanReversion
mean_rev = MeanReversion()
test = mean_rev.back_test(cur_ret,
                          true_covs,
                          noisy_covs, 
                          chunk_size=25, 
                          lookback=120, 
                          noise_props=[0,1/3,2/3,1], 
                          plot=True, 
                          pos_ratio=1/3)
# mean_rev.beta_plot()
# high_freq_model.beta_plot()
#%%
plt.figure(figsize=(20,10))

for commod in high_freq_model.covariates()['Noisy']:
    plt.plot(np.exp([i for i in high_freq_model.covariates()['True'][commod]]).cumprod())

plt.plot()
#%%
from models import model_generator
from trading_strats import MeanReversion
profit_loss_vec =[]

def avg_back_test(n):
    
    for i in range(n):
        high_freq_model = model_generator()
        cur_ret = high_freq_model.linear_model(num_obs=10000, num_covariates=30, beta_type='cor_bb', beta_sigma=0.0035, noise=0.0001, t=0.6)
        betas = high_freq_model.params
        noisy_covs = high_freq_model.covariates()['Noisy']
        true_covs = high_freq_model.covariates()['True']
        noise = high_freq_model.noise
         
        mean_rev = MeanReversion()
        test = mean_rev.back_test(cur_ret, true_covs, noisy_covs, chunk_size = 25, lookback = 120, noise_props=[1])
        profit_loss_vec.append(mean_rev.PL_curve_df["100.0%"].iloc[-1])
    
        print("Test {}/{}".format(i+1, n))
        
    print("Average Profit: {:.2f}".format(np.mean(profit_loss_vec)))
    print("Standard Deviation of Profit: {:.2f}".format(np.std(profit_loss_vec)))
    
avg_back_test(2)
# high_freq_model.beta_plot()
# mean_rev.Residuals()
#%%
from plotting_functions import pred_truth_vis
print(len(high_freq_model.covariates()['Noisy'][1][600:]),len([i for i in mean_rev.residuals_df[2].dropna()]))
# pred_truth_vis(high_freq_model.covariates()['Noisy'][2][600:-30], [i for i in mean_rev.residuals_df[2].dropna().iloc[30:]])
# plt.plot(np.exp([i for i in high_freq_model.covariates()['Noisy'][1][1:]]).cumprod()),plt.plot(np.exp([i for i in mean_rev.pred_series.iloc[121:,1]]).cumprod())
#%%
# plt.plot(high_freq_model.params[4]), plt.plot(mean_rev.beta_df[4])
series_plot(high_freq_model.params,'True Beta Values')
series_plot(mean_rev.beta_df,'Estimated Beta Values')
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


#%%

from models import model_generator

# Define a generative model to simulate 10000 days of data for 1 currency basket and 30 commodities
model = model_generator()
sim_currency = model.linear_model(num_obs=10000, 
                                       num_covariates=2, 
                                       beta_type='cor_bb', 
                                       beta_sigma=0.0035, 
                                       noise=0.0001,
                                       t=0.6)
sim_betas = model.params
sim_commods = model.covariates()['Noisy']
# model.model_plot()

next_day_returns = pd.DataFrame(sim_commods[1])
#%%
from rolling_functions import LSTM_predictor

lstm_class = LSTM_predictor()
lstm_class.train(next_day_returns[1:], sim_currency[1:], 120)
lstm_class.test()
